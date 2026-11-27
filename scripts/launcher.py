"""Open the WSLg service launcher for Kairos and NapCat."""

import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

try:
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("Gdk", "3.0")
    from gi.repository import Gdk, GdkPixbuf, GLib, Gtk
except (ImportError, ValueError):
    raise SystemExit("make launcher requires python3-gi and gir1.2-gtk-3.0") from None

_ROOT = Path(__file__).resolve().parents[1]
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ICONS = {
    "minimize": '<path d="M3.5 8.5h9"/>',
    "maximize": '<rect x="4" y="4" width="8" height="8" rx=".7"/>',
    "restore": '<path d="M6 5V3.5h6.5V10H11"/><rect x="3.5" y="6" width="6.5" height="6.5" rx=".6"/>',
    "close": '<path d="m4.5 4.5 7 7m0-7-7 7"/>',
}
_CSS = b"""
window.kairos-launcher { background: #ffffff; color: #232822; }
headerbar.kairos-titlebar {
    min-height: 44px;
    padding: 0 8px;
    background: #f2f4f0;
    border-bottom: 1px solid #dce1d8;
    box-shadow: none;
    color: #232822;
}
.launcher-title { font-weight: 700; }
button.caption-button {
    min-width: 33px;
    min-height: 29px;
    padding: 0;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    box-shadow: none;
}
button.caption-button:hover { background: #e2e7dd; }
button.caption-button:active { background: #d3dacd; }
button.caption-close:hover { background: #f8dfe0; }
button.caption-close:active { background: #efc8ca; }
button.caption-button:focus { border-color: #516916; }
.service-name { font-weight: 700; font-size: 14px; }
.service-status { color: #535d51; }
.service-status.running { color: #3d5811; }
.service-status.failed { color: #982e37; }
button.service-action { min-width: 58px; }
button.service-action.start {
    background: #4d6313;
    border-color: #4d6313;
    box-shadow: none;
    color: #ffffff;
}
button.service-action.start:hover { background: #3f500f; }
button.service-action.start:disabled { background: #e8ebdf; color: #55604f; }
.log-label { font-weight: 600; }
textview.launcher-log text { background: #f4f6f2; color: #283125; }
"""


def _icon(name: str) -> Gtk.Image:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">'
        '<g fill="none" stroke="#354033" stroke-width="1.35" stroke-linecap="round" '
        f'stroke-linejoin="round">{_ICONS[name]}</g></svg>'
    )
    loader = GdkPixbuf.PixbufLoader.new_with_type("svg")
    loader.write(svg.encode())
    loader.close()
    return Gtk.Image.new_from_pixbuf(loader.get_pixbuf())


class ServiceRow(Gtk.Box):
    def __init__(self, name: str, label: str, toggle: Callable[[str], None]) -> None:
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.label = label
        self.process: subprocess.Popen[str] | None = None
        self.output_closed = False
        self.stopping = False

        details = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        title = Gtk.Label(label=label, xalign=0)
        title.get_style_context().add_class("service-name")
        self.status = Gtk.Label(xalign=0)
        self.status.get_style_context().add_class("service-status")
        details.pack_start(title, False, False, 0)
        details.pack_start(self.status, False, False, 0)
        self.pack_start(details, True, True, 0)

        if name == "napcat":
            webui = Gtk.LinkButton.new_with_label("http://127.0.0.1:6099/webui/", "打开 WebUI")
            webui.set_valign(Gtk.Align.CENTER)
            self.pack_start(webui, False, False, 0)
        self.button = Gtk.Button()
        self.button.set_valign(Gtk.Align.CENTER)
        self.button.get_style_context().add_class("service-action")
        self.button.connect("clicked", lambda _: toggle(name))
        self.pack_start(self.button, False, False, 0)
        self.set_state("已停止")

    def set_state(self, text: str, tone: str = "") -> None:
        self.status.set_text(text)
        status_style = self.status.get_style_context()
        for state in ("running", "failed"):
            if tone == state:
                status_style.add_class(state)
            else:
                status_style.remove_class(state)
        action = "启动" if self.process is None else "停止"
        self.button.set_label(action)
        self.button.get_accessible().set_name(f"{action} {self.label}")
        button_style = self.button.get_style_context()
        if self.process is None:
            button_style.add_class("start")
        else:
            button_style.remove_class("start")
        self.button.set_sensitive(not self.stopping)


class Launcher(Gtk.Window):
    def __init__(self) -> None:
        super().__init__(title="Kairos 启动管理器")
        self.set_default_size(640, 460)
        self.set_size_request(420, 300)
        self.get_style_context().add_class("kairos-launcher")
        self.closing = False
        self.events: Queue[tuple[str, str | None]] = Queue(maxsize=600)

        header = Gtk.HeaderBar(spacing=4, show_close_button=False)
        header.get_style_context().add_class("kairos-titlebar")
        title = Gtk.Label(label="Kairos 启动管理器")
        title.get_style_context().add_class("launcher-title")
        header.set_custom_title(title)
        close = self._caption_button("close", "关闭", self.request_close)
        close.get_style_context().add_class("caption-close")
        self.maximize_button = self._caption_button("maximize", "最大化", self._toggle_maximize)
        minimize = self._caption_button("minimize", "最小化", self.iconify)
        for button in (close, self.maximize_button, minimize):
            header.pack_end(button)
        self.set_titlebar(header)
        self.connect("window-state-event", self._sync_maximize)
        self.connect("delete-event", self._on_delete)

        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12, margin=18)
        self.add(body)
        self.services = {
            "kairos": ServiceRow("kairos", "Kairos", self._toggle_service),
            "napcat": ServiceRow("napcat", "NapCat", self._toggle_service),
        }
        for row in self.services.values():
            body.pack_start(row, False, False, 0)
            body.pack_start(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL), False, False, 0)

        title = Gtk.Label(label="运行日志", xalign=0)
        title.get_style_context().add_class("log-label")
        body.pack_start(title, False, False, 0)
        self.log = Gtk.TextView(editable=False, cursor_visible=False, monospace=True)
        self.log.set_wrap_mode(Gtk.WrapMode.CHAR)
        self.log.set_left_margin(10)
        self.log.set_right_margin(10)
        self.log.set_top_margin(8)
        self.log.set_bottom_margin(8)
        self.log.get_style_context().add_class("launcher-log")
        self.scroller = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
        self.scroller.add(self.log)
        body.pack_start(self.scroller, True, True, 0)
        buffer = self.log.get_buffer()
        buffer.set_text("服务启动后，日志会显示在这里。\n")
        self.log_end = buffer.create_mark(None, buffer.get_end_iter(), False)
        GLib.timeout_add(100, self._poll)

    def _caption_button(self, icon: str, title: str, action: Callable[..., object]) -> Gtk.Button:
        button = Gtk.Button(image=_icon(icon))
        button.set_tooltip_text(title)
        button.get_accessible().set_name(title)
        button.get_style_context().add_class("caption-button")
        button.connect("clicked", lambda _: action())
        return button

    def _toggle_maximize(self) -> None:
        if self.is_maximized():
            self.unmaximize()
        else:
            self.maximize()

    def _sync_maximize(self, _window: Gtk.Window, event: Gdk.EventWindowState) -> bool:
        maximized = bool(event.new_window_state & Gdk.WindowState.MAXIMIZED)
        title = "还原" if maximized else "最大化"
        self.maximize_button.set_image(_icon("restore" if maximized else "maximize"))
        self.maximize_button.set_tooltip_text(title)
        self.maximize_button.get_accessible().set_name(title)
        return False

    def _toggle_service(self, name: str) -> None:
        row = self.services[name]
        if self.closing or row.stopping:
            return
        if row.process is not None:
            self._stop(row)
            return
        try:
            row.process = subprocess.Popen(
                ["make", "--no-print-directory", name],
                cwd=_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
        except OSError as exc:
            row.set_state("启动失败", "failed")
            self._append_log(f"[{name}] {exc}\n")
            return
        row.output_closed = False
        row.set_state("已启动", "running")
        Thread(target=self._read_output, args=(name, row.process), daemon=True).start()

    def _read_output(self, name: str, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        try:
            with process.stdout as output:
                for line in output:
                    self.events.put((name, _ANSI.sub("", line)))
        finally:
            self.events.put((name, None))

    def _stop(self, row: ServiceRow) -> None:
        if row.process is not None and not row.stopping:
            row.stopping = True
            row.set_state("正在停止")
            with suppress(ProcessLookupError):
                os.killpg(row.process.pid, signal.SIGTERM)

    def _poll(self) -> bool:
        lines = []
        for _ in range(180):
            try:
                name, line = self.events.get_nowait()
            except Empty:
                break
            if line is None:
                self.services[name].output_closed = True
            else:
                lines.append(f"[{name}] {line}")
        if lines:
            self._append_log("".join(lines))

        for row in self.services.values():
            if row.process is None or not row.output_closed:
                continue
            if (code := row.process.poll()) is not None:
                failed = code != 0 and not row.stopping
                row.process = None
                row.stopping = False
                row.set_state(
                    f"已退出（{code}）" if failed else "已停止", "failed" if failed else ""
                )
                if self.closing:
                    row.button.set_sensitive(False)
        if self.closing and all(row.process is None for row in self.services.values()):
            self.destroy()
            return False
        return True

    def _append_log(self, text: str) -> None:
        adjust = self.scroller.get_vadjustment()
        follow = adjust.get_value() + adjust.get_page_size() >= adjust.get_upper() - 4
        buffer = self.log.get_buffer()
        buffer.insert(buffer.get_end_iter(), text)
        if buffer.get_line_count() > 1200:
            buffer.delete(buffer.get_start_iter(), buffer.get_iter_at_line(200))
        if follow:
            self.log.scroll_mark_onscreen(self.log_end)

    def request_close(self) -> bool:
        self.closing = True
        for row in self.services.values():
            self._stop(row)
            row.button.set_sensitive(False)
        return False

    def _on_delete(self, _window: Gtk.Window, _event: Gdk.Event) -> bool:
        self.request_close()
        return True


def main() -> int:
    if not Gtk.init_check(None)[0]:
        print("make launcher requires a graphical session such as WSLg", file=sys.stderr)
        return 1
    style = Gtk.CssProvider()
    style.load_from_data(_CSS)
    Gtk.StyleContext.add_provider_for_screen(
        Gdk.Screen.get_default(), style, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    window = Launcher()
    window.connect("destroy", lambda *_: Gtk.main_quit())
    for signum in (signal.SIGINT, signal.SIGTERM):
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signum, window.request_close)
    window.show_all()
    Gtk.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
