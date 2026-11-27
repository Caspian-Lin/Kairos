# Kairos（知时）

Kairos 在一个 Python 进程内托管 1–3 个 QQ Bot，通过 OneBot v11 反向 WebSocket 接收群消息。它使用 PostgreSQL 保存群消息和决策审计，并按每个 Bot 的模型配置生成回复。Stage 1-B（M2）支持双级门控、贴表情、滚动摘要和免 Embedding 群聊记忆。M2.1 增加 64k→16k 压缩、按需工具 Schema、流水线追踪与本地 Admin 管理台。 [当前进度与交接](docs/HANDOFF.md) · [Stage 1 实现规划](docs/Kairos_设计文档/02_Kairos_Stage1_技术栈与2C4G工程实现规划.md)

## 在 WSL2 本地运行

在 WSL2 的 Linux 文件系统中检出仓库，例如 `~/Lab/Kairos`。**请勿放在 `/mnt/c` 等 Windows 挂载目录下。** 准备 Python 3.12 或更高版本、`uv`、原生 PostgreSQL 16 或更高版本、`xvfb` 和 `xauth`。首次启动时，在仓库根目录准备环境和配置副本。

```sh
uv sync --dev
test -e .env || cp .env.sample .env
chmod 600 .env
test -e config/global.yaml || cp config/global.example.yaml config/global.yaml
mkdir -p config/bots
test -e config/bots/1.yaml || cp config/examples/bots/1.yaml config/bots/1.yaml
```

### PostgreSQL

用 `pg_lsclusters` 查看 WSL 中的原生集群，用 `pg_isready -h /var/run/postgresql` 检查 Unix socket。首次初始化开发数据库时，用原生 PostgreSQL 的管理账号创建应用用户和数据库。第一条命令会提示设置数据库密码。

```sh
sudo -u postgres createuser --login --pwprompt kairos
sudo -u postgres createdb --owner=kairos kairos
```

在 2C4G 机器上将 [postgres_2c4g.conf](deploy/postgres_2c4g.conf) 纳入当前集群的 `postgresql.conf`。Ubuntu PostgreSQL 16 的默认集群可将该文件放入 `/etc/postgresql/16/main/conf.d/`。先用 `sudo -u postgres psql -Atc 'SHOW config_file'` 确认实际路径及 `include_dir`，安装后重启该集群以加载内存和连接数参数。

数据库迁移 `0001_m1_core_tables` 创建 `group_messages` 和 `decision_audit_logs`。`0002_m2_memory_and_summaries` 启用 `pg_trgm`，创建 `context_summaries`、`user_impressions` 和 `memory_notes`，并给审计表增加 `action_type`、`target_msg_id`、`emoji_id`。用应用数据库的所有者执行迁移。已有 M1 库通过 `alembic upgrade head` 升级，群消息和既有审计记录会保留。 [M2 迁移脚本](alembic/versions/0002_m2_memory_and_summaries.py)

```sh
uv run --env-file .env alembic upgrade head
uv run --env-file .env alembic current
```

### 配置和启动

用 [.env.sample](.env.sample) 设置本地 `.env`，填写 PostgreSQL 连接串、NapCat 反向 WebSocket 令牌、模型 API 密钥和要启用的 Bot ID。数据库密码中的 URL 保留字符需要编码。`.env` 由 Git 忽略。

`config/bots` 中每个 Bot 的文件名为数字，`1.yaml` 对应 Bot ID `1`。在 `.env` 中设置 `KAIROS_ENABLED_BOTS=1,2` 即可启用以下两份配置。QQ 号、显示名、群白名单和回复模型 `llm.chat` 都写在各自的 YAML 里。两个 Bot 可以引用不同的 `.env` 密钥。

```yaml
# config/bots/1.yaml
self_id: 10000000000000001
nickname: 林小满
persona_prompt: 我是林小满，喜欢聊做饭和游戏。
llm:
  chat:
    provider: google_ai_studio
    api_format: gemini_generate_content
    model: gemini-3.5-flash-lite
    api_key: "${GEMINI_API_KEY}"
    thinking_level: minimal
allowed_groups:
  - 123456789
  - 234567890
debug_groups:
  - 123456789
```

```yaml
# config/bots/2.yaml
self_id: 10000000000000002
nickname: 陈阿呆
persona_prompt: 我是陈阿呆，喜欢聊电脑和电影。
llm:
  chat:
    provider: openai
    api_format: openai_responses
    model: gpt-4o-mini
    api_key: "${BOT_2_OPENAI_API_KEY}"
allowed_groups:
  - 345678901
  - 456789012
```

把示例 `self_id` 换成 NapCat 登录的实际 QQ 号，每个启用的 Bot 使用不同的 QQ 号。实际群号填入该 Bot 的 `allowed_groups`，每群一项。空列表表示允许所有群，首次联调请填写测试群。更改启用名单或 YAML 后重启 Kairos。

`provider` 选择默认地址和认证方式，`api_format` 选择请求和响应格式。OpenCode Go 可按同一个 API Key 选择不同格式，`model` 填 Go API 的模型 ID。各预设的 YAML 写法见 [Bot 配置示例](config/examples/bots/1.yaml)。

每个模型可配置 `max_tokens`、`max_concurrency`、`max_retries`、`timeout_connect` 和 `timeout_read`。兼容代理的 `base_url` 写到 API 版本，例如 `https://your-provider.example/v1`。已有 `chat_model_key` 与全局 `llm_models` 的配置可继续加载。迁移时，把对应模型移到该 Bot 的 `llm.chat`，移除该 Bot 的 `chat_model_key`，并在所有 Bot 迁移后移除全局模型表。

```sh
make dev
```

`make dev` 从 `.env` 和 `config/global.yaml` 读取配置，连接已运行的原生 PostgreSQL，执行 `alembic upgrade head` 并启动 Kairos。OneBot 端口就绪后，`make dev` 复用已运行的 NapCat，或用 Xvfb 启动 `~/Napcat/opt/QQ/qq`。安装路径不同可用 `NAPCAT_QQ=/path/to/qq make dev`。按 Ctrl+C 会停止这次由 `make dev` 启动的进程。

使用 `make launcher` 打开 WSLg 桌面启动管理器。窗口可分别启动或停止两个服务，并显示运行日志。关闭窗口会停止由它启动的进程。桌面入口使用系统 Python、`python3-gi` 和 `gir1.2-gtk-3.0`。终端里可用 `make kairos` 或 `make napcat` 单独启动服务。

### 独立后台运行 NapCat

启用 systemd 的 WSL 可以用 `make napcat-start` 启动独立的用户服务。NapCat 读取自身安装目录的配置，在后台等待 QQ 登录。已有前台 NapCat 时，先正常退出该实例，再执行后台启动命令。安装路径不同可用 `NAPCAT_QQ=/path/to/qq make napcat-start`。

```sh
make napcat-start
journalctl --user -u kairos-napcat.service -f
```

启动后在 NapCat WebUI 登录 QQ，默认地址是 [http://127.0.0.1:6099/webui/](http://127.0.0.1:6099/webui/)。登录后可以用 `make kairos` 启动 Kairos。NapCat 若退出，先查看日志，再手动执行 `make napcat-start`。

按 Ctrl+C 退出日志查看。用 `make napcat-stop` 停止后台 NapCat。若需要退出所有登录会话，可用 `loginctl enable-linger "$USER"` 让 systemd 用户管理器继续运行。WSL 实例关闭时后台服务会一起停止。 [systemd 用户会话文档](https://www.freedesktop.org/software/systemd/man/latest/loginctl.html) · [WSL systemd 文档](https://devblogs.microsoft.com/commandline/systemd-support-is-now-available-in-wsl/)

### NapCat 反向 WebSocket

Kairos 的反向监听端口写在 `config/global.yaml` 的 `onebot_ws.port`，默认是 `8080`。改端口后重启 Kairos，并更新 NapCat WebSocket 客户端 URL 中的端口。

1. 在安装目录的 `resources/app/app_launcher/napcat/config/webui.json` 核对 WebUI 端口和登录令牌。本机联调可先将该文件的 `host` 设为 `127.0.0.1`。
2. 先运行 `make napcat`，再打开 NapCat WebUI，默认地址是 [http://127.0.0.1:6099/webui/](http://127.0.0.1:6099/webui/)。也可在 `~/Napcat/opt/QQ` 下运行 `xvfb-run -a ./qq --no-sandbox`。
3. 由账号操作者在 NapCat 登录测试 QQ，再新增一条 **WebSocket 客户端**。下列字段对应 NapCat 的反向连接配置。 [NapCat 接入文档](https://napcat.napneko.icu/use/integration.html) · [客户端配置字段](https://napcat.napneko.icu/config/basic.html)

```json
{
  "name": "Kairos test",
  "enable": false,
  "url": "ws://127.0.0.1:8080/onebot/v11/ws",
  "token": "replace-with-onebot-token",
  "messagePostFormat": "array",
  "reportSelfMessage": false,
  "debug": false
}
```

这条 WebSocket 连接的令牌取 `KAIROS_ONEBOT_ACCESS_TOKEN` 的值。在对应 Bot YAML 中填好 `self_id` 和 `allowed_groups`，将数字文件名加入 `.env` 的 `KAIROS_ENABLED_BOTS`，再运行 `make kairos`。启用连接后，在白名单群内 `@` Bot 验证收发。多个测试账号可连入同一 Kairos 地址。联调结束后停用新增连接，再按 Ctrl+C 清理本次启动的进程。

每次 NapCat 成功连接 Kairos 后，Bot 按北京时间向 `allowed_groups` 中的第一个群发送「早上好」「中午好」或「晚上好」。群友在白名单群内戳到这个 Bot 时，Bot 会在原群回复 `QAQ`。两种状态消息直接通过 OneBot 发送。

### 聊天触发、上下文与调试

白名单群里的直接 `@`、引用 Bot 聊天回复、或普通文本包含该 Bot 的 `nickname` 都能触发模型。名字匹配忽略大小写。短时间的多个触发会合并处理，群聊插话仍会进入上下文。可用 `trigger_mode: all_with_veto` 让普通群聊也进入门控和回复判断。

每个 Bot 在每个群独立维护默认 40 个上下文消息块。首次收到消息时从 PostgreSQL 读取最近的原始消息，随后合并同一群友在 10 秒内的相邻连发，并淘汰窗口里最早的块。原始消息逐条保存，窗口大小由 Bot YAML 的 `ring_buffer_size` 控制。启用压缩后，历史摘要先进入系统提示，摘要覆盖范围之后的原始消息按时间续接，聊天输入按估算 token 数裁剪。

模型直接生成聊天正文，整条输出为 `<|act:silent|>` 时表示沉默。通过过期和重复检查后，Kairos 把正文作为一条消息发送并写入聊天历史。Gemini 在普通群和调试群统一使用配置里的 `thinking_level: minimal`。Gemini 和 OpenAI 兼容接口默认使用服务端的输出限制。Anthropic Messages 接口要求填写 `max_tokens`，适配器在未配置时填入 4096。

`debug_groups` 可逐行添加需要联调的白名单群号。主模型在调试群选择沉默时，Kairos 引用触发消息发送「这条触发成功但是模型选择沉默」。处理群消息出错时，各白名单群都能收到错误类型和可用的 HTTP 状态码。系统提示与聊天历史分开处理。

### M2.1 可选配置

`gate`、`compaction`、`memory` 分别按 Bot 开启，默认均关闭。把下列字段加到该 Bot 的 YAML。示例复用现有 `llm.chat`，也可以在 `llm` 中新增不同的门控和后台模型槽，再把相应键写入 `gate_model_key`、`worker_model_key`。启用 `gate` 时必须配置 `gate_model_key`。后台生成摘要和提炼事件笔记使用 `worker_model_key`，已有记忆的召回可以独立启用。 [Bot 配置示例](config/examples/bots/1.yaml)

```yaml
trigger_mode: all_with_veto
gate_model_key: chat
worker_model_key: chat
prompt_style: companion
instruction_reminder: ""
enabled_tools: []
gate:
  enabled: true
  persona_card: 爱聊做饭，接得上话题才发言。
  recent_messages_limit: 12
  timeout_seconds: 1.5
  fail_open: false
  energy_max: 100.0
  energy_recover_per_min: 5.0
  energy_cost_reply: 25.0
  energy_cost_react: 5.0
compaction:
  enabled: true
  input_token_budget: 65536
  trigger_tokens: 65536
  target_summary_tokens: 16384
  max_retries: 3
  keep_recent_messages: 16
  max_persisted_summaries: 3
memory:
  enabled: true
  warm_trigger_messages: 6
  max_notes_recalled: 5
  max_impressions_recalled: 10
  note_extract_interval_messages: 12
```

- `gate` 按群维护精力值，并把短人设卡（最多 200 字）、近期消息和线程板发给轻量模型。模型可以放行、沉默或贴表情。直接 `@` 或引用 Bot 回复可绕过精力值；有可用工具时，初筛仍可为直接提问选择工具。普通群聊的门控超时或输出无效时由 `fail_open` 决定是否放行。
- 主模型可以回复、输出 `<|act:silent|>`，或用 `<|act:react:m123:76|>` 调用 NapCat 的 `set_msg_emoji_like`。贴表情的目标消息和表情 ID 记录在决策审计中。
- `compaction` 默认在估算输入达到 65,536 Token 或旧消息即将被窗口淘汰时总结较早的对话，摘要须在 16,384 Token 以内。超长、空输出或请求错误会重试，默认最多尝试三次。全部失败时清空旧上下文与热摘要，并记录重置水位线供重启恢复。`input_token_budget` 最少为 512。仅配置旧预算字段的 Bot 继续使用原有压缩触发与输出参数。
- `memory` 按消息量异步预热群友画像和相关事件笔记。主聊天流程直接读取进程内的有界热缓存。画像、笔记和摘要均按 Bot 与群隔离，事件笔记用关键词和 `pg_trgm` 召回。
- `prompt_style: strict` 强调格式和指令遵循。`instruction_reminder` 最多 500 字，随本轮最后一个 user turn 发送。`enabled_tools` 可填写应用注册的工具名，完整 Schema 仅随门控选中的请求发送。使用同一 QQ 账号测试两种人设时，可复制严格配置示例为 `config/bots/2.yaml`，每次只启用一个 ID。 [严格配置示例](config/examples/bots/2_strict_tester.yaml)

### 本地 Admin 管理台

Admin 默认关闭。启用时，在 `config/global.yaml` 中填写以下配置，并在本机 `.env` 中设置 `KAIROS_ADMIN_ACCESS_TOKEN`。

```yaml
admin_ui:
  enabled: true
  host: 127.0.0.1
  port: 8081
  access_token: "${KAIROS_ADMIN_ACCESS_TOKEN}"
  log_buffer_size: 500
  trace_buffer_size: 100
```

运行 `make kairos` 后打开 [本机 Admin 管理台](http://127.0.0.1:8081/)。页面包含实时日志、Bot 配置、Context 预览、消息流水线和运行控制。配置了令牌时，浏览器用 Bearer 头向 API 验证。日志和追踪缓冲按容量淘汰，可在流水线详情中查看实际请求的分区 Context。请保留回环地址和令牌，配置视图只返回非敏感字段及密钥占位符或掩码。

在页面保存 Bot YAML 后点「热重载生效」，Kairos 会验证并重新装配 Bot，现有 NapCat 连接继续使用。Context 预览可直接使用未保存的草稿。全进程重启会排空并释放服务资源，再用当前进程参数和环境重新启动。更改 `.env` 后通过原启动入口重新启动 Kairos，以加载新环境。

## 测试和内存压测

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q -rs
uv run pytest tests/integration/test_e2e_pipeline.py
uv run python scripts/bench_memory.py
```

端到端回放用本地 WebSocket 模拟 NapCat，用本地 HTTP 模拟模型接口。M1 场景覆盖独立模型配置、角色分离、过期回复和双 Bot 防互刷。M2 场景覆盖初筛沉默、Gate 与主模型贴表情及回执、主模型 Veto、摘要重启恢复、慢查询期间使用热记忆、精力耗尽后直接 `@` 回复。

内存压测开启门控、线程追踪、热记忆、滚动摘要和 Admin 日志与追踪缓冲，在 3 个 Bot、每个 10 个活跃群上回放 5,000 条消息。脚本检查缓冲区、线程表、摘要与热缓存容量，并要求 Kairos Core 的稳态 RSS 小于 180 MB。

M2.1 验收覆盖 19 项本地协议回放，以及 Admin 鉴权、日志与追踪 SSE、安全配置写入、草稿 Context 预览和运行控制。绑定两颗逻辑 CPU、cgroup 内存上限设为 4 GiB 时，压测稳态 RSS 为 **74.02 MB**，保留 500 条日志和 100 条带上下文的流水线追踪。 [Admin 服务测试](tests/admin/test_admin_server.py) · [内存压测](scripts/bench_memory.py)

可选 PostgreSQL 用例使用独立测试库。先设置 `KAIROS_TEST_PG_DSN`，再迁移并测试该数据库。

```sh
KAIROS_PG_DSN="$KAIROS_TEST_PG_DSN" uv run alembic upgrade head
KAIROS_TEST_PG_DSN="$KAIROS_TEST_PG_DSN" uv run pytest -m pg
```

## 2C4G 部署检查清单

- 在 `/opt/kairos` 检出代码并准备系统用户 `kairos`。由该用户在部署目录运行 `uv sync --no-dev`，生成 unit 使用的 `.venv`。
- 将全局配置、Bot 配置和权限为 `0600` 的 `.env` 放在 `/opt/kairos`。确保系统用户可读取这些文件。启用服务前，以该用户在 `/opt/kairos` 加载 `.env` 并运行 `.venv/bin/alembic upgrade head`。
- 给 PostgreSQL 加载内存配置片段，在部署机运行上述回放和 RSS 压测。将 OneBot 监听保留在同机回环接口，并核对每个 Bot 的群白名单。
- [kairos.service](deploy/kairos.service) 以 `kairos` 用户运行，将 Kairos Core 的 cgroup 内存上限设为 `256M`。安装 unit 后查看服务日志。

```sh
sudo install -m 0644 deploy/kairos.service /etc/systemd/system/kairos.service
sudo systemctl daemon-reload
sudo systemctl enable --now kairos.service
sudo journalctl -u kairos.service -f
```
