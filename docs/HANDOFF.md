# Kairos 当前进度与交接

更新于 2026-11-27（北京时间）。Stage 1-B（M2）已完成配置、协议、存储、运行时和本地回放验收。M2 功能按 Bot 启用，下一阶段是 Stage 1-C（M3）。 [M2 配置](../README.md#m2-可选配置) · [Stage 1 实现规划](Kairos_设计文档/02_Kairos_Stage1_技术栈与2C4G工程实现规划.md)

## 已实现的行为

| 范围 | 当前行为 |
| --- | --- |
| 运行方式 | Kairos 在一个 Python 进程中托管 1–3 个 Bot，通过 OneBot v11 反向 WebSocket 按 QQ `self_id` 分流。NapCat 独立运行，并可同时接入其他应用。 |
| Bot 配置 | 每个 Bot 使用 `config/bots/<数字 ID>.yaml`，其中填写 `self_id`、`nickname`、人设、`allowed_groups`、`debug_groups` 和 `llm.chat`。`.env` 的 `KAIROS_ENABLED_BOTS` 选择启用的数字 ID。 |
| 模型 | 各 Bot 独立选择 `provider`、`api_format`、模型和密钥引用，共享 HTTP 连接池。预设涵盖 OpenAI Responses、Chat Completions、Anthropic Messages、OpenCode Go 和 Google AI Studio。 |
| 触发 | 白名单群里直接 `@`、引用 Bot 聊天回复、文字中出现 Bot 名字均可触发。`trigger_mode: all_with_veto` 允许普通群聊参与门控。精力值与冷却按群维护，直接 `@` 或引用可通过精力限制和模型初筛。 |
| 动作 | 轻量模型可放行、沉默或贴表情。主模型保留 Veto，也能直接输出 `<\|act:react:m123:76\|>`。NapCat 通过 `set_msg_emoji_like` 回执，目标和表情 ID 进入决策审计。正文经过防复读和过期校验后发送并留档。 |
| 调试 | 调试群里主模型选择沉默时，引用触发消息发送「这条触发成功但是模型选择沉默」。处理失败时，白名单群会收到错误类型和可用的 HTTP 状态码。 |
| 记忆 | PostgreSQL 按 Bot 和群保存滚动摘要、群友画像和事件笔记。`pg_trgm` 与关键词用于召回，后台工作协程提炼并预热，主聊天从有界热缓存取值。 |
| 在线状态 | NapCat 反向连接建立后，Bot 按北京时间向白名单首群发问候。群友戳一戳 Bot 时，在原群回复 `QAQ`。 |

上下文默认保留 40 个消息块，同一群友间隔不超过 10 秒的相邻连发可合并，数据库逐条保留原始消息。启用压缩时，首次会话加载持久化摘要并按水位线续接原文。系统提示包含摘要、活跃线程板和可用的热记忆。预算裁剪后，历史中的 Bot 发言仍使用原生 `assistant` 角色，第一条对话轮次为 `user`。 [群会话](../src/kairos/runtime/group_session.py) · [上下文构建](../src/kairos/harness/context_builder.py)

## WSL 联调环境

- 上次真实群联调使用 WSL 原生 PostgreSQL 16，Kairos 通过 `/var/run/postgresql` 连接，当时数据库位于 `0001_m1_core_tables`。部署 M2 时用 `make kairos` 迁移到 `0002_m2_memory_and_summaries`。本次数据库验收使用独立 PostgreSQL 16 测试库。 [M2 迁移](../alembic/versions/0002_m2_memory_and_summaries.py)
- 上次联调启用了 Bot `1` 和 `gemini-3.5-flash-lite`。实际群号和 QQ 号由本机 Bot YAML 提供，数据库、OneBot 和模型密钥从本机 `.env` 读取。 [环境变量模板](../.env.sample) · [Bot 配置示例](../config/examples/bots/1.yaml)
- 上次联调通过 Xvfb 启动 NapCat。WebUI 端口为 `6099`，正向 WebSocket 使用 `3001`，Kairos 反向连接使用 `ws://127.0.0.1:8080/onebot/v11/ws`。
- 本机 NapCat 的运行入口在 `~/Napcat/opt/QQ/qq`。源码检出在 `~/Lab/NapCatQQ` 的 `upstream-only` 分支，交接时为 `72912f29`。网站文档镜像在 `~/Lab/NapCatDocs/site`。

## 启动与停止

在 WSL 中进入仓库根目录。首次准备依赖、数据库和配置副本时参照 [本地运行说明](../README.md#在-wsl2-本地运行)。已有数据库可用下面两条命令核对服务和迁移状态。

```sh
pg_isready -h /var/run/postgresql
uv run --env-file .env alembic current
```

| 命令 | 用途 |
| --- | --- |
| `make kairos` | 使用 `.env` 启动 Kairos，先对原生 PostgreSQL 执行 `alembic upgrade head`。已有 NapCat 时用这一条。 |
| `make dev` | 启动 Kairos，复用同路径已运行的 NapCat，或在 OneBot 监听就绪后启动 NapCat。Ctrl+C 会停止该命令启动的进程。 |
| `make napcat` | 在终端前台单独运行 NapCat。 |
| `make napcat-start` | 用 WSL 的 systemd 用户服务启动一次 NapCat，在后台等待登录。NapCat 退出后由操作者手动重开。 |
| `journalctl --user -u kairos-napcat.service -f` | 查看后台 NapCat 日志。 |
| `make napcat-stop` | 停止 `make napcat-start` 管理的后台实例。 |

NapCat 通过 [本机 WebUI](http://127.0.0.1:6099/webui/) 登录。接入 Kairos 时在 NapCat 新建 **WebSocket 客户端**，填写上述反向地址和 `.env` 中的 `KAIROS_ONEBOT_ACCESS_TOKEN`。改端口时同步修改 `config/global.yaml` 的 `onebot_ws.port` 和 NapCat 客户端 URL。需要重新登录 QQ 或向真实测试群发消息时，先与账号操作者对齐时间。测试结束后停止本次启动的临时实例。

## 验证记录

- 独立 PostgreSQL 16 测试库运行 `uv run pytest -q -rs`，通过 280 项，无跳过。已验证 M2 迁移升级与回滚、群级和 Bot 级数据隔离、关键词与 trigram 召回。`ruff check .`、`ruff format --check .` 和 `git diff --check` 均通过。 [测试说明](../README.md#测试和内存压测)
- 本地协议回放使用真实 WebSocket 和本地模型 HTTP 接口，覆盖 M1 交互及 M2 初筛沉默、两条贴表情路径、主脑 Veto、摘要重启恢复、热记忆和精力值限流。 [回放测试](../tests/integration/test_e2e_pipeline.py)
- M2 内存压测回放 3 个 Bot、30 个会话和 5,000 条消息，普通本地运行的稳态 RSS 为 68.08 MB，绑定两颗逻辑 CPU 且 cgroup 内存限制为 4 GiB 时为 68.17 MB。两次均低于 180 MB 红线，每群最多 40 个消息块、20 个线程和 3 条热摘要。 [压测脚本](../scripts/bench_memory.py)
- 上次真实群联调已确认反向连接、上线问候和 `QAQ`。M2 真实群验收待在约定白名单内联调。

## 接续工作

1. 在约定白名单群逐项启用 M2。通过决策审计观察门控放行后的主脑 Veto 比例，以及从防抖结束到第一条消息的延迟。
2. 接入 Stage 1-C 的生物节律 FSM，让 `CHATTING`、`BROWSING`、`SLEEPING` 状态参与群级规则门控，并验收休眠期的主动发言行为。
3. 给刷推扩展接入异步活动插件与灵感队列，控制内容筛选、过期与在群聊中的分享频率。
4. 接入短生命周期的工具沙箱，在独立子上下文中运行工具，再把事实交给主模型转述。 [M3 规划](Kairos_设计文档/02_Kairos_Stage1_技术栈与2C4G工程实现规划.md)
