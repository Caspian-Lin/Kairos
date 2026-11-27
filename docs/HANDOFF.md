# Kairos 当前进度与交接

更新于 2026-11-27（北京时间）。本地 `main` 位于 `0683837`。该版本已经包含 Stage 1 的配置、协议接入、PostgreSQL 存储、消息处理、多 Bot 运行时和回放测试。下面的 WSL 启动入口、每 Bot 模型预设和近期交互调整位于本地工作树。 [Stage 1 实现规划](Kairos_设计文档/02_Kairos_Stage1_技术栈与2C4G工程实现规划.md)

## 已实现的行为

| 范围 | 当前行为 |
| --- | --- |
| 运行方式 | Kairos 在一个 Python 进程中托管 1–3 个 Bot，通过 OneBot v11 反向 WebSocket 按 QQ `self_id` 分流。NapCat 独立运行，并可同时接入其他应用。 |
| Bot 配置 | 每个 Bot 使用 `config/bots/<数字 ID>.yaml`，其中填写 `self_id`、`nickname`、人设、`allowed_groups`、`debug_groups` 和 `llm.chat`。`.env` 的 `KAIROS_ENABLED_BOTS` 选择启用的数字 ID。 |
| 模型 | 各 Bot 独立选择 `provider`、`api_format`、模型和密钥引用，共享 HTTP 连接池。预设涵盖 OpenAI Responses、Chat Completions、Anthropic Messages、OpenCode Go 和 Google AI Studio。 |
| 触发 | 白名单群里的直接 `@`、引用 Bot 的聊天回复、文字中出现 Bot 名字均可触发。名字匹配忽略大小写。短时间的触发合并处理，已接受的点名会保留到这一轮。 |
| 回复 | 模型输出聊天正文，或单独输出 `<\|act:silent\|>`。通过防重复和过期检查的正文作为一条消息发送并留档。Gemini 普通群和调试群均使用 `minimal`，由接口决定输出上限。 |
| 调试 | 调试群里模型选择沉默时，引用触发消息发送「这条触发成功但是模型选择沉默」。处理失败时，白名单群会收到错误类型和可用的 HTTP 状态码。 |
| 在线状态 | NapCat 反向连接建立后，Bot 按北京时间向白名单首群发问候。群友戳一戳 Bot 时，在原群回复 `QAQ`。 |

上下文按 Bot 和群隔离，默认最多保留 40 个历史消息块。首次进入会话时，从 PostgreSQL 取最近 40 条原始记录。同一群友相邻间隔不超过 10 秒的连发可以拼成一个块。窗口满后淘汰最早的块，数据库保留逐条记录。当前的上下文容量由 `ring_buffer_size` 控制，摘要压缩和输入 token 预算属于后续工作。 [群会话](../src/kairos/runtime/group_session.py) · [上下文构建](../src/kairos/harness/context_builder.py)

## 本机状态

- WSL 原生 PostgreSQL 16 `main` 已运行。Kairos 使用 `/var/run/postgresql` 的 Unix socket，数据库版本已核对为 `0001_m1_core_tables`，包括 `group_messages` 和 `decision_audit_logs`。 [数据库迁移](../alembic/versions/0001_m1_core_tables.py)
- 本地只启用了 Bot `1`，模型为 `gemini-3.5-flash-lite`，配置了一个白名单群及一个调试群。实际群号和 QQ 号从本机 Bot YAML 读取，数据库、OneBot 和模型密钥从本机 `.env` 读取。 [环境变量模板](../.env.sample) · [Bot 配置示例](../config/examples/bots/1.yaml)
- NapCat 在终端通过 Xvfb 运行。WebUI 端口是 `6099`，已有正向 WebSocket 使用 `3001`。Kairos 的反向地址是 `ws://127.0.0.1:8080/onebot/v11/ws`，交接时已看到本机 NapCat 到 `8080` 的连接。
- 交接时的 Kairos 由终端中的 `make kairos` 启动，启动时间早于最近的输出和调试简化改动。新配置与代码会在下次启动时加载。
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

- 本地 `pytest -q -rs` 通过 172 项。可选 PostgreSQL 测试因 `KAIROS_TEST_PG_DSN` 为空跳过 1 项。测试库的迁移和执行命令见 [测试说明](../README.md#测试和内存压测)。
- `ruff check .` 和 `git diff --check` 均通过。
- 本地端到端回放使用模拟 NapCat 和模型 HTTP 服务，覆盖双 Bot 独立鉴权和协议、群聊触发、长正文、沉默标记、调试提示、过期处理和错误上报。 [回放测试](../tests/integration/test_e2e_pipeline.py)
- 最新内存压测回放 3 个 Bot、30 个会话、5,000 条消息，稳态 RSS 约 63 MB，每会话缓冲区最多 40 个块。 [压测脚本](../scripts/bench_memory.py)
- 真实群联调已确认反向连接、上线问候和 `QAQ`。最新输出简化后的真实群验收留到下一次 Kairos 重启后进行。

## 接续工作

1. 在原 Kairos 终端结束当前进程，再运行 `make kairos`。在约定白名单群核对上线问候、直接 `@`、按 Bot 名字触发、长正文、沉默提示和戳一戳。
2. 继续上下文压缩设计。当前原始消息与活跃窗口已有存储基础，下一步确定输入 token 预算、摘要生成时机和摘要恢复顺序。 [记忆设计](Kairos_设计文档/02_Kairos_Stage1_技术栈与2C4G工程实现规划.md)
3. 后续动作规划先接入 `REACT` 贴表情。到实现时再为动作增加独立的类型和目标消息字段。 [动作规划](Kairos_设计文档/02_Kairos_Stage1_技术栈与2C4G工程实现规划.md)
4. 整理工作树里的启动、配置、运行时、测试和文档改动。现有桌面启动管理器是实验入口，交接和验收以命令行为准。
