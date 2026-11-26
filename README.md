# Kairos（知时）

Kairos 在一个 Python 进程内托管 1–3 个 QQ Bot，通过 OneBot v11 反向 WebSocket 接收群消息。它使用 PostgreSQL 保存群消息和决策审计，并通过 OpenAI 兼容接口生成回复。消息聚合、角色分离、防抖和过期检查的设计见 [Stage 1 实现规划](Kairos_设计文档/02_Kairos_Stage1_技术栈与2C4G工程实现规划.md)。

## 在 WSL2 本地运行

在 WSL2 的 Linux 文件系统中检出仓库，例如 `~/Lab/Kairos`。**请勿放在 `/mnt/c` 等 Windows 挂载目录下。** 准备 Python 3.12 或更高版本、`uv` 和原生 PostgreSQL 16 或更高版本。首次启动时，在仓库根目录准备环境和配置副本。

```sh
uv sync --dev
cp config/global.example.yaml config/global.yaml
cp config/bots/example_companion.yaml config/bots/lin_xiaoman.yaml
```

### PostgreSQL

在专用开发集群中创建应用用户和数据库。第一条命令会提示设置数据库密码。

```sh
sudo -u postgres createuser --login --pwprompt kairos
sudo -u postgres createdb --owner=kairos kairos
```

在 2C4G 机器上将 [postgres_2c4g.conf](deploy/postgres_2c4g.conf) 纳入当前集群的 `postgresql.conf`。Ubuntu PostgreSQL 16 的默认集群可将该文件放入 `/etc/postgresql/16/main/conf.d/`。先用 `sudo -u postgres psql -Atc 'SHOW config_file'` 确认实际路径及 `include_dir`，安装后重启该集群以加载内存和连接数参数。

### 配置和启动

在仓库根目录的 `.env` 中设置以下变量。将示例值替换为本地值，数据库密码中的 URL 保留字符需要编码。使用 `chmod 600 .env` 限制凭据文件的读取权限。

```dotenv
KAIROS_PG_DSN='postgresql://kairos:replace-with-url-encoded-password@127.0.0.1:5432/kairos'
KAIROS_ONEBOT_ACCESS_TOKEN='replace-with-onebot-token'
OPENAI_API_KEY='replace-with-provider-key'
KAIROS_BOT_SELF_ID=90000001
KAIROS_SECOND_BOT_SELF_ID=90000002
```

`config/global.yaml` 的 `llm_models` 支持多个命名槽位。每个 Bot 的 `chat_model_key` 选择其中一个槽位。下列两个 YAML 文件各配置一个 Bot。

```yaml
# config/bots/lin_xiaoman.yaml
bot_id: lin_xiaoman
self_id: "${KAIROS_BOT_SELF_ID}"
nickname: 林小满
persona_prompt: 我是林小满，喜欢聊做饭和游戏。
chat_model_key: chat_model
trigger_mode: mention_only
allowed_groups: [900001]
```

```yaml
# config/bots/chen_adai.yaml
bot_id: chen_adai
self_id: "${KAIROS_SECOND_BOT_SELF_ID}"
nickname: 陈阿呆
persona_prompt: 我是陈阿呆，喜欢聊电脑和电影。
chat_model_key: chat_model
trigger_mode: mention_only
allowed_groups: [900001]
```

将示例 QQ 号和群号换成本次测试账号及隔离测试群。每个 Bot 必须有不同的 `bot_id` 和 `self_id`。`example_*.yaml` 是配置模板。首次联调使用 `mention_only` 和非空群白名单。

```sh
set -a
. ./.env
set +a
uv run alembic upgrade head
uv run python -m kairos.main --config config/global.yaml --bots-dir config/bots
```

迁移创建 `group_messages` 和 `decision_audit_logs`。进程出现 `kairos_ready` 日志后，反向 WebSocket 开始接收连接。

### NapCat 反向 WebSocket

在同一 WSL2 网络环境中，使用 NapCat WebUI 新增一条 WebSocket 客户端。下列字段对应 NapCat 的反向连接配置。[NapCat 接入文档](https://napneko.github.io/use/integration) · [客户端配置字段](https://github.com/NapNeko/NapCatDocs/blob/main/src/config/basic.md)

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

令牌使用与 `KAIROS_ONEBOT_ACCESS_TOKEN` 相同的值。真实 QQ 联调时，由账号操作者在 NapCat 完成登录并确认测试群，然后启用这条连接，在白名单测试群内 `@` Bot 验证收发。多个测试账号可连入同一 Kairos 地址。联调结束后停用新增连接，再用 Ctrl+C 停止本地 Kairos 进程。

## 测试和内存压测

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run pytest tests/integration/test_e2e_pipeline.py
uv run python scripts/bench_memory.py
```

端到端回放用本地 WebSocket 模拟 NapCat，用本地 HTTP 模拟 OpenAI 兼容接口。它覆盖同人连发、角色分离、`@` 快路径、过期引用与丢弃、Veto 和双 Bot 防互刷。压测在 3 个 Bot、每个 10 个活跃群上回放 5,000 条消息，检查每群缓冲区有界和 Kairos Core RSS 小于 180 MB。

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
