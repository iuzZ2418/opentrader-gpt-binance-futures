# Linux 部署与交易运行手册

## 0. 使用前提

- 仅在 Binance USDⓈ-M Futures、行情/社交数据 API 及用户所在司法辖区合法可用时部署。
- 首次运行一律使用 `paper/internal`；随后依次经过 Demo、10% canary、25% 和 100%。资金阶段只能人工解锁。
- 操作员应熟悉 [架构](crypto-architecture.md)、[安全边界](crypto-security.md) 和 [策略治理](strategy-governance.md)。
- 下列命令假定 Docker Engine 与 Compose v2 已安装，执行目录为仓库根目录。

## 1. 纸面环境

```bash
cp .env.example .env
chmod 600 .env
docker compose config --quiet
docker compose up -d --build
docker compose ps
curl --fail http://127.0.0.1:8000/health
```

确认 `/health` 至少显示：

- `trading_stage` 为 `paper`；
- `execution_venue` 为 `internal`；
- `control.new_positions_enabled` 符合预期；
- 没有 Binance 凭据也能运行纸面测试；
- API 端口只绑定 `127.0.0.1`。

查看日志和停止：

```bash
docker compose logs -f --tail=200 trader-api
docker compose down
```

`down` 不删除命名卷。不要使用 `down -v` 作为常规操作。

`CONTROL_API_TOKEN` 在示例中故意为空，因此除 `/health` 外的交易 API 均返回 401。需要查看审计、
策略或操作纸面管线时，先把由密钥服务生成的高熵 token 写入环境，再用 `X-Control-Token` 调用；
不要把 token 放进 URL、日志或 shell 历史。

## 2. Binance Demo

为 Demo 创建专用密钥并在单独的 `.env.demo` 中设置：

```dotenv
APP_ENV=demo
TRADING_STAGE=demo
EXECUTION_VENUE=binance_futures_demo
LIVE_TRADING_ENABLED=false
ALLOW_BINANCE_PRODUCTION=false
BINANCE_API_KEY=demo-key
BINANCE_API_SECRET=demo-secret
CONTROL_API_TOKEN=<long-random-value>
OPENAI_API_KEY=<demo-project-key>
OPENAI_PROJECT=<demo-project-id>
```

启动前检查展开后的配置，避免 Compose 插值错误：

```bash
docker compose --env-file .env.demo config > /tmp/trader-compose-demo.yaml
grep -E 'TRADING_STAGE|EXECUTION_VENUE|LIVE_TRADING_ENABLED|ALLOW_BINANCE_PRODUCTION' /tmp/trader-compose-demo.yaml
docker compose --env-file .env.demo --profile external up -d --build \
  trader-external-api trader-worker
curl --fail http://127.0.0.1:8002/health
```

不要把 `/tmp/trader-compose-demo.yaml` 上传或提交，因为展开配置可能含 secret。检查以下条件后才运行策略周期：

1. Futures server time 偏移在允许范围；
2. 账户为单向持仓、逐仓保证金，目标币种杠杆不高于 3×；
3. 该账户没有 COIN-M 持仓、挂单或其他机器人；
4. REST 余额/持仓/挂单与私有 WS 状态一致；
5. `exchangeInfo` filters 已加载；
6. mark price、账户流和私有流均未陈旧；
7. WS 日志确认 `/public`、`/market`、`/private` 三条路由均已连接；Demo 主机必须是
   `demo-fstream.binance.com`，不得使用已退役的 `fstream.binancefuture.com`；
8. `ALGO_UPDATE` 与 `CONDITIONAL_ORDER_TRIGGER_REJECT` 回放能够锁定新风险、写入归属审计并完成 REST 对账。
9. OpenAI 审批模型可用且 schema 探针成功；
10. 保护单拒绝和 503 unknown 回放测试通过。

Demo 至少运行 90 天并闭仓 30 笔才满足资金阶段候选条件；策略晋升仍受更严格的研究门槛约束。

### 2.1 外部情报 worker

情报采集与交易执行分进程部署。至少配置精确 OpenAI extraction model；GitHub 和 X 可以分别启用：

```dotenv
OPENAI_INTELLIGENCE_API_KEY=<least-privilege-intelligence-key>
OPENAI_INTELLIGENCE_PROJECT=<intelligence-project-id>
OPENAI_EXTRACTION_MODEL=gpt-5.6-luna
GITHUB_ALLOWED_REPOSITORIES=owner/repository,owner/second-repository
GITHUB_TOKEN=<read-only-token-or-empty-for-public-limits>
GITHUB_WEBHOOK_SECRET=<high-entropy-secret-for-authorized-repositories>
X_BEARER_TOKEN=
X_ALLOWED_ACCOUNT_IDS=
X_CONTENT_TO_OPENAI_ALLOWED=false
INTELLIGENCE_STREAM_NAME=trader:external-evidence
INTELLIGENCE_POLL_SECONDS=300
INTELLIGENCE_EVIDENCE_TTL_SECONDS=3600
GITHUB_POLL_LIMIT=30
X_STREAM_RECONNECT_SECONDS=5
```

启动和观察：

```bash
docker compose --env-file .env.demo --profile intelligence up -d --build \
  trader-intelligence-worker trader-intelligence-api
docker compose logs -f --tail=200 trader-intelligence-worker
```

验收日志必须先出现 `openai=required:<精确模型>`；未配置 X 时应出现
`x_collector_skipped reason=not_configured`。只配置 token 或只配置账号 allowlist 属于错误配置，进程会
fail-closed。GitHub allowlist 只接受 `owner/name`；采集器只调用 releases、security-advisories 和 commits
REST，不 checkout、不下载 artifact、不导入或执行外部代码。

自有或获授权仓库可把 `release`、`push`、`repository_advisory`/`security_advisory` webhook
发到反向代理后的 `POST /webhooks/github`。入口要求 GitHub 的 `X-Hub-Signature-256`、
`X-GitHub-Delivery` 和 `X-GitHub-Event`，且 payload 中的 `repository.full_name` 必须在 allowlist；
默认端口只绑定 `127.0.0.1:8003`，公开接收时必须经过 TLS、请求大小限制与反向代理防护。

Redis Stream 默认为 `trader:external-evidence`，每条 envelope 的 `task_type` 为
`external_evidence.normalized.v1`，payload schema 为 `external-evidence.v1`：

- 身份：`evidence_record_id`、稳定 `evidence_id`、单调 `version`、`ORIGINAL|EDIT|DELETE`；
- 范围：`source/source_id`、`symbols`（`*` 表示市场级）；
- 结构化结果：`event_type`、`sentiment`、`confidence`、`source_ids`、数值 `aggregates`；
- 安全状态：`normalization_status`、`usable_for_trading`、精确模型/提示词版本、SHA-256；
- 时间：`occurred_at`、`observed_at`、`expires_at`、可选 `deleted_at`。

消息不含原文。PostgreSQL `external_evidence` 保存源内容的追加式原始/编辑/删除版本并作为 outbox：
只有审计落库和 Redis 发布成功后 ETag 才前移；若 Redis 发布失败，下一轮从同一审计版本按确定性 task ID
重放，消费者必须幂等。消费端使用 `EvidenceInbox`，只取指定 symbol 在 TTL 内、最新、未删除且
`usable_for_trading=true` 的版本。删除 tombstone 会立即遮蔽旧版本。

默认禁止向 OpenAI 发送 X 原文。本地规则先对原文做事件、情绪与资产映射，模型只收到聚合特征和
source ID；本地 UNKNOWN 固定为低可信 `NONE`，不允许模型根据纯元数据臆测。只有确认数据许可并显式设置
`X_CONTENT_TO_OPENAI_ALLOWED=true` 才改变此边界。

## 3. Live / canary 变更流程

禁止把纸面或 Demo 目录原地改成生产。建立新主机/namespace、PostgreSQL、OpenAI project 和 Binance live 密钥，并执行双人复核。生产 `.env` 至少应：

```dotenv
APP_ENV=production
TRADING_STAGE=live
EXECUTION_VENUE=binance_futures_live
CAPITAL_ALLOCATION_FRACTION=1.0
LIVE_TRADING_ENABLED=true
ALLOW_BINANCE_PRODUCTION=true
CONTROL_API_TOKEN=<secret-manager-injected-random-value>
OPENAI_PROJECT=<dedicated-live-project-id>
TRADER_EXTERNAL_API_BIND=127.0.0.1
```

资金阶段由代码硬绑定，不能只改一个百分比绕过：`canary=0.10`、`scaled=0.25`、
`live=1.0`。每次切换都需要重新部署；进程启动后仍保持 Redis 运行时锁定，随后由
认证操作员再次调用 `/control/unlock-live`。例如首个生产 canary 必须同时设置：

```dotenv
TRADING_STAGE=canary
CAPITAL_ALLOCATION_FRACTION=0.10
EXECUTION_VENUE=binance_futures_live
LIVE_TRADING_ENABLED=true
ALLOW_BINANCE_PRODUCTION=true
```

另外必须更换 PostgreSQL 密码、限制固定出口 IP、把控制 API 放入 VPN/mTLS 认证反代，并从密钥服务注入 Binance/OpenAI secret。代码只验证 `OPENAI_PROJECT` 非空；项目确为生产专用必须由部署审核与密钥策略保证。不要依赖示例文件的本地密码。

部署并核验后，系统仍应处于运行时锁定。操作员带 token 解锁：

```bash
curl --fail -X POST \
  -H "X-Control-Token: ${CONTROL_API_TOKEN}" \
  http://127.0.0.1:8002/control/unlock-live
curl --fail \
  -H "X-Control-Token: ${CONTROL_API_TOKEN}" \
  http://127.0.0.1:8002/control/status
```

解锁只是允许已批准执行器接收新增风险，不会替代资金阶段授权。先使用代码强制的
`canary/0.10`，验证至少一个完整观察期后再人工部署 `scaled/0.25`；`live/1.0` 同理。
每次变更记录操作者、审批单、部署摘要、策略版本、账户净值和 UTC 时间。

## 4. 日常开盘前检查

- 容器、PostgreSQL、Redis 健康，无重启循环和磁盘告警；
- UTC 时钟同步，Binance server time 偏移正常；
- 用户流连接、listen key、market stream sequence 正常；
- REST 对账为零差异，未知订单队列为空；
- 当前 champion、提示词和模型版本与批准记录一致；
- 过去 24 小时 GPT JSON/schema、延迟、配额与置信度校准无异常；
- 日亏损、峰值回撤、gross/net/币种/相关簇敞口未越界；
- 交易池的上市天数、价差和深度条件仍满足；
- 交易所状态、地区服务可用性和计划维护无异常。

## 5. Kill switch 与熔断

手工立即禁止新增风险：

```bash
curl --fail -X POST \
  -H 'Content-Type: application/json' \
  -H "X-Control-Token: ${CONTROL_API_TOKEN}" \
  -d '{"reason":"operator_incident"}' \
  http://127.0.0.1:8002/control/kill
```

- 日净亏损达到 3%：撤销挂单，只允许减仓/平仓，冻结到下一 UTC 日，并需人工确认才能恢复。
- 峰值回撤达到 20%：全平并永久关闭 live，直到独立事故复盘和人工解除。
- 对账差异、市场/账户流陈旧、风控越界、保护单缺失或策略性能明显漂移：立即禁止新增风险；必要时确定性减仓。

重置 kill switch 只能在根因消除、REST/WS 再对账、保护单确认和事件记录完成后执行：

```bash
curl --fail -X POST \
  -H "X-Control-Token: ${CONTROL_API_TOKEN}" \
  http://127.0.0.1:8002/control/reset
```

## 6. `UNKNOWN` 订单与对账

TIMEOUT、断线、5xx/503、重复 client ID 或响应不可解析都不等于“订单失败”。处置顺序固定：

1. 立即停止该 symbol 的新增风险和重试；保留原 client order ID、请求参数、发送/超时时间和响应头。
2. 查询本地用户流缓存是否已有该 client ID 的 `NEW/PARTIALLY_FILLED/FILLED/CANCELED/EXPIRED/REJECTED`。
3. 用同一 client ID / exchange order ID 查询 Futures REST 订单状态和成交记录。
4. 拉取账户、持仓和挂单快照；与本地订单/成交账本逐项核对。
5. 若仍未知，保持冻结并按 Binance 支持流程升级；绝不换 ID 补单。
6. 只有确定订单不存在或已有最终状态后，才由状态机决定是否生成新的、独立审批的意图。
7. 把查询证据、最终结论、敞口修正和操作者追加到同一 trace。

启动、WS 重连和每个周期至少对账：余额、逐仓持仓数量/入场价/未实现 PnL、挂单、algo 保护单、最近成交、funding 和手续费。交易所多出的持仓视为最高优先级事故；本地多出的持仓不能仅通过改数据库“修好”。

## 7. 常见事故

### 行情或账户流陈旧

冻结 `OPEN/ADD`，从 REST 获取快照并按文档重建 WS；验证 sequence 连续和更新时间后再恢复。已有止损不依赖模型或市场流。

### 429 / 418 / 动态限频

遵守响应头和 `Retry-After`，指数退避并降低任务速率。418 表示封禁，不能切换 IP 规避。退出风险请求也要预算独立限频容量。

### OpenAI 不可用

停止 `OPEN/ADD`，不切换未批准模型。记录失败类别和延迟；已有仓位按交易所保护单、确定性失效条件与熔断器管理。
交易 worker 启动时还会探测精确 research model；研究模型不可用不会替换模型，并会触发 kill switch，
从而禁止新增风险但保留对账、保护单和确定性持仓管理。
情报 worker 的精确 extraction model 探针失败时进程不会开始采集；运行中抽取失败的源版本仍写审计，
但通知标记 `normalization_status=REJECTED`、`usable_for_trading=false`。

### 数据库或 Redis 不可用

PostgreSQL 不可写或 trace 不完整时停止所有新订单。Redis 故障后，只有已实现数据库 outbox 的
外部证据通知可从 PostgreSQL 确定性重发；锁、控制状态和其他任务必须按各自恢复流程重新建立，
不能假设 Redis 保存了原始点时数据，也不能把未确认消息当作未下单而重发。

### 保护单缺失/拒绝

立即停止新增风险，查询真实持仓和 algo order；无法快速恢复等价保护时确定性减仓或平仓，并记录交易所拒绝原因与 filters 快照。

## 8. 备份、升级与回滚

- PostgreSQL 每日加密备份，保留可恢复的 WAL/PITR；每季度在隔离环境演练恢复。
- 备份策略注册表、部署镜像摘要和配置非密钥部分；Redis AOF 不替代数据库备份。
- 升级前 kill 新仓、等待/撤销挂单并对账；在 Demo 回放协议测试，再以不可变镜像部署。
- 版本回滚时同时回滚代码、提示词与 champion 指针，不能删除新版本已产生的审计事实。
- 恢复后先保持锁定，执行完整 REST/WS 对账和模型探针，再人工解锁。

## 9. 导入研究与配对影子证据

验证器不是回测器，也不创造交易。受审计的外部点时模拟器/影子 runner 先输出纯 JSON 清单；清单
不得包含需要执行的插件、模块或脚本。`BACKTEST` 必须包含完整 walk-forward 与 sealed holdout 场景矩阵、
逐笔费用/滑点/funding/市场数据摘要，以及与精确统计请求摘要绑定的独立 DSR/PBO 结果。
`PAIRED_SHADOW` 必须同时绑定当前 champion/challenger，提交逐日覆盖和双方逐笔真实核算证据。

由模拟器操作者交付清单后，另一名操作员在可信、只读环境中核对文件来源和完整字节 SHA-256。
不要把清单自己声明的摘要当作独立证明。将已核对值和绝对文件路径放入运行环境：

```bash
chmod 0444 /srv/trader/research/research-manifest.json
export RESEARCH_MANIFEST_PATH=/srv/trader/research/research-manifest.json
export RESEARCH_MANIFEST_SHA256=<独立核对的 64 位十六进制 SHA-256>
docker compose --profile research run --rm trader-research-validator
```

该 profile 只挂载只读清单和内部 PostgreSQL 网络，没有互联网出口，也不继承 Binance、OpenAI、X、
GitHub、Redis 或控制面凭据。验证器先核对原始字节摘要，再解析严格 schema，最后才连接数据库。
清单生成时间超出当前时钟容差、研究/影子截止时间晚于生成时间、策略身份冲突、重复 JSON key、
缺失场景或未经审计库确认的影子覆盖/成本 evidence ID 都会拒绝，且不会写入可晋升结果。

paired-shadow 成本证据不是任意已存在的 evidence ID。fee、slippage、funding 必须使用三个不同的
`paired-shadow-cost-v1` 记录，并逐项匹配同一成交的 `trade_id`、`episode_id`、真实 `trace_id`、
`symbol`、`strategy_version`、`closed_at` 与对应金额；审计行的 trace/发生时间也必须一致。真实
`trace_id` 必须预先出现在 external evidence 之外的审计事实中。导入旧式普通 external evidence、
复用一个 ID 或只修改 payload 来伪造 trace 都会在预检阶段拒绝，journal 保持不变。

退出码及处理方式：

- `0`：`APPENDED`、`ALREADY_APPENDED` 或 `JOURNALED`；下一次晋升调度才会读取成熟结果；
- `2`：清单或证据被拒绝；保留文件和独立摘要，调查 `reason_code`，禁止手工补写数据库；
- `3`：`NOT_MATURE`；日志可以保留，但至少 90 天的每个 UTC 日期和双方各 30 笔闭仓尚未齐备。

长周期影子盘可先用 `finalize:false` 幂等追加覆盖/成交日志；准备验收时提交同一 pair 的
`finalize:true` 清单。重复 event key 必须内容一致，冲突事件会失败，服务重启后从 PostgreSQL 日志恢复。

## 10. A 股服务 profile

原有 A 股 API 与 dashboard 保留为可选 profile，不随交易服务默认启动：

```bash
docker compose --profile company up -d company-api company-dashboard
```

默认端口为回环地址的 `8001` 和 `8501`，与交易 API 隔离。
