# GPT 审批型 Binance Futures 实施与验收路线

本页只描述 `crypto_event_trader`。仓库中的 A 股研究工作台继续独立维护，不与合约交易的
依赖、数据库、容器入口或资金权限混用。勾选表示软件能力已实现，不表示策略有正期望，
也不表示真实账户、数据授权或运行时长已经验收。

## 已实现的软件边界

- [x] USDⓈ-M Futures `/fapi/*`、`fstream` `/public`/`market`/`private` 强制分流、
  签名、校时、动态 filters、
  429/418 退避和 5xx/超时未知订单恢复。
- [x] 单向逐仓、最高 3× 杠杆、唯一 client order ID、两次受保护限价尝试、
  `/fapi/v1/algoOrder` 保护单、`reduceOnly` 退出与启动/重连对账。
- [x] 30 日点时流动性观察、180 日上市年龄、10 bps 价差、20× 深度及 top-10/top-12
  滞后交易池；覆盖不足时保持空池。
- [x] 24/72/168 小时动量与 7/21 日 Donchian 五票 champion、EWMA 波动定仓，
  funding/基差/OI/ADL/盘口仅作 `1/0.5/0` 缩放。
- [x] OpenAI Responses 严格结构化审批、精确模型探测、超时/模型不匹配失败关闭，
  GPT 不持有 Binance 密钥且不能越过候选方向、数量或硬风控。
- [x] `OPEN/ADD/HOLD/REDUCE/CLOSE/REJECT` 生命周期、追加式持仓论点、一次盈利加仓、
  反向信号只触发原仓管理，不会直接翻仓。
- [x] 代码级不可放宽的日亏损、总回撤、gross/net/单币/相关簇、杠杆和单仓风险上限；
  加仓不能放宽旧止损，合并仓位到止损的最坏损失不超过净值 1%。
- [x] PostgreSQL 追加式审计、Redis 共享控制与 leader、Streams pending reclaim、
  订单/成交/funding 权威账本、反事实 1h/4h/24h 结算及崩溃恢复。
- [x] 单 leader 的自主 paper worker、公开行情、内部撮合、逐 episode funding coverage、
  独立 1 秒 ATR 模拟保护止损、
  重启账本重建；paper HTTP API 只有控制/审计能力，不能触发另一条交易管线。
- [x] X 白名单 filtered stream、默认不向 OpenAI 发送 X 原文、GitHub allowlist + ETag +
  签名 webhook、编辑/删除版本和高影响事件复审；不 checkout 或运行外部代码。
- [x] 受限 `StrategySpec`、每日错误分析、每周 challenger 提案、晋升/回滚状态机，以及
  严格校验外部 backtest/paired-shadow JSON 证据的一次性无外网作业。

## 软件仍需补充的能力

- [ ] 内置、可独立重放的 point-in-time 历史模拟器。当前仓库严格验证由受审计外部模拟器
  提交的 manifest，但不会自行证明外部程序真的执行了退市、深度、部分成交和分段场景。
- [ ] 同时运行 champion/challenger 并自动生成逐日 paired-shadow 成交证据的内部 runner。
  当前 shadow journal 能恢复和验收，数据由外部 Demo shadow runner 导入。
- [ ] 足以重建所有历史回测的授权数据湖，包括上市/退市、逐档盘口、filters 版本、
  手续费层级及完整市场事件。决策时输入的闭合 K 线可审计，不等于完整历史数据湖。
- [ ] Binance 公告/RSS 采集器，以及来源可靠度、MAE/MFE 的生产统计。
- [ ] 研究验证器专用的最小权限 PostgreSQL 角色和数据库级权限自动验收。
- [ ] 跨主机高可用、备份恢复演练、指标告警和 API worker 均不可用时的独立远端撤单守护。

## 必须由部署或时间提供的证据

- [ ] 在目标司法辖区确认 Binance、X、GitHub、市场数据和 OpenAI 的合法可用性及许可。
- [ ] 验证指定 OpenAI 模型在独立 paper/demo/live projects 中真实可用；不得静默换模型。
- [ ] 使用独立数据库、独立 OpenAI project、不同 Binance Demo/live 密钥，完成 IP 白名单、
  只开 Futures 权限、禁提现和 secret rotation 演练。
- [ ] 用真实 Binance Demo 凭证验证部分成交、手续费、funding、listenKey 过期、503 unknown、
  algo 拒绝、断网重连和远端/本地不一致恢复。
- [ ] 由独立操作者核验外部回测 manifest 的完整文件 SHA-256、数据授权与模拟器版本，
  并保存 walk-forward、封存 12 个月、±25% 扰动、延迟/placebo、2×/3× 成本及 DSR/PBO 证据。
- [ ] paired Demo shadow 连续覆盖至少 90 天，champion/challenger 双方各至少 30 笔闭仓；
  challenger 净收益至少高 10% 且风险不恶化。
- [ ] 完成 `paper → Demo → 10% canary → 25% → 100%` 每一级人工解锁、回滚和灾难恢复演练。

## 当前可接受的运行范围

当前适合本地测试、持续 paper 和受监督 Binance Demo 联调。未完成以上外部证据、90 天
paired shadow、真实 Demo 故障演练和部署隔离前，不得解锁 live。即使全部通过，也只说明
流程与风险边界达到预先定义的验收标准，不保证盈利。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,trader]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
```

```bash
docker compose --profile paper up -d --build trader-paper-api trader-paper-worker
docker compose --profile research run --rm trader-research-validator
```
