# FedForge: 模块化联邦学习系统（Federated Learning）

> 作者：晨星 (CJX0712) · 2026-09-27
> 复用顶级开源：Flower (flwr 1.38) / Optuna / scikit-learn / numpy，零自研 SOTA

FedForge 是一套**可运行、可复现、CPU-only** 的模块化联邦学习系统：合成联邦数据 →
Dirichlet non-IID 划分 → FedAvg / FedMedian / FedTrimmedMean 聚合策略 →
收敛与公平性评测 → Optuna HPO → benchmark 落盘。

## 一键复现

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows; Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest tests -q -W ignore::UserWarning           # 55 passed, 1 skipped
python examples/run_demo.py                      # 生成 benchmark.json
python -m fedforge.cli train --strategy fedavg --partition dirichlet --alpha 0.1
python -m fedforge.cli hpo --trials 10
```

## 基准结果（seed=42，固定可复现，详见 benchmark.json）

| backend | partition | alpha | strategy | final_acc | worst_acc | rounds@90% |
|---------|-----------|-------|----------|-----------|-----------|------------|
| numpy | iid | 0.00 | fedavg | 0.9185 | 0.9000 | 1 |
| numpy | iid | 0.00 | fedmedian | 0.9170 | 0.9000 | 1 |
| numpy | dirichlet | 0.50 | fedavg | 0.9185 | 0.8900 | 1 |
| numpy | dirichlet | 0.50 | fedmedian | 0.9135 | 0.8850 | 2 |
| numpy | dirichlet | 0.10 | fedavg | 0.9150 | 0.8950 | 5 |
| numpy | dirichlet | 0.10 | fedmedian | 0.9120 | 0.8950 | 9 |
| flwr | dirichlet | 0.10 | fedavg | skipped（本机 ray 无法启动，见下） |

**结论**：non-IID 程度（Dirichlet α 从 ∞→0.1）显著拖慢联邦收敛（1 → 9 rounds@90%）；
Byzantine 鲁棒策略（median / trimmed-mean）在干净的合成数据上收敛略慢于 FedAvg。

## 架构

调用单向无环：`cli → pipeline → {data, training, hpo, fed, eval, flwr_backend} → core`
详见 [docs/architecture.md](docs/architecture.md)。

| 模块 | 职责 |
|------|------|
| `core/` | dataclass 类型、E100~E500 错误层级、ENV_FED_* 配置覆盖、Protocol 契约 |
| `data/` | 全局高斯混合合成数据 + IID / Dirichlet non-IID 划分（无拒绝采样，保证终止） |
| `fed/` | 纯 numpy softmax 回归（可精确聚合的 [W, b]）、联邦客户端、服务器、聚合策略 |
| `training/` | 联邦训练主循环（客户端采样、轮次评测、通信量核算） |
| `eval/` | accuracy / worst-client fairness / rounds-to-target / comm-MB |
| `hpo/` | Optuna TPE 搜索（lr / local_epochs / batch_size / strategy） |
| `flwr_backend/` | Flower 1.38 仿真后端（可选），失败自动降级 numpy |
| `pipeline/` | FedPipeline.run() / benchmark()，benchmark.json 落盘 |

## 离线兜底说明

flwr 1.38 的 Ray 仿真引擎在部分受限 Windows 环境无法启动（raylet 超时）。
系统按「SOTA 后端优先、离线兜底」设计：

- `flwr_runtime_usable()` 一次性探测 ray 可用性；不可用时 benchmark 明确标注
  `skipped: flwr/ray runtime unavailable`，不伪造数字。
- numpy 后端（FedAvg/FedMedian/FedTrimmedMean，算法出自 McMahan 2017 / Yin 2018 /
  Blanchard 2017 / Hsu 2019）承载全部功能，零下载可跑。
- 在 ray 正常的机器上，flwr 后端自动启用（`tests/test_flwr_backend.py` 的 smoke 会执行）。

## 已知坑（实现期踩过）

1. Dirichlet 划分用拒绝采样会在类别池耗尽时**无限循环** → 改为「截断+补差」，
   总样本数严格守恒。
2. flwr 1.38 已废弃 `client_fn(cid)`，需 `client_fn(context)` + `context.node_config["partition-id"]`
   + `NumPyClient(...).to_client()`。
3. `flwr[simulation]` extra 在 Windows/py3.13 未拉起 ray，需显式 `pip install ray`。
4. 特征尺度过大（类中心间距 6σ）时 mini-batch SGD 会震荡 → 合成数据难度旋钮
   `class_sep` 需扫参定甜点（本项目 1.8）。

## 环境变量（ENV_FED_*）

`ENV_FED_NUM_CLIENTS` `ENV_FED_NUM_ROUNDS` `ENV_FED_CLIENTS_PER_ROUND`
`ENV_FED_LOCAL_EPOCHS` `ENV_FED_BATCH_SIZE` `ENV_FED_SEED` `ENV_FED_STRATEGY`
`ENV_FED_PARTITION` `ENV_FED_DIRICHLET_ALPHA` `ENV_FED_LR` 等，优先级：显式参数 > 环境 > 默认。

## License

MIT
