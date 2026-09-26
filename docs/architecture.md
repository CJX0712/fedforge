# FedForge 架构文档

> 作者：晨星 (CJX0712) · 版本 0.1.0 · 2026-09-27

## 1. 总体拓扑（单向无环）

```
cli.py ──► pipeline/ ──► ┬─ data/        (synthetic · partition)
                         ├─ training/    (loop)
                         ├─ hpo/         (optuna)
                         ├─ fed/         (model · client · server · strategies)
                         ├─ flwr_backend/(可选 SOTA 后端)
                         └─ eval/        (metrics)
                                  │
                                  ▼
                                core/   (types · errors · config · interfaces)
```

任何模块只允许向下依赖 core；core 不依赖任何其他模块。

## 2. 核心契约

### 2.1 权重格式（跨模块统一语义）
`Weights = [W, b]`：`W ∈ R^{d×k}`，`b ∈ R^k`。聚合策略对该结构做逐元素运算，
保证 FedAvg/Median/TrimmedMean 可互换且公平评测。

### 2.2 Protocol（core/interfaces.py）
- `AggregationStrategy.aggregate(weights, n_samples) -> weights`：纯函数。
- `FederatedClient`：`fit(global_weights, epochs, batch_size, lr, l2, seed)
  -> (new_weights, n_samples, comm_bytes_up)`；`evaluate(weights) -> (loss, acc, n)`。
  numpy 客户端与 flwr 客户端共用该语义。

### 2.3 错误层级（core/errors.py）
`E1xx Config` / `E2xx Data` / `E3xx Strategy` / `E4xx Train` / `E5xx Eval`，
全部继承 `FedForgeError`，CLI 层统一捕获输出 `[Exxx] message`。

## 3. 数据流

1. **生成**：`make_global_mixture` 定义 k 个类中心（圆周分布，半径 ∝ class_sep·k），
   每类多元高斯（共享协方差 = noise²·I）。全局测试集 10×单客户端样本量。
2. **划分**：池化样本 → `iid_partition`（均匀洗牌均分）或 `dirichlet_partition`
   （每客户端类比例 ~ Dirichlet(α)；want 截断到库存 + 补差，总样本严格守恒，无死循环）。
3. **训练**：`FedServer.run_round`：按 `clients_per_round` 无放回采样 →
   各客户端 `train_local`（mini-batch SGD，固定 seed 确定性）→
   策略聚合更新全局权重；通信量 = Σ(上传 + 下发) 字节数。
4. **评测**：每轮记录 global_acc / worst_client_acc（公平性）/ mean_client_acc /
   comm_bytes；round 0 为训练前基线。
5. **HPO**：Optuna TPESampler(seed)，目标 = 末 3 个 eval 点的全局精度均值
   （奖励稳定收敛而非单轮运气）。

## 4. 聚合策略

| 策略 | 公式 | 出处 | 特性 |
|------|------|------|------|
| FedAvg | Σ(nᵢ/Σn)·wᵢ | McMahan et al. 2017 | 样本数加权平均 |
| FedMedian | coord-wise median | Yin et al. 2018 | Byzantine 鲁棒 |
| FedTrimmedMean | 掐头去尾后取均值 | Blanchard 2017 / Yin 2018 | 每坐标剔除极端值 |

## 5. 双后端设计

- **numpy（默认，离线兜底）**：纯 numpy softmax 回归，权重全可控，确定性可复现。
- **flwr（可选 SOTA）**：`flwr.simulation.start_simulation` + `FedAvg` 策略 +
  Ray 引擎；`_SavingFedAvg` 子类在 `aggregate_fit` 中捕获最终全局权重。
  运行时一次性探测 `flwr_runtime_usable()`，失败 → benchmark 标注 skipped，绝不伪造。

## 6. 复现性

- 所有随机性来自 `np.random.default_rng(seed)`（数据 / 划分 / 客户端本地训练 /
  客户端采样 / Optuna 各自独立派生种子）。
- `benchmark.json` 不含墙钟时间，仅含 seed、版本号与结果数字。
- 实测：同 seed 两次运行 final_acc 完全一致（tests/test_training.py 固化该不变量）。

## 7. 量化基线（seed=42，15 rounds，8 clients，4 per round）

| 场景 | final_acc | worst_client_acc | rounds@90% |
|------|-----------|------------------|------------|
| IID | 0.9185 | 0.9000 | 1 |
| Dirichlet α=0.5 | 0.9185 | 0.8900 | 1(2 for robust) |
| Dirichlet α=0.1 | 0.9150 | 0.8950 | 5 (fedavg) / 9 (robust) |

单轮通信量 0.00238 MB（8×float64 [W∈R^{12×3}, b∈R³] × 双向 ×4 客户端）。
