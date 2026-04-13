# SingleHyper-TKAN（单超图 + TKAN）

本项目在 `D:\bishe\hyper_tkan` 下实现了一个可训练、可预测、可复现实验的气象预测框架：

- 空间：**单超图**（语义相似度 + 地理相似度在构图前融合）
- 时序：**TKAN**（含 RKAN 子层 + LSTM-like 门控）
- 输出：**非自回归一次性输出未来 12 步**

参考与对齐对象：`D:\bishe\code\hyper_kan`

输出目录：`D:\bishe\hyper_tkan\outputs`

---

## 1. 与 hyper_kan 的关系与差异

### 对齐部分
- 数据读取与窗口协议（input=12, output=12）
- 标准化/反标准化流程
- 训练/预测脚本风格与参数（`train.py` / `predict.py` / `main.py`）
- 输出目录与核心产物（checkpoint、metrics、predictions、loss_curve）

### 差异部分
- `hyper_kan` 为双超图分支卷积（邻域 + 语义）
- 本项目为**单超图**：先融合相似度，再构图，再做单分支超图卷积
- 时序从 GRU 改为 TKAN

---

## 2. 模型设计

## 2.1 单超图构图

\[
S^{fusion} = \alpha \hat S^{sem} + (1-\alpha)\hat S^{geo}
\]

- 语义相似度：仅用训练集统计特征（mean/std/min/max/median/range/diff_mean/diff_std）
- 地理相似度：仅用经纬度（Haversine 距离转相似度）
- 变规模超边：阈值选择 + `min_hyperedge_size` / `max_hyperedge_size` 保护

## 2.2 空间卷积

标准超图归一化聚合：
\[
X' = \Phi(D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X)
\]

时间维按步处理：每个输入时间步独立做单超图卷积。

## 2.3 TKAN

实现包含：
- `KANLinear`
- `RKANSubLayer`
- `TKANCell`
- `TKANLayer`

输出门由多个 RKAN 子层拼接后线性映射得到，不是普通 GRU/LSTM 改名。

---

## 3. 目录结构

```text
hyper_tkan/
├── configs/config.yaml
├── src/
│   ├── data/
│   ├── graph/
│   ├── models/
│   ├── training/
│   └── utils/
├── outputs/
├── data/cache/
├── visuals/
├── train.py
├── predict.py
├── main.py
├── requirements.txt
└── test/smoke_test.py
```

---

## 4. 环境安装

```bash
pip install -r requirements.txt
```

---

## 5. 训练

```bash
python train.py --config configs/config.yaml --gpu 0
```

支持参数：
- `--config`
- `--gpu`
- `--resume`
- `--device`
- `--dataset`

---

## 6. 预测

```bash
python predict.py --config configs/config.yaml
```

可选：
- `--checkpoint` 指定模型
- `--output_dir` 指定实验目录
- `--gpu` / `--device`

---

## 7. 一键 train+predict

```bash
python main.py all --config configs/config.yaml --gpu 0
```

---

## 8. 输出内容说明

每次训练会在 `outputs/<timestamp>_<Element>/` 下生成：
- `config_snapshot.yaml`
- `preprocessor.pkl`
- `checkpoints/best_model.pt`
- `checkpoints/last.pt`
- `train_*.log`
- `loss_curve.png`
- `predictions.npz`（预测后）
- `metrics.json`（预测后）
- `test_summary.txt`（预测后）

---

## 9. RTX 5070 8GB 推荐配置

### A. 快速调试档
- `num_stations: 128`
- `train_sample_ratio: 0.1`
- `epochs: 3`
- `batch_size: 4`

### B. 8GB 推荐档（默认）
- `num_stations: 512`
- `train_sample_ratio: 0.4~1.0`
- `epochs: 50~100`
- `batch_size: 4`
- `accumulation_steps: 4`
- `use_amp: true`

### C. 全量尝试档
- `num_stations: 1024+`（可能 OOM）
- 建议同时减小：`batch_size`、`hidden_size`，或增大梯度累积

---

## 10. 常见问题

1. OOM：降低 `batch_size`，减少 `num_stations`，降低 `hidden_size`
2. 训练慢：先用快速调试档确认流程
3. 缓存冲突：清理 `data/cache` 后重跑
4. 结果波动：固定 seed，并启用 deterministic

---

## 11. 可复现性

- `config_snapshot.yaml` 固化实验参数
- 训练中固定随机种子
- 保存 best/last checkpoint

---

## 12. 与 baseline 对比注意事项

为保证公平对比，请保持：
- 相同数据划分
- 相同窗口长度（12→12）
- 相同反归一化评估口径
- 尽量与 `D:\bishe\code\hyper_kan` 一致的数据与日志流程
