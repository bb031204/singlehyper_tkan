# 训练暂停与恢复功能

## 快速开始

**请确保在项目根目录 `D:\bishe\hyper_tkan` 下执行命令！**

```bash
cd D:\bishe\hyper_tkan

# 立即暂停
python pause_resume/pause.py

# 60分钟后暂停
python pause_resume/pause.py --pause-time 60

# 恢复训练
python pause_resume/resume.py

# 查看 checkpoint 信息
python pause_resume/resume.py --info
```

---

## 功能说明

### 1. 暂停训练 (`pause.py`)

在指定时间后完成当前 epoch 并自动暂停，保存训练状态。

```bash
python pause_resume/pause.py                  # 立即暂停
python pause_resume/pause.py --pause-time 5   # 5分钟后暂停
python pause_resume/pause.py --pause-time 120 # 2小时后暂停
```

**工作原理：**
1. 创建 `.pause` 标志文件，包含目标时间戳
2. 训练器在每个 epoch 后检查该文件
3. 到达指定时间后，完成当前 epoch
4. 自动保存 checkpoint 并退出
5. 自动清除 `.pause` 标志文件

### 2. 恢复训练 (`resume.py`)

自动查找最新的训练结果并恢复训练。

```bash
python pause_resume/resume.py                                      # 自动恢复
python pause_resume/resume.py --checkpoint path/to/checkpoint.pt   # 指定 checkpoint
python pause_resume/resume.py --resume-time 50                     # 恢复后50分钟自动暂停
python pause_resume/resume.py --info                               # 仅查看信息
python pause_resume/resume.py --config configs/config.yaml         # 指定 config
```

**恢复的状态包括：**
- 模型参数
- 优化器状态（包括动量）
- 学习率调度器状态
- 当前 epoch
- 训练/验证损失历史
- 最佳验证 MAE 记录

---

## 完整工作流程示例

### 场景1：长时间训练分批进行

```bash
# 第1天：训练2小时
python pause_resume/pause.py --pause-time 120

# ... 训练进行中，自动暂停 ...

# 第2天：继续训练
python pause_resume/resume.py

# 第3天：再次继续，训练1小时后暂停
python pause_resume/resume.py --resume-time 60
```

### 场景2：快速查看进度

```bash
python pause_resume/resume.py --info
```

### 场景3：指定特定 checkpoint 恢复

```bash
python pause_resume/resume.py --checkpoint outputs/.../checkpoints/checkpoint_epoch_10.pt
```

---

## 注意事项

- 恢复训练时会自动使用训练时保存的 `config_snapshot.yaml`
- 暂停信号在当前 epoch 结束后才生效，不会中断正在进行的 epoch
- 如果修改了 config，需要用 `--config` 明确指定
- 训练质量与不间断训练完全一致
