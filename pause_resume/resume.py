"""
resume.py - 恢复暂停的训练

使用方法：
    # 自动检测最新训练并恢复
    python pause_resume/resume.py

    # 指定 checkpoint 路径恢复
    python pause_resume/resume.py --checkpoint outputs/.../checkpoints/last.pt

    # 恢复后50分钟自动暂停
    python pause_resume/resume.py --resume-time 50

    # 仅显示 checkpoint 信息，不启动训练
    python pause_resume/resume.py --info

功能：
    1. 从上次保存的 checkpoint 继续训练
    2. 完全恢复训练状态（模型、优化器、调度器、epoch、历史记录）
    3. 训练质量与不间断训练完全一致
    4. 支持定时暂停功能

恢复的状态包括：
    - 模型参数
    - 优化器状态（包括动量）
    - 学习率调度器状态
    - 当前 epoch
    - 训练/验证损失历史
    - 最佳模型指标
"""

import os
import sys
import argparse
import time
import subprocess

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def find_latest_training_dir(base_dir: str = "outputs"):
    """查找最新的训练目录"""
    outputs_dir = os.path.join(project_root, base_dir)

    if not os.path.exists(outputs_dir):
        return None

    timestamp_dirs = []
    for dir_name in os.listdir(outputs_dir):
        if dir_name.count('_') >= 2:
            dir_path = os.path.join(outputs_dir, dir_name)
            if os.path.isdir(dir_path):
                timestamp_dirs.append((dir_path, dir_name))

    if not timestamp_dirs:
        return None

    timestamp_dirs.sort(key=lambda x: x[1], reverse=True)
    return timestamp_dirs[0][0]


def find_saved_config(training_dir: str) -> str:
    """在训练目录中查找保存的 config（兼容 config_snapshot.yaml 和 config.yaml）"""
    for name in ('config_snapshot.yaml', 'config.yaml'):
        path = os.path.join(training_dir, name)
        if os.path.exists(path):
            return path
    return None


def get_latest_checkpoint_from_dir(checkpoint_dir: str) -> str:
    """从目录中查找最新的 checkpoint 文件（优先 last.pt）"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pt'):
            filepath = os.path.join(checkpoint_dir, filename)
            mtime = os.path.getmtime(filepath)
            checkpoints.append((filepath, mtime, filename))

    if not checkpoints:
        return None

    for cp in checkpoints:
        if cp[2] == 'last.pt':
            return cp[0]

    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints[0][0]


def print_checkpoint_info(checkpoint_path: str):
    """打印 checkpoint 信息"""
    print("\n" + "=" * 60)
    print("Checkpoint 信息")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)

    print(f"文件: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")

    if 'best_val_mae' in checkpoint:
        print(f"最佳验证 MAE: {checkpoint['best_val_mae']:.4f}")
    elif 'best_val_loss' in checkpoint:
        print(f"最佳验证 Loss: {checkpoint['best_val_loss']:.4f}")

    print(f"训练历史长度: {len(checkpoint.get('train_losses', []))} epochs")

    if 'config' in checkpoint:
        cfg = checkpoint['config']
        total_epochs = cfg.get('training', {}).get('epochs', 100)
        current_epoch = checkpoint.get('epoch', 0)
        remaining = total_epochs - current_epoch
        print(f"总计划轮数: {total_epochs}")
        print(f"剩余轮数: {remaining}")

    print("=" * 60 + "\n")


def setup_auto_pause(out_dir: str, resume_minutes: float):
    """设置自动暂停"""
    if resume_minutes <= 0:
        return

    pause_flag = os.path.join(out_dir, '.pause')

    if os.path.exists(pause_flag):
        print(f"警告: 已存在暂停标志，将覆盖")
        try:
            os.remove(pause_flag)
        except OSError:
            pass

    target_time = time.time() + resume_minutes * 60
    try:
        with open(pause_flag, 'w') as f:
            f.write(str(target_time))

        hours = int(resume_minutes // 60)
        mins = int(resume_minutes % 60)
        if hours > 0:
            time_str = f"{hours}小时{mins}分钟"
        else:
            time_str = f"{mins}分钟"

        print(f"已设置自动暂停: {time_str} 后")
        print(f"  预计暂停时间: {time.ctime(target_time)}")
    except Exception as e:
        print(f"设置自动暂停失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='SingleHyperTKAN - 训练恢复工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pause_resume/resume.py                                    # 自动恢复最新训练
  python pause_resume/resume.py --checkpoint path/to/checkpoint.pt # 指定 checkpoint
  python pause_resume/resume.py --resume-time 50                   # 50分钟后自动暂停
  python pause_resume/resume.py --info                             # 仅显示信息
        """
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint文件路径')
    parser.add_argument('--config', type=str, default=None,
                        help='Config文件路径')
    parser.add_argument('--resume-time', type=float, default=0,
                        help='恢复后多少分钟自动暂停（0表示不自动暂停）')
    parser.add_argument('--info', action='store_true',
                        help='仅显示 checkpoint 信息，不启动训练')

    args = parser.parse_args()

    print("=" * 60)
    print("SingleHyperTKAN - 训练恢复工具")
    print("=" * 60)

    # --- 确定 checkpoint 路径 ---
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"指定的 checkpoint 不存在: {checkpoint_path}")
            return 1
        print(f"使用指定的 checkpoint: {checkpoint_path}")
    else:
        print("自动查找最新 checkpoint...")
        latest_dir = find_latest_training_dir()

        if not latest_dir:
            print("未找到可恢复的训练目录")
            print("  请确保已经完成过至少一次训练")
            return 1

        print(f"最新训练目录: {latest_dir}")

        checkpoint_dir = os.path.join(latest_dir, 'checkpoints')
        checkpoint_path = get_latest_checkpoint_from_dir(checkpoint_dir)

        if not checkpoint_path:
            print(f"在 {checkpoint_dir} 中找不到 checkpoint")
            return 1

        print(f"找到 Checkpoint: {checkpoint_path}")

    print_checkpoint_info(checkpoint_path)

    if args.info:
        return 0

    # --- 确定 config 路径 ---
    if args.config:
        config_path = args.config
        print(f"使用指定的 Config: {config_path}")
    else:
        ckpt_dir = os.path.dirname(checkpoint_path)
        training_dir = os.path.dirname(ckpt_dir)

        config_path = find_saved_config(training_dir)

        if not config_path or not os.path.exists(config_path):
            print("警告: 找不到保存的 config，使用默认 configs/config.yaml")
            config_path = os.path.join(project_root, 'configs', 'config.yaml')
        else:
            print(f"使用训练时保存的 Config: {config_path}")

    if not os.path.exists(config_path):
        print(f"Config 文件不存在: {config_path}")
        return 1

    # --- 设置自动暂停 ---
    if args.resume_time > 0:
        ckpt_dir = os.path.dirname(checkpoint_path)
        out_dir = os.path.dirname(ckpt_dir)
        print("\n" + "=" * 60)
        print("设置自动暂停")
        print("=" * 60)
        setup_auto_pause(out_dir, args.resume_time)

    # --- 启动恢复训练 ---
    print("\n" + "=" * 60)
    print("准备恢复训练...")
    print("=" * 60)
    print("\n将从 checkpoint 继续训练，状态包括:")
    print("  - 模型参数")
    print("  - 优化器状态（动量等）")
    print("  - 学习率调度器状态")
    print("  - 训练历史记录")
    print("  - 最佳模型记录")

    if args.resume_time > 0:
        print(f"\n将在 {args.resume_time} 分钟后自动暂停")

    print("\n训练质量将与不间断训练完全一致\n")

    cmd = [
        sys.executable,
        os.path.join(project_root, 'train.py'),
        '--config', config_path,
        '--resume', checkpoint_path
    ]

    print("=" * 60)
    print("启动训练命令:")
    print(" ".join(cmd))
    print("=" * 60)
    print()

    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("恢复训练完成!")
        print("=" * 60)
        return 0
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"\n恢复训练出错: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
