import os
import time
import torch
from tqdm import tqdm
from ..utils.metrics import compute_metrics
from ..utils.checkpoint import save_checkpoint, load_checkpoint
from ..utils.visualization import plot_loss_curve


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler,
                 loss_fn, H, W, device, config, preprocessor=None,
                 output_dir=None, weather_dim: int = 1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.H = H.to(device)
        self.W = W.to(device)
        self.device = device
        self.config = config
        self.preprocessor = preprocessor
        self.weather_dim = weather_dim

        self.epochs = config['training']['epochs']
        self.grad_clip = config['training']['grad_clip']
        self.use_amp = config['training']['use_amp']
        self.accum_steps = config['training']['accumulation_steps']
        self.step_weighted_mae = config['training']['loss'].get(
            'step_weighted_mae', False)

        self.current_epoch = 0
        self.best_val_mae = float('inf')
        self.bad_epochs = 0
        self.patience = config['training']['early_stopping']['patience']
        self.min_delta = config['training']['early_stopping']['min_delta']
        self.train_losses = []
        self.val_losses = []

        self.output_dir = output_dir
        self.ckpt_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.scaler = (torch.amp.GradScaler('cuda')
                       if self.use_amp and device == 'cuda' else None)

    def _loss(self, pred, target):
        if not self.step_weighted_mae:
            return self.loss_fn(pred, target)
        T = pred.shape[1]
        w = torch.linspace(1.0, 1.0, T, device=pred.device)
        w = w / w.sum()
        step_mae = torch.mean(torch.abs(pred - target), dim=(0, 2, 3))
        return torch.sum(step_mae * w)

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        total = 0.0
        n = 0
        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {self.current_epoch + 1}/{self.epochs}",
                    colour='green')
        for i, batch in enumerate(pbar):
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            x_raw = x[:, :, :, :self.weather_dim]
            if self.scaler is not None:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    pred = self.model(x, self.H, self.W,
                                      output_length=y.shape[1],
                                      x_raw=x_raw)
                    loss = self._loss(pred, y) / self.accum_steps
                if not torch.isfinite(loss):
                    self.optimizer.zero_grad()
                    continue
                self.scaler.scale(loss).backward()
                if (i + 1) % self.accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                pred = self.model(x, self.H, self.W,
                                  output_length=y.shape[1],
                                  x_raw=x_raw)
                loss = self._loss(pred, y) / self.accum_steps
                if not torch.isfinite(loss):
                    self.optimizer.zero_grad()
                    continue
                loss.backward()
                if (i + 1) % self.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            lv = loss.item() * self.accum_steps
            total += lv
            n += 1
            pbar.set_postfix({'loss': f'{lv:.4f}'})
        return total / max(n, 1)

    def validate(self):
        self.model.eval()
        total = 0.0
        n = 0
        preds, trues = [], []
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['x'].to(self.device)
                y = batch['y'].to(self.device)
                x_raw = x[:, :, :, :self.weather_dim]
                if self.scaler is not None:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        pred = self.model(x, self.H, self.W,
                                          output_length=y.shape[1],
                                          x_raw=x_raw)
                        loss = self._loss(pred, y)
                else:
                    pred = self.model(x, self.H, self.W,
                                      output_length=y.shape[1],
                                      x_raw=x_raw)
                    loss = self._loss(pred, y)
                total += loss.item()
                n += 1
                preds.append(pred.float().cpu())
                trues.append(y.float().cpu())
        pred = torch.cat(preds, 0)
        true = torch.cat(trues, 0)
        if self.preprocessor is not None and self.preprocessor.fitted:
            pred = torch.from_numpy(
                self.preprocessor.inverse_transform(pred.numpy())).float()
            true = torch.from_numpy(
                self.preprocessor.inverse_transform(true.numpy())).float()
        m = compute_metrics(pred, true,
                            metrics=self.config['evaluation']['metrics'])
        step_mae = {}
        for t in range(pred.shape[1]):
            step_mae[t + 1] = torch.mean(
                torch.abs(pred[:, t:t + 1] - true[:, t:t + 1])).item()
        return {'loss': total / max(n, 1), **m, 'step_mae': step_mae}

    def save_ckpt(self, is_best=False, logger=None):
        state = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (self.scheduler.state_dict()
                                     if self.scheduler else None),
            'best_val_mae': self.best_val_mae,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
        }
        save_checkpoint(state, self.ckpt_dir, is_best=is_best)
        if logger:
            if is_best:
                logger.info(f"  Checkpoint saved: last.pt [best_model.pt updated]")
            else:
                logger.info(f"  Checkpoint saved: last.pt")

    def check_pause_flag(self) -> bool:
        """检查是否存在暂停标志文件且已到达暂停时间。"""
        pause_flag = os.path.join(self.output_dir, '.pause')
        if not os.path.exists(pause_flag):
            return False
        try:
            with open(pause_flag, 'r') as f:
                pause_time = float(f.read().strip())
            if time.time() >= pause_time:
                return True
            return False
        except Exception:
            return False

    def clear_pause_flag(self):
        """清除暂停标志文件。"""
        pause_flag = os.path.join(self.output_dir, '.pause')
        if os.path.exists(pause_flag):
            try:
                os.remove(pause_flag)
            except OSError:
                pass

    def resume_training(self, path):
        ckpt = load_checkpoint(path, self.model, self.optimizer,
                               self.scheduler, self.device)
        self.current_epoch = ckpt.get('epoch', 0)
        self.best_val_mae = ckpt.get(
            'best_val_mae', ckpt.get('best_val_loss', float('inf')))
        self.train_losses = ckpt.get('train_losses', [])
        self.val_losses = ckpt.get('val_losses', [])

    def train(self, resume_from=None, logger=None):
        if resume_from:
            self.resume_training(resume_from)
            if logger:
                logger.info(f"Resumed from epoch {self.current_epoch}")
        t0 = time.time()
        for ep in range(self.current_epoch, self.epochs):
            self.current_epoch = ep
            tr = self.train_epoch()
            vm = self.validate()
            vl = vm['loss']
            val_mae = vm.get('mae', float('inf'))
            self.train_losses.append(tr)
            self.val_losses.append(vl)

            if self.scheduler is not None:
                if isinstance(self.scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(vl)
                else:
                    self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            msg = (f"Epoch {ep + 1}/{self.epochs} | "
                   f"train_loss={tr:.4f} val_loss={vl:.4f} "
                   f"val_mae={val_mae:.4f} "
                   f"rmse={vm.get('rmse', 0):.4f} lr={lr:.2e}")

            is_best = val_mae < self.best_val_mae - self.min_delta
            if is_best:
                self.best_val_mae = val_mae
                self.bad_epochs = 0
                msg += f" \033[92m★ New Best\033[0m"  # 绿色显示
            else:
                self.bad_epochs += 1

            if logger:
                logger.info(msg)
                for k, v in vm['step_mae'].items():
                    logger.info(f"  val_mae_step_{k}: {v:.4f}")
            else:
                print(msg)

            # ---- 打印 W_dynamic 统计（整轮累积） ----
            get_fn = getattr(self.model, 'get_w_dynamic_stats', None)
            if get_fn is not None:
                w_stats = get_fn()
                if w_stats is not None:
                    dyn_msg = (
                        f"  [W_dynamic] min={w_stats['min']:.4f}  "
                        f"max={w_stats['max']:.4f}  "
                        f"mean={w_stats['mean']:.4f}  "
                        f"std_across_batches={w_stats['std_of_batch_means']:.4f}  "
                        f"(accumulated over {w_stats['num_batches']} batches)")
                    if logger:
                        logger.info(dyn_msg)
                    else:
                        print(dyn_msg)

            self.save_ckpt(is_best=is_best, logger=logger)
            plot_loss_curve(self.train_losses, self.val_losses,
                            os.path.join(self.output_dir, 'loss_curve.png'),
                            'Loss Curve')

            # --- 暂停标志检测 ---
            if self.check_pause_flag():
                _log = logger.warning if logger else print
                _log("=" * 60)
                _log("PAUSE SIGNAL RECEIVED - Pausing training")
                _log("=" * 60)
                _log(f"Completed {ep + 1} epochs")
                _log(f"Current best val MAE: {self.best_val_mae:.4f}")
                self.save_ckpt(is_best=False, logger=logger)
                self.clear_pause_flag()
                _log("=" * 60)
                _log("Checkpoint saved successfully!")
                _log("To resume training, use:")
                _log(f"  python train.py --resume "
                     f"{os.path.join(self.ckpt_dir, 'last.pt')}")
                _log("Or:")
                _log("  python pause_resume/resume.py")
                _log("=" * 60)
                break

            if self.bad_epochs >= self.patience:
                if logger:
                    logger.info(
                        f"Early stopping triggered at epoch {ep + 1} "
                        f"(patience={self.patience})")
                break

        elapsed = (time.time() - t0) / 60
        if logger:
            logger.info(
                f"Training finished in {elapsed:.1f} min, "
                f"best_val_mae={self.best_val_mae:.4f}")
