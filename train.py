"""
train.py
========

Training Script für den Event-basierten GNN Objektdetektor.

Basierend auf:
- AEGNN (Graph Preprocessing / Event Pipeline)
- DAGR (Detection Architektur und Trainingsinterface)
- Prophesee GEN1 Dataset

Verwendete Frameworks:
- PyTorch
- PyTorch Geometric
- Torch Scatter

------------------------------------------------------------
WICHTIG: Anpassungen vor dem Training
------------------------------------------------------------

1. Dataset Pfad anpassen:
   --data Pfad muss auf den lokalen GEN1 Datensatz zeigen.

2. config.yaml prüfen:
   - batch_size
   - num_workers
   - learning rate
   - hidden_dim

3. CUDA/GPU:
   Das Script erwartet CUDA Unterstützung.

4. Vorverarbeitung:
   AEGNN Preprocessing muss vorher ausgeführt werden,
   damit die .pkl Dateien existieren.

5. Erwartete Ordnerstruktur:

   data/
      processed/
         train/*.pkl
         val/*.pkl
      train/*.dat
      val/*.dat

------------------------------------------------------------
Quellen
------------------------------------------------------------

AEGNN:
https://github.com/uzh-rpg/aegnn

DAGR:
https://github.com/uzh-rpg/dagr

GEN1 Dataset:
https://www.prophesee.ai/2020/01/24/gen1-automotive-detection-dataset/
"""

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import yaml
import random
from pathlib import Path
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from gen1_dataset      import Gen1Dataset
from gen1_gnn_detector import Gen1GNNDetector


# ── Reproducibility ───────────────────────────────────────────
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ── Gradient Utilities ────────────────────────────────────────
def fix_gradients(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = torch.nan_to_num(p.grad, nan=0.0)


# ── Training Epoch ────────────────────────────────────────────
def train_epoch(loader, model, optimizer, scheduler,
                cfg, epoch, writer):
    model.train()
    total_loss = 0.

    for i, data in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
        data = data.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(data)
        loss      = loss_dict["total_loss"]

        loss.backward()
        torch.nn.utils.clip_grad_value_(
            model.parameters(), cfg['clip_grad'])
        fix_gradients(model)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if i % cfg['log_every'] == 0:
            step = epoch * len(loader) + i
            writer.add_scalar("train/loss",
                              loss.item(), step)
            writer.add_scalar("train/loss_box",
                              loss_dict["loss_box"].item(), step)
            writer.add_scalar("train/loss_obj",
                              loss_dict["loss_obj"].item(), step)
            writer.add_scalar("train/loss_cls",
                              loss_dict["loss_cls"].item(), step)
            writer.add_scalar("train/lr",
                              scheduler.get_last_lr()[-1], step)
            writer.add_scalar("train/gt_per_batch",
                              data.bbox.shape[0], step)

        # Detections alle 200 Batches
        if i % 200 == 0:
            model.eval()
            with torch.no_grad():
                dets, _ = model(data, compute_loss=False)
                scores  = dets[0][1]
                n_dets  = (scores > 0.1).sum().item() \
                          if len(scores) > 0 else 0
            writer.add_scalar("train/detections_per_batch",
                              n_dets, epoch * len(loader) + i)
            model.train()

    return total_loss / len(loader)


# ── Validation Epoch ──────────────────────────────────────────
def val_epoch(loader, model):
    model.eval()
    total_loss = 0.

    with torch.no_grad():
        for data in tqdm(loader, desc="Validation"):
            data = data.cuda()
            loss_dict, _ = model(data, compute_loss=True)
            total_loss += loss_dict["total_loss"].item()

    return total_loss / len(loader)


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--data',   default='data')
    parser.add_argument('--output', default='runs/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))

    # ── Output Ordner + Writer ────────────────────────────────
    out_dir = Path(args.output) / cfg.get('run_name', 'baseline')
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    # ── Datasets ──────────────────────────────────────────────
    print("Loading datasets...")
    train_ds = Gen1Dataset(root=args.data, split="train",
                           max_edges=cfg['max_edges'])
    val_ds   = Gen1Dataset(root=args.data, split="val",
                           max_edges=cfg['max_edges'])

    train_loader = DataLoader(
        train_ds,
        batch_size   = cfg['batch_size'],
        shuffle      = True,
        num_workers  = cfg['num_workers'],
        drop_last    = True,
        follow_batch = ['bbox'],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size   = cfg['batch_size'],
        shuffle      = False,
        num_workers  = cfg['num_workers'],
        drop_last    = False,
        follow_batch = ['bbox'],
    )

    # ── Modell ────────────────────────────────────────────────
    print("Building model...")
    model = Gen1GNNDetector(
        hidden_dim  = cfg['hidden_dim'],
        num_classes = 2,
    ).cuda()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter: {n_params:,}")
    writer.add_scalar("model/params", n_params, 0)

    # ── Optimizer + Scheduler ─────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg['lr'],
        weight_decay = cfg['weight_decay'],
    )

    total_steps = cfg['epochs'] * len(train_loader)

    def lr_lambda(step):
        warmup = int(0.05 * total_steps)
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (total_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda)

    best_val_loss = float('inf')

    # ── Checkpoint wiederherstellen ───────────────────────────
    resume_path = str(out_dir / "last.pth")
    start_epoch = 0

    if os.path.exists(resume_path):
        print(f"Lade Checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cuda')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        print(f"Starte bei Epoche {start_epoch}")
    else:
        print("Kein Checkpoint — starte von Anfang")

    # ── Training Loop ─────────────────────────────────────────
    print("Starting training...")
    for epoch in range(start_epoch, cfg['epochs']):

        train_loss = train_epoch(
            train_loader, model, optimizer,
            scheduler, cfg, epoch, writer)

        val_loss = val_epoch(val_loader, model)

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss",   val_loss,   epoch)

        print(f"Epoch {epoch:3d} | "
              f"Train: {train_loss:.4f} | "
              f"Val:   {val_loss:.4f}")

        ckpt = {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss':  val_loss,
            'cfg':       cfg,
        }
        torch.save(ckpt, out_dir / 'last.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, out_dir / 'best.pth')
            print(f"  → Neues Best Model gespeichert "
                  f"(val_loss={val_loss:.4f})")

    writer.close()
    print("Training abgeschlossen.")


if __name__ == '__main__':
    main()

