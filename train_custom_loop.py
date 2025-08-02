#!/usr/bin/env python
"""
Custom training loop for DeiT-Base on the diabetic-retinopathy dataset.
Save as train_deit_custom.py and run with:
    python train_deit_custom.py
"""

import os
import random
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

import evaluate

# ---------------------------------------------------------------------
# 0. Hyperparameters & Paths
# ---------------------------------------------------------------------
CSV_PATH    = "data/aptos2019-blindness-detection/train.csv"
IMG_DIR     = "data/aptos2019-blindness-detection/train_images"
OUTPUT_DIR  = Path("deit_retina_ckpt")
LOG_DIR     = "./tensorboard_logs/at_04"
VAL_FRAC    = 0.2
SEED        = 42
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 5e-5
WD          = 1e-4
WARMUP_FRAC = 0.1   # fraction of total steps
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------
class DRRetinaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, processor, train: bool = True):
        self.df      = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.iproc   = processor
        if train:
            self.augment = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            self.augment = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = self.img_dir / f"{row.id_code}.png"
        img      = Image.open(img_path).convert("RGB")
        img      = self.augment(img)
        pixel_values = self.iproc(img, return_tensors="pt").pixel_values.squeeze(0)
        label = int(row.diagnosis)
        return {"pixel_values": pixel_values, "labels": label}


# ---------------------------------------------------------------------
# 2. Prepare data
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(
    df, test_size=VAL_FRAC, stratify=df.diagnosis, random_state=SEED
)

processor = AutoImageProcessor.from_pretrained("facebook/deit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained(
    "facebook/deit-base-patch16-224",
    num_labels=5,
    id2label={i: str(i) for i in range(5)},
    label2id={str(i): i for i in range(5)},
    use_safetensors=True,
    ignore_mismatched_sizes=True, 
)

# 3) Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# 4) Unfreeze only the classification head
for param in model.classifier.parameters():
    param.requires_grad = True

# 5) (Optional) Unfreeze the last k transformer blocks as well
k = 15 # e.g. last two encoder layers
# DeiT wraps a ViT under .deit; fallback to .vit if needed
encoder = getattr(model, "deit", None) or model.vit  
num_layers = model.config.num_hidden_layers

for idx in range(num_layers - k, num_layers):
    for param in encoder.encoder.layer[idx].parameters():
        param.requires_grad = True
model.to(DEVICE)

train_ds = DRRetinaDataset(train_df, IMG_DIR, processor, train=True)
val_ds   = DRRetinaDataset(val_df,   IMG_DIR, processor, train=False)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=default_data_collator, num_workers=4
)
val_loader   = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=default_data_collator, num_workers=4
)

# ---------------------------------------------------------------------
# 3. Optimizer, scheduler, metrics
# ---------------------------------------------------------------------
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)



# optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

# replace—or sit alongside—your existing linear scheduler:
plateau_scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",          # we want to maximize F1
    factor=0.2,          # multiply LR by 0.5 on plateau
    patience=2,          # wait 2 epochs without improvement
    threshold=1e-3,      # minimal change to count as improvement
    # verbose=True,        # prints a message when LR is reduced
    min_lr=1e-7          # don’t go below this
)

total_steps    = EPOCHS * len(train_loader)
warmup_steps   = int(WARMUP_FRAC * total_steps)
scheduler      = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

accuracy_metric  = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric    = evaluate.load("recall")
f1_metric        = evaluate.load("f1")

def compute_metrics(preds, labels):
    prec   = precision_metric.compute(predictions=preds, references=labels, average="weighted")
    rec    = recall_metric.compute(predictions=preds, references=labels, average="weighted")
    f1     = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    acc    = accuracy_metric.compute(predictions=preds, references=labels)
    return {**acc, **prec, **rec, **f1}

# ---------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------
writer = SummaryWriter(LOG_DIR)
global_step = 0
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # --- Training ---
    model.train()
    print(f"Epoch {epoch} Training Step")
    epoch_loss = 0.0
    for batch in tqdm(train_loader):
        batch = {k:v.to(DEVICE) for k,v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        writer.add_scalar("train/loss", loss.item(), global_step)
        global_step += 1

    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} | Train loss: {avg_train_loss:.4f}")

    # --- Evaluation ---
    model.eval()
    print(f"Epoch {epoch} Evaluation Step")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k:v.to(DEVICE) for k,v in batch.items()}
            logits = model(**batch).logits
            preds  = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    metrics = compute_metrics(all_preds, all_labels)

    print(
        f"Epoch {epoch} | "
        f"Val Acc: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    )

    # log metrics
    for k, v in metrics.items():
        writer.add_scalar(f"eval/{k}", v, epoch)

    # save best
    if metrics["accuracy"] > best_acc:
        best_acc = metrics["accuracy"]
        print(f"New best accuracy: {best_acc:.4f}. Saving model…")
        model.save_pretrained(OUTPUT_DIR / "best_model")
        processor.save_pretrained(OUTPUT_DIR / "preprocessor")

writer.close()
print("✅ Training complete.")
