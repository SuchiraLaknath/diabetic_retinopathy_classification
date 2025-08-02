#!/usr/bin/env python
"""
Fine-tune DeiT-Base (facebook/deit-base-patch16-224) on the diabetic-retinopathy
dataset stored as a single CSV (id_code, diagnosis) + images.
Save as train_deit_hf.py and `python train_deit_hf.py`
"""

# ---------------------------------------------------------------------
# 0. Std / 3rd-party imports
# ---------------------------------------------------------------------
import os, random, json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments

from transformers import (
    AutoImageProcessor,          # a.k.a. feature extractor
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import evaluate                 # for accuracy metric
# ---------------------------------------------------------------------
# 1. Custom Dataset that returns dict(pixel_values, labels)
# ---------------------------------------------------------------------
class DRRetinaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, image_processor,
                 train: bool = True):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.iproc = image_processor
        # --- minimal augments ---
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
        row = self.df.iloc[idx]
        img_path = self.img_dir / f"{row.id_code}.png"
        img = Image.open(img_path).convert("RGB")
        img = self.augment(img)

        # `return_tensors="pt"` gives shape (1,3,224,224); we squeeze to (3,224,224)
        pixel_values = self.iproc(img, return_tensors="pt").pixel_values.squeeze(0)
        label = int(row.diagnosis)
        return {"pixel_values": pixel_values, "labels": label}


# ---------------------------------------------------------------------
# 2. Paths, hyper-params, splits
# ---------------------------------------------------------------------
CSV_PATH = "data/aptos2019-blindness-detection/train.csv"   # your CSV
IMG_DIR  = "data/aptos2019-blindness-detection/train_images" 
OUTPUT_DIR = Path("deit_retina_ckpt")
VAL_FRAC  = 0.15          # 85 % train / 15 % val
SEED      = 42
BATCH     = 32
EPOCHS    = 30
LR        = 5e-4
WD        = 1e-4
set_seed(SEED)

# ---------------------------------------------------------------------
# 3. Prepare DataFrames & label info
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
num_labels = df["diagnosis"].nunique()
train_df, val_df = train_test_split(
    df,
    test_size   = VAL_FRAC,
    stratify    = df["diagnosis"],
    random_state= SEED,
)

# ---------------------------------------------------------------------
# 4. Load processor & model
# ---------------------------------------------------------------------
processor = AutoImageProcessor.from_pretrained(
    "facebook/deit-base-patch16-224",
    # use_safetensors=True
)
model = AutoModelForImageClassification.from_pretrained(
    "facebook/deit-base-patch16-224",
    use_safetensors=True,
    # num_labels = 5,
    # id2label   = {i: str(i) for i in range(num_labels)},
    # label2id   = {str(i): i for i in range(num_labels)},
)

# ---------------------------------------------------------------------
# 5. Build Dataset objects
# ---------------------------------------------------------------------
train_ds = DRRetinaDataset(train_df, IMG_DIR, processor, train=True)
val_ds   = DRRetinaDataset(val_df,   IMG_DIR, processor, train=False)

# ---------------------------------------------------------------------
# 6. Metric
# ---------------------------------------------------------------------
accuracy_metric  = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")   # you were missing this
recall_metric    = evaluate.load("recall")
f1_metric        = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=preds,
                                         references=labels,
                                         average="weighted")
    recall    = recall_metric.compute(predictions=preds,
                                      references=labels,
                                      average="weighted")
    f1        = f1_metric.compute(predictions=preds,
                                  references=labels,
                                  average="weighted")
    accuracy  = accuracy_metric.compute(predictions=preds,
                                        references=labels)

    # .compute() returns dicts like {'precision': 0.93}; unpack & merge:
    return {**accuracy, **precision, **recall, **f1}

# ---------------------------------------------------------------------
# 7. TrainingArguments & Trainer
# ---------------------------------------------------------------------
args = TrainingArguments(
    output_dir="./deit_results",
        num_train_epochs= 20,
        per_device_train_batch_size= 32,
        per_device_eval_batch_size=32,
        logging_dir="./deit_logs",
        logging_strategy="epoch",
        eval_strategy ="epoch",
        save_strategy="epoch",
        report_to=["tensorboard"],  
        load_best_model_at_end=True,
)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    data_collator   = default_data_collator,   # stacks pixel_values + labels
    compute_metrics = compute_metrics,
)

# ---------------------------------------------------------------------
# 8. Train & save the best model
# ---------------------------------------------------------------------
train_results = trainer.train()
print(train_results.metrics)

print("\n✅ Training complete. Evaluating best checkpoint…")
metrics = trainer.evaluate()
print(metrics)

# best model is already loaded (default). Save weights + processor for later use
processor.save_pretrained(OUTPUT_DIR / "preprocessor")
trainer.save_model(OUTPUT_DIR / "best_model")  # will create best_model/ with pytorch_model.bin

