"""
Train Router Model using Soft Labels
-----------------------------------
- Uses KL Divergence loss
- Logs metrics to TensorBoard + W&B
- Saves checkpoints per epoch
- Saves BEST model based on lowest loss
"""

# =========================
# IMPORTS
# =========================
import os
import json
import glob
import random
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import wandb


# =========================
# CONFIGURATION
# =========================
LEVELS = ["half", "1", "2", "4", "8"]
OUTPUT_DIM = len(LEVELS)

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FOLDER = str(BASE_DIR / "medrag" / "soft_labels")

LOG_DIR = str(BASE_DIR / "logs" / "router_exp")
CHECKPOINT_DIR = str(BASE_DIR / "checkpoints")

BATCH_SIZE = 64
EMBEDDING_BATCH_SIZE = 128
EPOCHS = 30
LR = 5e-5
SEED = 42
K_FOLDS = 5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2
EARLY_STOPPING_PATIENCE = 6

# =========================
# UTILITIES
# =========================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# DATASET
# =========================
class SoftLabelDataset(Dataset):
    def __init__(self, folder_path: str):
        self.samples: List[Dict] = []

        files = glob.glob(os.path.join(folder_path, "*.json"))
        if not files:
            raise ValueError(f"No JSON files found in {folder_path}")

        for file in files:
            print(f"[INFO] Loading {file}")
            with open(file, "r") as f:
                data = json.load(f)
                self.samples.extend(data)

        if not self.samples:
            raise ValueError(f"No samples found in JSON files under {folder_path}")

        print(f"[INFO] Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        soft_label = item["soft_label"]

        if len(soft_label) != OUTPUT_DIM:
            raise ValueError(
                f"Expected soft_label of length {OUTPUT_DIM}, got {len(soft_label)}"
            )

        soft_label_tensor = torch.tensor(soft_label, dtype=torch.float32)
        if torch.any(soft_label_tensor < 0):
            raise ValueError("soft_label values must be non-negative")

        label_sum = soft_label_tensor.sum()
        if not torch.isclose(label_sum, torch.tensor(1.0), atol=1e-4):
            raise ValueError(
                f"soft_label values must sum to 1.0, got {label_sum.item():.6f}"
            )

        return {
            "question": item["question"],
            "soft_label": soft_label_tensor,
        }


class EncodedDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "soft_label": self.labels[idx],
        }


# =========================
# MODEL
# =========================
class Router(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, output_dim),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# =========================
# TRAINER
# =========================
class RouterTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        # =========================
        # W&B INIT
        # =========================
        wandb.init(
            project="router-model",
            name=f"lr_{LR}_bs_{BATCH_SIZE}_ep_{EPOCHS}",
            config={
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "model": "stsb-roberta-large",
                "k_folds": K_FOLDS,
                "weight_decay": WEIGHT_DECAY,
                "dropout": DROPOUT,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            },
            sync_tensorboard=True
        )

        # Dataset
        full_dataset = SoftLabelDataset(DATA_FOLDER)

        if len(full_dataset) < 2:
            raise ValueError(
                "Need at least 2 samples to run cross-validation"
            )

        self.dataset = full_dataset
        self.num_folds = min(K_FOLDS, len(full_dataset))

        # Encoder
        self.encoder = SentenceTransformer(
            "stsb-roberta-large",
            device=self.device
        )

        sample_embedding = self.encoder.encode(
            self.dataset[0]["question"],
            convert_to_tensor=True
        )
        self.input_dim = sample_embedding.shape[0]
        self.criterion = nn.KLDivLoss(reduction="batchmean", log_target=False)
        self.encoded_dataset = self._precompute_embeddings()

        # TensorBoard
        self.writer = SummaryWriter(LOG_DIR)

        self.global_step = 0
        self.best_loss = float("inf")

    def _precompute_embeddings(self):
        print("[INFO] Precomputing sentence embeddings...")
        questions = [sample["question"] for sample in self.dataset.samples]
        labels = torch.stack([self.dataset[idx]["soft_label"] for idx in range(len(self.dataset))])
        embeddings = self.encoder.encode(
            questions,
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=False,
        ).cpu()
        print(f"[INFO] Cached embeddings for {len(questions)} samples")
        return EncodedDataset(embeddings, labels)

    def _build_fold_indices(self):
        dataset_len = len(self.encoded_dataset)
        indices = torch.randperm(dataset_len, generator=torch.Generator().manual_seed(SEED)).tolist()
        fold_sizes = [dataset_len // self.num_folds] * self.num_folds
        for i in range(dataset_len % self.num_folds):
            fold_sizes[i] += 1

        folds = []
        start = 0
        for fold_size in fold_sizes:
            end = start + fold_size
            folds.append(indices[start:end])
            start = end
        return folds

    def _setup_fold(self, fold_idx: int, train_indices: List[int], val_indices: List[int]):
        self.train_loader = DataLoader(
            Subset(self.encoded_dataset, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device == "cuda",
        )

        self.val_loader = DataLoader(
            Subset(self.encoded_dataset, val_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device == "cuda",
        )

        self.model = Router(self.input_dim, OUTPUT_DIM).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
        self.fold_best_loss = float("inf")
        self.epochs_without_improvement = 0

        print(
            f"[INFO] Fold {fold_idx + 1}/{self.num_folds}: "
            f"train={len(train_indices)} val={len(val_indices)}"
        )

    #validation function
    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                embeddings = batch["embedding"].to(self.device, non_blocking=True)
                targets = batch["soft_label"].to(self.device)

                preds = self.model(embeddings)

                loss = self.criterion(preds, targets)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train_fold(self, fold_idx: int):
        for epoch in range(EPOCHS):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Fold {fold_idx + 1}/{self.num_folds} Epoch {epoch+1}"
            )

            for batch in progress_bar:
                embeddings = batch["embedding"].to(self.device, non_blocking=True)
                targets = batch["soft_label"].to(self.device)

                preds = self.model(embeddings)

                loss = self.criterion(preds, targets)

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

                # TensorBoard
                self.writer.add_scalar(
                    "Loss/train_step",
                    loss.item(),
                    self.global_step
                )

                # W&B
                wandb.log(
                    {
                        "fold": fold_idx + 1,
                        "train_loss_step": loss.item(),
                    }
                )

                self.global_step += 1

            avg_loss = total_loss / len(self.train_loader)

            val_loss = self.validate()
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"[INFO] Fold {fold_idx + 1}/{self.num_folds} Epoch {epoch+1} "
                f"Train Loss: {avg_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6g}"
            )

            # TensorBoard
            self.writer.add_scalar(
                f"Fold_{fold_idx + 1}/Loss/train_epoch",
                avg_loss,
                epoch + 1
            )
            self.writer.add_scalar(f"Fold_{fold_idx + 1}/Loss/val_epoch", val_loss, epoch + 1)
            self.writer.add_scalar(f"Fold_{fold_idx + 1}/LR", current_lr, epoch + 1)

            # W&B
            wandb.log(
                {
                    "fold": fold_idx + 1,
                    "train_loss_epoch": avg_loss,
                }
            )
            wandb.log(
                {
                    "fold": fold_idx + 1,
                    "val_loss_epoch": val_loss,
                    "learning_rate": current_lr,
                }
            )

            # =========================
            # SAVE BEST MODEL
            # =========================
            if val_loss < self.fold_best_loss:
                self.fold_best_loss = val_loss
                self.epochs_without_improvement = 0
                self.best_loss = min(self.best_loss, val_loss)

                best_model_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"best_model_fold_{fold_idx + 1}.pt"
                )
                torch.save(self.model.state_dict(), best_model_path)

                print(
                    f"[BEST] Saved best model for fold {fold_idx + 1} at "
                    f"epoch {epoch+1} (val_loss={val_loss:.4f})"
                )
                wandb.log(
                    {
                        "fold": fold_idx + 1,
                        "best_loss": self.fold_best_loss,
                        "best_epoch": epoch + 1,
                    }
                )

                wandb.save(best_model_path)
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"[EARLY STOP] No validation improvement for "
                    f"{EARLY_STOPPING_PATIENCE} epochs. Stopping training."
                )
                break

        final_model_path = os.path.join(
            CHECKPOINT_DIR,
            f"router_fold_{fold_idx + 1}_final.pt"
        )
        torch.save(self.model.state_dict(), final_model_path)
        wandb.save(final_model_path)
        return self.fold_best_loss

    def train(self):
        folds = self._build_fold_indices()
        fold_best_losses = []

        for fold_idx in range(self.num_folds):
            val_indices = folds[fold_idx]
            train_indices = []
            for other_fold_idx, fold_indices in enumerate(folds):
                if other_fold_idx != fold_idx:
                    train_indices.extend(fold_indices)

            self._setup_fold(fold_idx, train_indices, val_indices)
            fold_best_loss = self.train_fold(fold_idx)
            fold_best_losses.append(fold_best_loss)

        mean_best_val_loss = sum(fold_best_losses) / len(fold_best_losses)
        print(f"[DONE] Mean best validation loss across {self.num_folds} folds: {mean_best_val_loss:.4f}")
        for fold_idx, fold_best_loss in enumerate(fold_best_losses):
            print(f"[DONE] Fold {fold_idx + 1} best validation loss: {fold_best_loss:.4f}")
        wandb.log({"mean_best_val_loss": mean_best_val_loss})

        self.writer.close()
        wandb.finish()


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    set_seed(SEED)

    trainer = RouterTrainer()
    trainer.train()
