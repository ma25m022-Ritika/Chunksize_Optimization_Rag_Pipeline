"""
Train Router Model using Soft Labels
-----------------------------------
- Uses KL Divergence loss
- Logs metrics to TensorBoard
- Saves checkpoints per epoch
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# =========================
# CONFIGURATION
# =========================
LEVELS = ["half", "1", "2", "4", "8"]
OUTPUT_DIM = len(LEVELS)

DATA_FOLDER = "./soft_labels"
LOG_DIR = "./runs/router_exp"
CHECKPOINT_DIR = "./checkpoints"

BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
SEED = 42


# =========================
# UTILITIES
# =========================
def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# DATASET
# =========================
class SoftLabelDataset(Dataset):
    """Loads soft-label JSON files"""

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

        print(f"[INFO] Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        return {
            "question": item["question"],
            "soft_label": torch.tensor(item["soft_label"], dtype=torch.float32),
        }


# =========================
# MODEL
# =========================
class Router(nn.Module):
    """Simple MLP Router"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1),
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

        # Dataset
        self.dataset = SoftLabelDataset(DATA_FOLDER)
        self.loader = DataLoader(
            self.dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
        )

        # Encoder (outside model for efficiency)
        self.encoder = SentenceTransformer(
            "stsb-roberta-large",
            device=self.device
        )

        # Get embedding dimension
        sample_embedding = self.encoder.encode(
            self.dataset[0]["question"],
            convert_to_tensor=True
        )
        input_dim = sample_embedding.shape[0]

        # Model
        self.model = Router(input_dim, OUTPUT_DIM).to(self.device)

        # Loss function
        self.criterion = nn.KLDivLoss(reduction="batchmean")

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

        # TensorBoard
        self.writer = SummaryWriter(LOG_DIR)

        self.global_step = 0

    def train(self):
        for epoch in range(EPOCHS):
            self.model.train()
            total_loss = 0

            progress_bar = tqdm(self.loader, desc=f"Epoch {epoch+1}")

            for batch in progress_bar:
                questions = batch["question"]
                targets = batch["soft_label"].to(self.device)

                # Encode questions
                embeddings = self.encoder.encode(
                    questions,
                    convert_to_tensor=True
                ).to(self.device)

                # Forward pass
                preds = self.model(embeddings)

                # KL Divergence Loss
                loss = self.criterion(
                    torch.log(preds + 1e-8),
                    targets
                )

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (stability)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                self.optimizer.step()

                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())

                # TensorBoard logging (step)
                self.writer.add_scalar(
                    "Loss/train_step",
                    loss.item(),
                    self.global_step
                )
                self.global_step += 1

            avg_loss = total_loss / len(self.loader)

            print(f"[INFO] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

            # TensorBoard logging (epoch)
            self.writer.add_scalar(
                "Loss/train_epoch",
                avg_loss,
                epoch
            )

            # Save checkpoint
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"router_epoch_{epoch+1}.pt"
            )
            torch.save(self.model.state_dict(), checkpoint_path)

            print(f"[INFO] Saved checkpoint: {checkpoint_path}")

        # Save final model
        final_model_path = os.path.join(CHECKPOINT_DIR, "router_final.pt")
        torch.save(self.model.state_dict(), final_model_path)

        print(f"[DONE] Final model saved at: {final_model_path}")

        self.writer.close()


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    set_seed(SEED)

    trainer = RouterTrainer()
    trainer.train()
