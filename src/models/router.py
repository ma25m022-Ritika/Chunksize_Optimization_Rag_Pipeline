import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

LEVELS = ["half", "1", "2", "4", "8"]

class Router(nn.Module):
    def __init__(self, input_dim=1024, output_dim=5, dropout=0.2):
        super(Router, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.LogSoftmax(dim=-1),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._encoder = None

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = SentenceTransformer("stsb-roberta-large", device=self.device)
        return self._encoder

    def forward(self, x):
        return self.network(x)

    def run(self, question: str):
        embedding = self.encoder.encode(question, convert_to_tensor=True).to(self.device)
        with torch.no_grad():
            log_probs = self.forward(embedding.unsqueeze(0))
            probs = torch.exp(log_probs).cpu().numpy()[0]
        return dict(zip(LEVELS, probs))