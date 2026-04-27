import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """
    Router model for Mixture of Granularity (MoG)

    Input:
        embedding vector of question (e.g., from SentenceTransformer)

    Output:
        probability distribution over granularity levels
        e.g. [half, 1, 2, 4, 8]
    """

    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=5, dropout=0.1):
        super(Router, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch_size, input_dim]
        returns: [batch_size, output_dim] (probabilities)
        """

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        # softmax → probability distribution
        probs = F.softmax(x, dim=-1)

        return probs