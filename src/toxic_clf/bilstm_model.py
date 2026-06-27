"""BiLSTM classifier for multilabel toxic comment classification (Phase 7).

Single bidirectional LSTM layer with mean+max pooling over timesteps,
followed by a linear head. Deliberately simple: no attention, no
stacked layers -- the dataset doesn't need it, and the project's
philosophy is one solid model per family, not a sprawl of variants.
"""

import torch
import torch.nn as nn

NUM_LABELS = 6


class BiLSTMClassifier(nn.Module):
    """Embedding -> BiLSTM -> mean+max pooling -> dropout -> linear.

    Outputs raw logits (no sigmoid applied) -- pair with
    nn.BCEWithLogitsLoss(pos_weight=...) during training, and torch.sigmoid()
    at inference/evaluation time to get probabilities for compute_metrics().
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        padding_idx: int = 0,
        num_labels: int = NUM_LABELS,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # hidden_dim * 2 (bidirectional) * 2 (mean + max pooling concatenated)
        self.classifier = nn.Linear(hidden_dim * 2 * 2, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Args:
            input_ids: (batch_size, seq_len) tensor of token indices.

        Returns:
            (batch_size, num_labels) tensor of raw logits.
        """
        embedded = self.embedding(input_ids)  # (B, L, E)
        lstm_out, _ = self.lstm(embedded)  # (B, L, 2*H)

        mean_pooled = lstm_out.mean(dim=1)  # (B, 2*H)
        max_pooled, _ = lstm_out.max(dim=1)  # (B, 2*H)
        pooled = torch.cat([mean_pooled, max_pooled], dim=1)  # (B, 4*H)

        pooled = self.dropout(pooled)
        return self.classifier(pooled)  # (B, num_labels) raw logits
