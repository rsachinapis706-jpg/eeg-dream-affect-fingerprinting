"""
DreamFM: a channel-agnostic self-supervised EEG encoder.

Why this architecture
---------------------
Pretraining happens on Sleep-EDF (2 EEG channels); the target task is DEED
(6 channels). A model with a fixed channel dimension cannot transfer between them.
So the encoder is applied INDEPENDENTLY PER CHANNEL and the channel dimension is
pooled afterwards. The same weights therefore work for any montage, and
cross-dataset transfer is architecturally honest rather than a reshape trick.

    (B, C, T)
      -> fold channels into batch      (B*C, 1, T)
      -> CNN stem: local waveform morphology, downsamples T -> T/100 tokens
      -> Transformer: long-range temporal structure within the window
      -> mean-pool over time           (B*C, D)
      -> unfold and pool over channels (B, D)
      -> head

Two heads share the encoder:
  * MaskedReconstructionHead - self-supervised pretraining. Random spans of the
    token sequence are replaced with a learned mask embedding and the model
    predicts the original CNN features. This is masked EEG modelling, the audio /
    EEG analogue of masked language modelling.
  * ClassificationHead - supervised fine-tuning (sleep stage or dream affect).

Nothing here is dream-specific. That is deliberate: the same encoder is trained
and evaluated on sleep staging, where signal is known to exist, so that a null
result on dream affect cannot be blamed on the architecture.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNStem(nn.Module):
    """Per-channel 1-D convolutional encoder: (N, 1, T) -> (N, T', D)."""

    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        # 4000 samples @200Hz -> 40 tokens, each summarising 0.5 s
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=51, stride=5, padding=25),   # 4000 -> 800
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=25, stride=5, padding=12),  # 800 -> 160
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4),   # 160 -> 80
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),

            nn.Conv1d(128, d_model, kernel_size=5, stride=2, padding=2),  # 80 -> 40
            nn.BatchNorm1d(d_model), nn.GELU(),
        )

    def forward(self, x):              # (N, 1, T)
        return self.net(x).transpose(1, 2)      # (N, T', D)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DreamFMEncoder(nn.Module):
    def __init__(self, d_model=128, n_heads=4, n_layers=4, dropout=0.1,
                 channel_pool="attention"):
        super().__init__()
        self.d_model = d_model
        self.stem = CNNStem(d_model, dropout)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

        self.channel_pool = channel_pool
        if channel_pool == "attention":
            # learned scalar score per channel -> softmax weights.
            # Works for any channel count, which is what makes transfer possible.
            self.chan_score = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.GELU(),
                nn.Linear(d_model // 2, 1))

    def encode_tokens(self, x, mask=None):
        """x: (B, C, T) -> tokens (B*C, T', D). mask: (B*C, T') bool, True = masked."""
        B, C, T = x.shape
        h = self.stem(x.reshape(B * C, 1, T))          # (B*C, T', D)
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), self.mask_token.to(h.dtype), h)
        return self.transformer(self.pos(h))

    def forward(self, x, return_channel_weights=False):
        """x: (B, C, T) -> (B, D)"""
        B, C, T = x.shape
        h = self.encode_tokens(x)                      # (B*C, T', D)
        h = h.mean(dim=1).reshape(B, C, self.d_model)  # (B, C, D)

        if self.channel_pool == "attention":
            w = torch.softmax(self.chan_score(h).squeeze(-1), dim=1)  # (B, C)
            out = (h * w.unsqueeze(-1)).sum(dim=1)
            if return_channel_weights:
                return out, w
            return out
        out = h.mean(dim=1)
        if return_channel_weights:
            return out, torch.full((B, C), 1.0 / C, device=x.device)
        return out


class MaskedReconstructionHead(nn.Module):
    """Predicts the un-masked CNN features from the transformer output."""

    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model))

    def forward(self, h):
        return self.net(h)


class ClassificationHead(nn.Module):
    def __init__(self, d_model=128, n_classes=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, n_classes))

    def forward(self, z):
        return self.net(z)


class DreamFM(nn.Module):
    """Encoder plus whichever head the current stage needs."""

    def __init__(self, d_model=128, n_heads=4, n_layers=4, n_classes=3,
                 dropout=0.1, channel_pool="attention"):
        super().__init__()
        self.encoder = DreamFMEncoder(d_model, n_heads, n_layers, dropout,
                                      channel_pool)
        self.recon_head = MaskedReconstructionHead(d_model)
        self.cls_head = ClassificationHead(d_model, n_classes, dropout=0.3)

    # ---------- self-supervised ----------
    def ssl_loss(self, x, mask_ratio=0.5, min_span=2, max_span=6):
        """Masked EEG modelling. Spans, not isolated tokens, so the task cannot be
        solved by trivial interpolation between neighbours."""
        B, C, T = x.shape
        with torch.no_grad():
            target = self.encoder.stem(x.reshape(B * C, 1, T))   # (N, T', D)
        N, Tp, D = target.shape

        mask = torch.zeros(N, Tp, dtype=torch.bool, device=x.device)
        n_mask = int(Tp * mask_ratio)
        for i in range(N):
            filled = 0
            while filled < n_mask:
                span = int(torch.randint(min_span, max_span + 1, (1,)).item())
                start = int(torch.randint(0, max(1, Tp - span), (1,)).item())
                mask[i, start:start + span] = True
                filled = int(mask[i].sum().item())

        h = self.encoder.encode_tokens(x, mask=mask)
        pred = self.recon_head(h)
        if mask.sum() == 0:
            return pred.sum() * 0.0
        # normalise targets per token so the loss is not dominated by high-variance
        # channels; standard practice in masked prediction objectives
        t = (target - target.mean(-1, keepdim=True)) / (target.std(-1, keepdim=True) + 1e-6)
        return F.smooth_l1_loss(pred[mask], t[mask])

    # ---------- supervised ----------
    def forward(self, x):
        return self.cls_head(self.encoder(x))

    def embed(self, x):
        return self.encoder(x)

    # ---------- utilities ----------
    def freeze_encoder(self, freeze=True):
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    m = DreamFM(n_classes=3)
    print(f"parameters: {m.n_params():,}")

    # DEED shape
    xd = torch.randn(4, 6, 4000)
    # Sleep-EDF shape -- same weights, different channel count
    xs = torch.randn(4, 2, 4000)

    print("DEED      logits:", tuple(m(xd).shape), "| embed:", tuple(m.embed(xd).shape))
    print("Sleep-EDF logits:", tuple(m(xs).shape), "| embed:", tuple(m.embed(xs).shape))
    print("SSL loss (6ch):", float(m.ssl_loss(xd)))
    print("SSL loss (2ch):", float(m.ssl_loss(xs)))
    _, w = m.encoder(xd, return_channel_weights=True)
    print("channel weights:", w[0].detach().numpy().round(3),
          "sum:", float(w[0].sum()))
