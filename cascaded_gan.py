# Code was modified/commented by Copilot

"""
Cascaded Conditional WGAN-GP for Productivity-Oriented EEG Synthesis
with Research-Standard Evaluation and Neurofeedback Constraints

NOTE: designed for Google Colab environment
Currently meant for extremely quick training for development speed/iteration
"""

# Mount drive
from google.colab import drive
drive.mount("/content/drive")

# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import accuracy_score

import mne
from torcheeg.models import EEGNet

# ============================================================
# CONFIGURATION
# ============================================================

HIGH_END_SETUP = True
FULL_EVALUATION = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {DEVICE}")

SAMPLING_RATE = 128
WINDOW_SECONDS = 1.5
TIME_POINTS = int(SAMPLING_RATE * WINDOW_SECONDS)

if HIGH_END_SETUP:
    EEG_CHANNELS = 9   # actual EEG channels in your dataset (columns 1-9)
    HIDDEN_DIM = 256
    GAN_NOISE_DIM = 64
    BATCH_SIZE = 64
    GAN_EPOCHS = 50
    DECODER_EPOCHS = 30
else:
    EEG_CHANNELS = 9   # same adjustment for low-end
    HIDDEN_DIM = 64
    GAN_NOISE_DIM = 16
    BATCH_SIZE = 16
    GAN_EPOCHS = 10
    DECODER_EPOCHS = 10

N_CLASSES = 2  # low vs high workload

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# WGAN-GP UTILITIES (CONDITIONAL)
# ============================================================

def gradient_penalty(critic, real, fake, labels):
    alpha = torch.rand(real.size(0), 1, 1, device=DEVICE)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    scores = critic(interpolated, labels)
    grads = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]

    grads = grads.view(grads.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

# ============================================================
# CASCADED CONDITIONAL WGAN-GP FOR EEG SYNTHESIS
# ============================================================

class EEGRefinementGenerator(nn.Module):
    """
    Residual EEG refinement generator
    """
    def __init__(self, in_channels):
        super().__init__()
        self.label_emb = nn.Embedding(N_CLASSES, 16)

        self.net = nn.Sequential(
            nn.Conv1d(in_channels + 16, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, in_channels, 3, padding=1)
        )

    def forward(self, x, y):
        y_emb = self.label_emb(y).unsqueeze(-1).repeat(1, 1, x.size(2))
        inp = torch.cat([x, y_emb], dim=1)
        return x + self.net(inp)  # residual refinement


class EEGCritic(nn.Module):
    def __init__(self):
        super().__init__()

        FEATURE_DIM = 128  # single source of truth

        self.label_emb = nn.Embedding(N_CLASSES, FEATURE_DIM)

        self.conv = nn.Sequential(
            nn.Conv1d(EEG_CHANNELS, 64, 5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, FEATURE_DIM, 5, padding=2),
            nn.LeakyReLU(0.2)
        )

        self.fc = nn.Linear(FEATURE_DIM, 1)

    def forward(self, x, y):
        h = self.conv(x).mean(dim=2)
        out = self.fc(h).squeeze(1)
        y_emb = self.label_emb(y)

        # quick assertion test for dimensions
        assert h.shape[1] == self.label_emb.embedding_dim, (
            f"Critic feature dim {h.shape[1]} != label emb dim {self.label_emb.embedding_dim}"
        )

        return out + (h * y_emb).sum(dim=1)

def psd_loss(real, fake):
    real_fft = torch.fft.rfft(real, dim=-1)
    fake_fft = torch.fft.rfft(fake, dim=-1)

    real_psd = torch.mean(torch.abs(real_fft) ** 2, dim=1)
    fake_psd = torch.mean(torch.abs(fake_fft) ** 2, dim=1)

    return torch.mean((real_psd - fake_psd) ** 2)

def autocorr_loss(real, fake, max_lag=32):
    def autocorr(x):
        x = x - x.mean(dim=-1, keepdim=True)
        result = []
        for lag in range(1, max_lag + 1):
            result.append(
                (x[..., :-lag] * x[..., lag:]).mean(dim=-1)
            )
        return torch.stack(result, dim=-1)

    real_ac = autocorr(real)
    fake_ac = autocorr(fake)

    return torch.mean((real_ac - fake_ac) ** 2)

# ============================================================
# CASCADED GAN PIPELINE
# ============================================================

class CascadedWGAN:
    def __init__(self):
        self.stages = []
        self.critics = []
        self.opt_G = []
        self.opt_C = []

        for _ in range(4):
            G = EEGRefinementGenerator(EEG_CHANNELS).to(DEVICE)
            C = EEGCritic().to(DEVICE)

            self.stages.append(G)
            self.critics.append(C)

            self.opt_G.append(
                optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
            )
            self.opt_C.append(
                optim.Adam(C.parameters(), lr=1e-4, betas=(0.5, 0.9))
            )

    def train(self):
        for stage_idx in range(4):
            print(f"--- Training GAN Stage {stage_idx+1}/4 ---")

            G = self.stages[stage_idx]
            C = self.critics[stage_idx]

            # Freeze previous stages
            for prev in self.stages[:stage_idx]:
                for p in prev.parameters():
                    p.requires_grad = False


            for epoch in range(GAN_EPOCHS):
                for _ in range(5):
                    labels = torch.randint(0, N_CLASSES, (BATCH_SIZE,), device=DEVICE)

                    # Sample real EEG from dataset
                    real_idx = torch.randint(0, real_eeg.size(0), (BATCH_SIZE,))
                    real = real_eeg[real_idx].to(DEVICE)

                    if stage_idx == 0:
                        fake_input = real + 0.1 * torch.randn_like(real)
                    else:
                        # use output of previous stage as input
                        with torch.no_grad():
                            fake_input = real.clone()
                            for prev in self.stages[:stage_idx]:
                                fake_input = prev(fake_input, labels)

                    fake = G(fake_input, labels)

                    self.opt_C[stage_idx].zero_grad()
                    loss_C = (
                        C(fake.detach(), labels).mean()
                        - C(real, labels).mean()
                        + 10 * gradient_penalty(C, real, fake.detach(), labels)
                    )
                    loss_C.backward()
                    self.opt_C[stage_idx].step()

                # Generator update
                self.opt_G[stage_idx].zero_grad()
                fake = G(fake_input, labels)

                lambda_psd = 0.1
                lambda_ac = 0.05

                loss_G = (
                    -C(fake, labels).mean()
                    + lambda_psd * psd_loss(real, fake)
                    + lambda_ac * autocorr_loss(real, fake)
                )

                loss_G.backward()
                self.opt_G[stage_idx].step()

    def generate(self, n):
        x = torch.randn(n, EEG_CHANNELS, TIME_POINTS, device=DEVICE)
        labels = torch.randint(0, N_CLASSES, (n,), device=DEVICE)

        for G in self.stages:
            x = G(x, labels)

        return x, labels

# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_mmd(x, y, sigma=10.0):
    def gaussian_kernel(a, b):
        dist = torch.cdist(a, b, p=2) ** 2
        return torch.exp(-dist / (2 * sigma ** 2))

    K_xx = gaussian_kernel(x, x)
    K_yy = gaussian_kernel(y, y)
    K_xy = gaussian_kernel(x, y)

    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

def plot_psd(real, fake, title):
    real = real.cpu().detach().numpy()
    fake = fake.cpu().detach().numpy()

    freqs = np.fft.rfftfreq(real.shape[-1], d=1 / SAMPLING_RATE)
    real_psd = np.mean(np.abs(np.fft.rfft(real, axis=-1)) ** 2, axis=(0, 1))
    fake_psd = np.mean(np.abs(np.fft.rfft(fake, axis=-1)) ** 2, axis=(0, 1))

    plt.figure()
    plt.plot(freqs, real_psd, label="Real")
    plt.plot(freqs, fake_psd, label="Synthetic")
    plt.legend()
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{title}.png"))
    plt.show()

# ============================================================
# DECODER TRAINING
# ============================================================

def train_decoder(loader, epochs=DECODER_EPOCHS):
    model = EEGNet(
        chunk_size=TIME_POINTS,      # number of time points per segment
        num_electrodes=EEG_CHANNELS, # number of EEG channels
        num_classes=N_CLASSES        # target class count
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Freeze feature extractor for synthetic-only training
    for name, param in model.named_parameters():
        if "conv" in name:
            param.requires_grad = False

    model.train()

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.unsqueeze(1)  # shape [B, 1, C, T]
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
    return model


def evaluate_decoder(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.unsqueeze(1)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total * 100

# ============================================================
# LOAD REAL DATA (ARITHMETIC + STROOP WORKLOAD EEG)
# ============================================================

DATASET_ROOT = "/content/drive/MyDrive/eeg_datasets"
SAMPLING_RATE_RAW = 250
TARGET_SAMPLING_RATE = 128
WINDOW_SAMPLES = TIME_POINTS

def workload_label_from_filename(fname):
    fname = fname.lower()
    if "casual" in fname or "lowlevel" in fname or "natural" in fname:
        return 0  # low workload
    if "midlevel" in fname or "highlevel" in fname:
        return 1  # high workload
    raise ValueError(f"Unknown workload level: {fname}")

def load_txt_eeg(path):
    """
    Loads a .txt EEG file.
    Each row = 1 timepoint, columns = [sample_index, EEG1, EEG2, ..., metadata]
    Extract only EEG columns (columns 1-9 in your dataset).
    Output shape: channels x samples
    """
    data = np.genfromtxt(path, delimiter=",", dtype=np.float32)

    # pick only EEG columns (adjust 1:10 to actual EEG columns)
    eeg = data[:, 1:10]  # columns 1-9 = EEG channels

    # transpose to (channels, samples)
    eeg = eeg.T

    # Remove any NaN columns (optional)
    if np.isnan(eeg).any():
        eeg = eeg[~np.isnan(eeg).all(axis=0)]

    return eeg

def resample_eeg(eeg, orig_sr, target_sr):
    n_samples = eeg.shape[1]
    duration = n_samples / orig_sr
    new_n = int(duration * target_sr)

    old_idx = np.linspace(0, duration, n_samples)
    new_idx = np.linspace(0, duration, new_n)

    return np.stack([np.interp(new_idx, old_idx, ch) for ch in eeg])

def segment_eeg(eeg, label):
    segments, labels = [], []
    for start in range(0, eeg.shape[1] - WINDOW_SAMPLES, WINDOW_SAMPLES):
        seg = eeg[:, start:start + WINDOW_SAMPLES]
        if seg.shape[1] == WINDOW_SAMPLES:
            segments.append(seg)
            labels.append(label)
    return segments, labels

def load_full_dataset():
    all_segments = []
    all_labels = []

    file_count = 0
    used_files = 0

    for root, _, files in os.walk(DATASET_ROOT):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue

            file_count += 1
            file_path = os.path.join(root, fname)

            try:
                label = workload_label_from_filename(fname)
            except ValueError:
                print(f"[SKIP] Could not infer label from filename: {fname}")
                continue

            eeg = load_txt_eeg(file_path)

            # Check channel count
            if eeg.shape[0] != EEG_CHANNELS:
                print(f"[SKIP] {fname}: expected {EEG_CHANNELS} channels, got {eeg.shape[0]}")
                continue

            eeg = resample_eeg(eeg, SAMPLING_RATE_RAW, TARGET_SAMPLING_RATE)

            if eeg.shape[1] < WINDOW_SAMPLES:
                print(f"[SKIP] {fname}: too short after resampling")
                continue

            segments, labels = segment_eeg(eeg, label)

            if len(segments) == 0:
                print(f"[SKIP] {fname}: no valid segments")
                continue

            used_files += 1
            all_segments.extend(segments)
            all_labels.extend(labels)

    print(f"Found {file_count} .txt files")
    print(f"Used {used_files} files")
    print(f"Total segments: {len(all_segments)}")

    if len(all_segments) == 0:
        raise RuntimeError(
            "No EEG segments were loaded. "
            "Check DATASET_ROOT, filename patterns, channel count, and window length."
        )

    eeg_array = np.stack(all_segments)
    label_array = np.array(all_labels)

    mean = eeg_array.mean(axis=(0, 2), keepdims=True)
    std = eeg_array.std(axis=(0, 2), keepdims=True) + 1e-6
    eeg_array = (eeg_array - mean) / std

    return (
        torch.tensor(eeg_array, dtype=torch.float32).to(DEVICE),
        torch.tensor(label_array, dtype=torch.long).to(DEVICE)
    )

real_eeg, real_labels = load_full_dataset()
real_dataset = TensorDataset(real_eeg, real_labels)

# Main experimental GAN
gan = CascadedWGAN()
gan.train()

synthetic_eeg, synthetic_labels = gan.generate(1000)

synthetic_eeg += 0.01 * torch.randn_like(synthetic_eeg)

# Normalize synthetic EEG using REAL statistics
synthetic_eeg = (synthetic_eeg - real_eeg.mean(dim=(0,2), keepdim=True)) / (
    real_eeg.std(dim=(0,2), keepdim=True) + 1e-6
)

# DETACH to avoid autograd graph issues
synthetic_dataset = TensorDataset(
    synthetic_eeg.detach().float(),   # <-- detach here
    synthetic_labels
)

real_loader = DataLoader(real_dataset, batch_size=BATCH_SIZE, shuffle=True)
synthetic_loader = DataLoader(synthetic_dataset, batch_size=BATCH_SIZE)
combined_loader = DataLoader(
    ConcatDataset([real_dataset, synthetic_dataset]),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=False
)

# Train EEG decoders
real_model = train_decoder(real_loader)
combined_model = train_decoder(combined_loader)

# Evaluate EEG decoders
real_acc = evaluate_decoder(real_model, real_loader)
synthetic_acc = evaluate_decoder(real_model, synthetic_loader)
combined_acc = evaluate_decoder(combined_model, real_loader)

print(f"EEG Decoder Accuracy:")
print(f"Real data: {real_acc:.2f}%")
print(f"Synthetic data: {synthetic_acc:.2f}%")
print(f"Real+Synthetic: {combined_acc:.2f}%")

# ============================================================
# OPTIONAL FULL EVALUATION
# ============================================================

if FULL_EVALUATION:
    n = min(len(real_eeg), len(synthetic_eeg))
    mmd = compute_mmd(
        real_eeg[:n].view(n, -1),
        synthetic_eeg[:n].view(n, -1)
    )
    pd.DataFrame({"MMD": [mmd.item()]}).to_csv(
        os.path.join(OUTPUT_DIR, "distribution_metrics.csv"),
        index=False
    )

    plot_psd(real_eeg[:n], synthetic_eeg[:n], "PSD_Comparison")
