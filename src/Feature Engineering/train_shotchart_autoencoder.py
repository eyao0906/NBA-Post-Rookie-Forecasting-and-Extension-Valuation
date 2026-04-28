import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/Shot Feature/shot_tensors"
OUT_DIR = PROJECT_ROOT / "Output/ShotChartDetail"

FALLBACK_DATA_DIRS = [
    PROJECT_ROOT / "data/ShotFeature",
    PROJECT_ROOT / "data/ShotFeature/shot_tensors",
    PROJECT_ROOT / "data/Shot Feature",
    PROJECT_ROOT / "data/Shot Feature/shot_tensors",
]

DEFAULT_TENSOR_FILENAME = "player_season_shot_tensors.npz"
DEFAULT_INDEX_FILENAME = "player_shot_tensor_index.csv"


def resolve_default_input(filename: str) -> Path:
    direct = DATA_DIR / filename
    if direct.exists():
        return direct
    nested = DATA_DIR / "shot_tensors" / filename
    if nested.exists():
        return nested
    for base in FALLBACK_DATA_DIRS:
        candidate = base / filename
        if candidate.exists():
            return candidate
        candidate_nested = base / "shot_tensors" / filename
        if candidate_nested.exists():
            return candidate_nested
    return direct


@dataclass
class TrainConfig:
    tensor_path: str
    index_path: str
    out_dir: str
    latent_dim: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    train_fraction: float
    min_shots: int
    early_stopping_patience: int
    random_seed: int
    player_agg_method: str
    loss_weights: List[float]
    device: str
    subset_fraction: float
    num_workers: int
    amp: bool
    grad_clip_norm: float


class ShotTensorDataset(Dataset):
    def __init__(self, encoder_inputs: np.ndarray, raw_targets: np.ndarray, row_ids: np.ndarray):
        self.encoder_inputs = torch.from_numpy(encoder_inputs.astype(np.float32))
        self.raw_targets = torch.from_numpy(raw_targets.astype(np.float32))
        self.row_ids = torch.from_numpy(row_ids.astype(np.int64))

    def __len__(self) -> int:
        return int(self.encoder_inputs.shape[0])

    def __getitem__(self, idx: int):
        return self.encoder_inputs[idx], self.raw_targets[idx], self.row_ids[idx]


class ConstrainedConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder with physics-aware decoder constraints:
    - attempt density is nonnegative and sums to 1 via spatial softmax
    - made density is nonnegative and bounded by attempt density via attempt * sigmoid(prob)
    - smoothed make-rate surface is bounded in [0, 1] via sigmoid
    """

    def __init__(self, input_channels: int = 3, latent_dim: int = 24):
        super().__init__()
        self.input_channels = input_channels
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_fc = nn.Linear(32 * 4 * 4, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 32 * 4 * 4)
        self.decoder_trunk = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=0),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h = h.reshape(h.size(0), -1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.size(0), 32, 4, 4)
        raw = self.decoder_trunk(h)

        attempt_logits = raw[:, 0:1]
        made_prob_logits = raw[:, 1:2]
        rate_logits = raw[:, 2:3]

        b, _, hgt, wid = attempt_logits.shape
        attempt_density = torch.softmax(attempt_logits.reshape(b, -1), dim=1).reshape(b, 1, hgt, wid)
        made_prob = torch.sigmoid(made_prob_logits)
        made_density = attempt_density * made_prob
        make_rate = torch.sigmoid(rate_logits)

        return torch.cat([attempt_density, made_density, make_rate], dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class WeightedMSELoss(nn.Module):
    def __init__(self, channel_weights: List[float]):
        super().__init__()
        weight_tensor = torch.tensor(channel_weights, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("channel_weights", weight_tensor)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq = (prediction - target) ** 2
        return (sq * self.channel_weights).mean()


class ChannelNormalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.clip(std.astype(np.float32), 1e-6, None)

    @classmethod
    def fit(cls, tensors: np.ndarray) -> "ChannelNormalizer":
        mean = tensors.mean(axis=(0, 2, 3), keepdims=True)
        std = tensors.std(axis=(0, 2, 3), keepdims=True)
        return cls(mean=mean, std=std)

    def transform(self, tensors: np.ndarray) -> np.ndarray:
        return ((tensors - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, tensors: np.ndarray) -> np.ndarray:
        return (tensors * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "channel_mean": self.mean.reshape(-1).tolist(),
            "channel_std": self.std.reshape(-1).tolist(),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_device(explicit_device: str) -> str:
    if explicit_device != "auto":
        return explicit_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tensor_inputs(tensor_path: Path, index_path: Path, min_shots: int) -> Tuple[np.ndarray, pd.DataFrame]:
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Tensor index file not found: {index_path}")

    tensor_file = np.load(tensor_path)
    if "tensors" not in tensor_file:
        raise ValueError(f"Expected key 'tensors' in {tensor_path}")
    tensors = tensor_file["tensors"].astype(np.float32)
    index_df = pd.read_csv(index_path)

    if len(index_df) != tensors.shape[0]:
        raise ValueError(f"Tensor/index mismatch: {tensors.shape[0]} tensors vs {len(index_df)} index rows")
    if tensors.ndim != 4 or tensors.shape[1] != 3:
        raise ValueError(f"Expected [N, 3, H, W] tensors, got {tensors.shape}")

    if min_shots > 0 and "shot_attempts" in index_df.columns:
        keep = index_df["shot_attempts"].fillna(0).astype(float) >= float(min_shots)
        index_df = index_df.loc[keep].reset_index(drop=True)
        tensors = tensors[keep.to_numpy()]

    index_df["row_id"] = np.arange(len(index_df), dtype=np.int64)
    return tensors, index_df


def maybe_subsample(tensors: np.ndarray, index_df: pd.DataFrame, subset_fraction: float, seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    if not (0 < subset_fraction <= 1.0):
        raise ValueError("subset-fraction must be in (0, 1].")
    if subset_fraction >= 1.0:
        return tensors, index_df

    rng = np.random.default_rng(seed)
    keep_n = max(32, int(round(len(index_df) * subset_fraction)))
    keep_n = min(keep_n, len(index_df))
    keep_idx = np.sort(rng.choice(len(index_df), size=keep_n, replace=False))
    out_tensors = tensors[keep_idx]
    out_index = index_df.iloc[keep_idx].reset_index(drop=True)
    out_index["row_id"] = np.arange(len(out_index), dtype=np.int64)
    return out_tensors, out_index


def player_level_split(index_df: pd.DataFrame, train_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if "PLAYER_ID" not in index_df.columns:
        raise ValueError("PLAYER_ID is required in tensor index for player-aware split.")
    if not (0 < train_fraction < 1):
        raise ValueError("train-fraction must be between 0 and 1.")

    players = index_df["PLAYER_ID"].dropna().astype(int).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(players)

    n_train_players = int(round(len(players) * train_fraction))
    n_train_players = min(max(n_train_players, 1), max(len(players) - 1, 1))
    train_players = set(players[:n_train_players].tolist())

    train_mask = index_df["PLAYER_ID"].astype(int).isin(train_players).to_numpy()
    val_mask = ~train_mask

    if val_mask.sum() == 0:
        order = np.arange(len(index_df))
        rng.shuffle(order)
        n_train_rows = min(max(int(round(len(order) * train_fraction)), 1), max(len(order) - 1, 1))
        train_mask = np.zeros(len(order), dtype=bool)
        train_mask[order[:n_train_rows]] = True
        val_mask = ~train_mask

    return train_mask, val_mask


def _maybe_to_device(x: torch.Tensor, device: str) -> torch.Tensor:
    return x.to(device, non_blocking=device.startswith("cuda"))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    for encoder_x, raw_target, _ in loader:
        encoder_x = _maybe_to_device(encoder_x, device)
        raw_target = _maybe_to_device(raw_target, device)
        recon, _ = model(encoder_x)
        loss = criterion(recon, raw_target)
        total_loss += float(loss.item()) * raw_target.size(0)
        total_rows += raw_target.size(0)
    return total_loss / max(total_rows, 1)


def train_autoencoder(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str, epochs: int, patience: int, amp: bool, grad_clip_norm: float) -> Tuple[nn.Module, List[Dict[str, float]]]:
    history: List[Dict[str, float]] = []
    best_state = None
    best_val = math.inf
    best_epoch = 0
    patience_counter = 0

    use_amp = amp and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for encoder_x, raw_target, _ in train_loader:
            encoder_x = _maybe_to_device(encoder_x, device)
            raw_target = _maybe_to_device(raw_target, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                recon, _ = model(encoder_x)
                loss = criterion(recon, raw_target)

            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * raw_target.size(0)
            total_rows += raw_target.size(0)

        train_loss = total_loss / max(total_rows, 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | best_val={best_val:.6f}", flush=True)
        if patience_counter >= patience:
            print(f"Early stopping triggered after epoch {epoch}. Best epoch: {best_epoch}.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def encode_dataset(model: ConstrainedConvAutoencoder, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_rows, all_embeddings = [], []
    for encoder_x, _, row_id in loader:
        encoder_x = _maybe_to_device(encoder_x, device)
        z = model.encode(encoder_x).detach().cpu().numpy()
        all_embeddings.append(z)
        all_rows.append(row_id.detach().cpu().numpy())
    return np.concatenate(all_rows), np.concatenate(all_embeddings)


def compute_rowwise_metrics(prediction: np.ndarray, target: np.ndarray, row_ids: np.ndarray, weights: List[float]) -> pd.DataFrame:
    rows = []
    weight_arr = np.array(weights, dtype=np.float32).reshape(-1, 1, 1)
    for i in range(prediction.shape[0]):
        pred_i = prediction[i]
        target_i = target[i]
        sq = (pred_i - target_i) ** 2
        out = {"row_id": int(row_ids[i]), "weighted_mse_total": float((sq * weight_arr).mean())}
        for c, name in enumerate(["attempt", "made", "rate"]):
            pred_flat = pred_i[c].reshape(-1)
            targ_flat = target_i[c].reshape(-1)
            out[f"{name}_mse"] = float(np.mean((pred_flat - targ_flat) ** 2))
            out[f"{name}_mass_in"] = float(targ_flat.sum())
            out[f"{name}_mass_recon"] = float(pred_flat.sum())
            if np.std(pred_flat) > 1e-8 and np.std(targ_flat) > 1e-8:
                out[f"{name}_corr"] = float(np.corrcoef(pred_flat, targ_flat)[0, 1])
            else:
                out[f"{name}_corr"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_validation_diagnostics(model: ConstrainedConvAutoencoder, encoder_inputs: np.ndarray, raw_targets: np.ndarray, row_ids: np.ndarray, device: str, loss_weights: List[float]) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    model.eval()
    x = torch.from_numpy(encoder_inputs.astype(np.float32))
    all_pred = []
    batch_size = 256
    for start in range(0, len(x), batch_size):
        xb = _maybe_to_device(x[start:start + batch_size], device)
        recon, _ = model(xb)
        all_pred.append(recon.detach().cpu().numpy())
    pred = np.concatenate(all_pred, axis=0)
    metrics_df = compute_rowwise_metrics(prediction=pred, target=raw_targets, row_ids=row_ids, weights=loss_weights)

    ordered = metrics_df.sort_values("weighted_mse_total").reset_index(drop=True)
    if len(ordered) >= 3:
        chosen_row_ids = [int(ordered.iloc[0]["row_id"]), int(ordered.iloc[len(ordered) // 2]["row_id"]), int(ordered.iloc[-1]["row_id"])]
        chosen_labels = ["best", "median", "worst"]
    else:
        chosen_row_ids = ordered["row_id"].astype(int).tolist()
        chosen_labels = [f"example_{i + 1}" for i in range(len(chosen_row_ids))]

    lookup = {rid: pos for pos, rid in enumerate(row_ids.tolist())}
    chosen_pos = [lookup[rid] for rid in chosen_row_ids]
    example_dict = {
        "input_original_scale": raw_targets[chosen_pos].astype(np.float32),
        "recon_original_scale": pred[chosen_pos].astype(np.float32),
        "recon_diff": (pred[chosen_pos] - raw_targets[chosen_pos]).astype(np.float32),
        "sample_rows": np.array(chosen_row_ids, dtype=np.int64),
        "sample_labels": np.array(chosen_labels),
    }
    return example_dict, ordered


def save_training_curve(history_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted reconstruction MSE")
    plt.title("ShotChart CNN Autoencoder Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def _maybe_smooth_for_display(arr2d: np.ndarray, sigma: float = 0.55) -> np.ndarray:
    if gaussian_filter is None:
        return arr2d
    return gaussian_filter(arr2d, sigma=sigma, mode="nearest")


def save_channel_recon_plot(example_dict: Dict[str, np.ndarray], metrics_lookup: pd.DataFrame, out_path: Path) -> None:
    input_arr = example_dict["input_original_scale"]
    recon_arr = example_dict["recon_original_scale"]
    diff_arr = example_dict["recon_diff"]
    sample_rows = example_dict["sample_rows"].astype(int).tolist()
    sample_labels = example_dict["sample_labels"].tolist()

    n_show = min(3, input_arr.shape[0])
    if n_show == 0:
        return

    fig, axes = plt.subplots(n_show, 9, figsize=(22, 4.2 * n_show))
    if n_show == 1:
        axes = np.expand_dims(axes, axis=0)

    channel_meta = [("Attempt density", "attempt"), ("Made density", "made"), ("Smoothed make rate", "rate")]
    metrics_lookup = metrics_lookup.set_index("row_id", drop=False)

    for i in range(n_show):
        row_id = sample_rows[i]
        example_label = sample_labels[i]
        metric_row = metrics_lookup.loc[row_id]

        for c, (display_name, prefix) in enumerate(channel_meta):
            inp = input_arr[i, c]
            rec = recon_arr[i, c]
            dif = diff_arr[i, c]

            if c < 2:
                show_inp = _maybe_smooth_for_display(inp)
                show_rec = _maybe_smooth_for_display(rec)
                show_dif = _maybe_smooth_for_display(dif)
            else:
                show_inp = inp
                show_rec = rec
                show_dif = dif

            vmax = float(max(np.max(show_inp), np.max(show_rec), 1e-6))
            vmin = float(min(np.min(show_inp), np.min(show_rec), 0.0))
            diff_lim = float(max(np.max(np.abs(show_dif)), 1e-6))

            ax_in = axes[i, 3 * c]
            ax_rec = axes[i, 3 * c + 1]
            ax_diff = axes[i, 3 * c + 2]

            im1 = ax_in.imshow(show_inp, origin="lower", interpolation="bilinear", aspect="auto", vmin=vmin, vmax=vmax)
            im2 = ax_rec.imshow(show_rec, origin="lower", interpolation="bilinear", aspect="auto", vmin=vmin, vmax=vmax)
            im3 = ax_diff.imshow(show_dif, origin="lower", interpolation="bilinear", aspect="auto", cmap="coolwarm", vmin=-diff_lim, vmax=diff_lim)

            corr_val = metric_row[f"{prefix}_corr"]
            mse_val = metric_row[f"{prefix}_mse"]
            ax_in.set_title(f"{example_label.capitalize()} | Input: {display_name}", fontsize=11)
            ax_rec.set_title(f"Recon: {display_name}\nMSE={mse_val:.4f} | Corr={corr_val:.2f}" if pd.notna(corr_val) else f"Recon: {display_name}\nMSE={mse_val:.4f}", fontsize=11)
            ax_diff.set_title("Recon - Input", fontsize=11)

            for ax in (ax_in, ax_rec, ax_diff):
                ax.axis("off")

            fig.colorbar(im1, ax=ax_in, fraction=0.046, pad=0.03)
            fig.colorbar(im2, ax=ax_rec, fraction=0.046, pad=0.03)
            fig.colorbar(im3, ax=ax_diff, fraction=0.046, pad=0.03)

    plt.suptitle("ShotChart Autoencoder Reconstruction Diagnostics", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def build_player_embeddings(season_embedding_df: pd.DataFrame, embedding_cols: List[str], method: str) -> pd.DataFrame:
    df = season_embedding_df.copy()
    if method == "simple_mean":
        df["_weight"] = 1.0
    elif method == "shot_weighted_mean":
        df["_weight"] = df["shot_attempts"].fillna(1.0).clip(lower=1.0).astype(float)
    elif method == "season4_weighted_mean":
        base = df["shot_attempts"].fillna(1.0).clip(lower=1.0).astype(float)
        boost = np.where(df["season_num"].fillna(0).astype(float) == 4.0, 1.5, 1.0)
        df["_weight"] = base * boost
    else:
        raise ValueError(f"Unsupported player aggregation method: {method}")

    group_cols = [c for c in ["PLAYER_ID", "PLAYER_NAME", "DRAFT_YEAR"] if c in df.columns]

    def agg_one(g: pd.DataFrame) -> pd.Series:
        w = g["_weight"].to_numpy(dtype=float)
        out = {
            "seasons_with_embeddings": int(g["season_num"].nunique()),
            "total_shot_attempts_covered": float(g["shot_attempts"].sum()),
            "avg_shot_attempts_per_embedded_season": float(g["shot_attempts"].mean()),
        }
        for col in embedding_cols:
            out[col] = float(np.average(g[col].to_numpy(dtype=float), weights=w))
        return pd.Series(out)

    player_df = (df.groupby(group_cols, dropna=False).apply(agg_one, include_groups=False).reset_index().sort_values(group_cols).reset_index(drop=True))
    player_df["player_embedding_aggregation"] = method
    return player_df


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, device: str, num_workers: int) -> DataLoader:
    pin_memory = device.startswith("cuda")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a constrained CNN autoencoder on player-season shot tensors and export shot embeddings.")
    parser.add_argument("--tensor-path", type=Path, default=resolve_default_input(DEFAULT_TENSOR_FILENAME))
    parser.add_argument("--index-path", type=Path, default=resolve_default_input(DEFAULT_INDEX_FILENAME))
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument("--min-shots", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--player-agg-method", type=str, default="shot_weighted_mean", choices=["simple_mean", "shot_weighted_mean", "season4_weighted_mean"])
    parser.add_argument("--loss-weights", type=float, nargs=3, default=[3.0, 3.0, 1.0])
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--subset-fraction", type=float, default=1.0, help="Optional development shortcut in (0, 1].")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision when running on CUDA.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.random_seed)
    device = infer_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    tensors, index_df = load_tensor_inputs(args.tensor_path, args.index_path, args.min_shots)
    tensors, index_df = maybe_subsample(tensors, index_df, args.subset_fraction, args.random_seed)
    train_mask, val_mask = player_level_split(index_df, args.train_fraction, args.random_seed)

    train_raw = tensors[train_mask]
    val_raw = tensors[val_mask]
    if len(train_raw) == 0 or len(val_raw) == 0:
        raise RuntimeError("Train/validation split produced an empty partition.")

    normalizer = ChannelNormalizer.fit(train_raw)
    train_encoder_input = normalizer.transform(train_raw)
    val_encoder_input = normalizer.transform(val_raw)
    all_encoder_input = normalizer.transform(tensors)

    train_dataset = ShotTensorDataset(train_encoder_input, train_raw, index_df.loc[train_mask, "row_id"].to_numpy())
    val_dataset = ShotTensorDataset(val_encoder_input, val_raw, index_df.loc[val_mask, "row_id"].to_numpy())
    full_dataset = ShotTensorDataset(all_encoder_input, tensors, index_df["row_id"].to_numpy())

    train_loader = build_dataloader(train_dataset, args.batch_size, True, device, args.num_workers)
    val_loader = build_dataloader(val_dataset, args.batch_size, False, device, args.num_workers)
    full_loader = build_dataloader(full_dataset, args.batch_size, False, device, args.num_workers)

    model = ConstrainedConvAutoencoder(input_channels=3, latent_dim=args.latent_dim).to(device)
    criterion = WeightedMSELoss(channel_weights=args.loss_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model, history = train_autoencoder(model, train_loader, val_loader, criterion, optimizer, device, args.epochs, args.early_stopping_patience, args.amp, args.grad_clip_norm)

    history_df = pd.DataFrame(history)
    history_df.to_csv(args.out_dir / "training_history.csv", index=False)
    save_training_curve(history_df, args.out_dir / "training_curve.png")

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_class": "ConstrainedConvAutoencoder",
        "latent_dim": args.latent_dim,
        "input_channels": 3,
        "decoder_constraints": {
            "attempt_density": "spatial_softmax_sum_to_1",
            "made_density": "attempt_density_times_sigmoid_prob",
            "smoothed_make_rate": "sigmoid_0_1",
        },
        "normalizer": normalizer.to_dict(),
        "loss_weights": [float(w) for w in args.loss_weights],
        "tensor_path": str(args.tensor_path),
        "index_path": str(args.index_path),
    }, args.out_dir / "shot_autoencoder_model.pt")

    row_ids, embeddings = encode_dataset(model, full_loader, device)
    embedding_cols = [f"shot_emb_{i + 1}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=embedding_cols)
    emb_df["row_id"] = row_ids

    season_embedding_df = index_df.merge(emb_df, on="row_id", how="left", validate="one_to_one")
    season_embedding_df["split"] = np.where(train_mask, "train", "validation")
    season_keep_cols = [c for c in ["PLAYER_ID", "PLAYER_NAME", "season_num", "season_string", "DRAFT_YEAR", "games_with_shots", "shot_attempts", "shot_makes", "fg_pct", "split"] if c in season_embedding_df.columns] + embedding_cols
    season_embedding_df = season_embedding_df[season_keep_cols].copy().sort_values(["PLAYER_ID", "season_num"]).reset_index(drop=True)
    season_embedding_df.to_csv(args.out_dir / "player_season_shot_embedding.csv", index=False)

    player_embedding_df = build_player_embeddings(season_embedding_df, embedding_cols, args.player_agg_method)
    player_embedding_df.to_csv(args.out_dir / "player_shot_embedding.csv", index=False)

    val_row_ids = index_df.loc[val_mask, "row_id"].to_numpy(dtype=np.int64)
    example_dict, diagnostics_df = collect_validation_diagnostics(model, val_encoder_input, val_raw, val_row_ids, device, args.loss_weights)
    np.savez_compressed(args.out_dir / "reconstruction_examples.npz", **example_dict)
    diagnostics_df.to_csv(args.out_dir / "reconstruction_diagnostics.csv", index=False)
    save_channel_recon_plot(example_dict, diagnostics_df, args.out_dir / "reconstruction_examples.png")

    split_summary = {
        "train_player_seasons": int(train_mask.sum()),
        "validation_player_seasons": int(val_mask.sum()),
        "train_unique_players": int(index_df.loc[train_mask, "PLAYER_ID"].nunique()),
        "validation_unique_players": int(index_df.loc[val_mask, "PLAYER_ID"].nunique()),
    }
    best_epoch = int(history_df.loc[history_df["val_loss"].idxmin(), "epoch"])
    metadata = {
        "config": asdict(TrainConfig(
            tensor_path=str(args.tensor_path),
            index_path=str(args.index_path),
            out_dir=str(args.out_dir),
            latent_dim=int(args.latent_dim),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            train_fraction=float(args.train_fraction),
            min_shots=int(args.min_shots),
            early_stopping_patience=int(args.early_stopping_patience),
            random_seed=int(args.random_seed),
            player_agg_method=str(args.player_agg_method),
            loss_weights=[float(w) for w in args.loss_weights],
            device=str(device),
            subset_fraction=float(args.subset_fraction),
            num_workers=int(args.num_workers),
            amp=bool(args.amp),
            grad_clip_norm=float(args.grad_clip_norm),
        )),
        "normalizer": normalizer.to_dict(),
        "split_summary": split_summary,
        "best_epoch": best_epoch,
        "best_val_loss": float(history_df["val_loss"].min()),
        "final_train_loss": float(history_df.iloc[-1]["train_loss"]),
        "final_val_loss": float(history_df.iloc[-1]["val_loss"]),
        "n_player_season_embeddings": int(len(season_embedding_df)),
        "n_player_embeddings": int(len(player_embedding_df)),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "decoder_constraints": {
            "attempt_density": "spatial_softmax_sum_to_1",
            "made_density": "attempt_density_times_sigmoid_prob",
            "smoothed_make_rate": "sigmoid_0_1",
        },
    }
    (args.out_dir / "shot_autoencoder_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Training complete.", flush=True)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved outputs to: {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
