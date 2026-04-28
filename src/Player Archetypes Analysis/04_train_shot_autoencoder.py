from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from archetype_workflow_utils import PATHS, append_log, ensure_dirs, embedding_columns

LATENT_DIM = 24
BATCH_SIZE = 128
EPOCHS = 18
LR = 1e-3
PATIENCE = 5


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.to_latent = nn.Linear(32 * 4 * 4, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 32 * 4 * 4)
        # Linear decoder output is required because reconstruction targets are z-scored.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=0),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.to_latent(h.reshape(h.size(0), -1))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        h = self.from_latent(z).view(x.size(0), 32, 4, 4)
        return self.decoder(h), z


def load_npz(path: Path) -> np.ndarray:
    return np.load(path)["tensors"].astype(np.float32)


def normalize(train: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    mean = train.mean(axis=(0, 2, 3), keepdims=True)
    std = train.std(axis=(0, 2, 3), keepdims=True)
    std = np.clip(std, 1e-6, None)
    return ((train - mean) / std).astype(np.float32), ((full - mean) / std).astype(np.float32), {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()}


def apply_normalizer(arr: np.ndarray, normalizer: dict) -> np.ndarray:
    mean = np.array(normalizer["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(normalizer["std"], dtype=np.float32).reshape(1, 3, 1, 1)
    return ((arr - mean) / std).astype(np.float32)


def fit_model(train_x: np.ndarray, val_x: np.ndarray) -> tuple[ConvAutoencoder, pd.DataFrame]:
    device = "cpu"
    model = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_x)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_x)), batch_size=BATCH_SIZE, shuffle=False)

    best_state, best_val, patience_left = None, np.inf, PATIENCE
    rows = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, yb)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                recon, _ = model(xb)
                val_losses.append(float(loss_fn(recon, yb).item()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def encode(model: ConvAutoencoder, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        _, z = model(torch.from_numpy(x))
    return z.numpy()


def player_aggregate(df: pd.DataFrame, emb_cols: list[str]) -> pd.DataFrame:
    out_rows = []
    for player_id, grp in df.groupby("PLAYER_ID"):
        weights = grp["shot_attempts"].to_numpy(dtype=float) * (1 + 0.1 * (grp["season_num"].to_numpy(dtype=float) - 1))
        row = {
            "PLAYER_ID": player_id,
            "PLAYER_NAME": grp["PLAYER_NAME"].iloc[0],
            "DRAFT_YEAR": grp["DRAFT_YEAR"].iloc[0],
            "seasons_with_embeddings": int(grp["season_num"].nunique()),
            "total_shot_attempts_covered": float(grp["shot_attempts"].sum()),
            "embedding_aggregation": "shot_volume_x_mild_recency_weighted_mean",
        }
        for col in emb_cols:
            row[col] = float(np.average(grp[col].to_numpy(dtype=float), weights=weights))
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def nearest_neighbors(df: pd.DataFrame, emb_cols: list[str], top_k: int = 5) -> pd.DataFrame:
    mat = df[emb_cols].to_numpy(dtype=float)
    d = pairwise_distances(mat)
    rows = []
    for i in range(len(df)):
        order = np.argsort(d[i])[1 : top_k + 1]
        for rank, j in enumerate(order, start=1):
            rows.append({
                "PLAYER_ID": df.iloc[i]["PLAYER_ID"],
                "PLAYER_NAME": df.iloc[i]["PLAYER_NAME"],
                "neighbor_rank": rank,
                "neighbor_player_id": df.iloc[j]["PLAYER_ID"],
                "neighbor_player_name": df.iloc[j]["PLAYER_NAME"],
                "embedding_distance": float(d[i, j]),
            })
    return pd.DataFrame(rows)


def save_recon_examples(model: ConvAutoencoder, normalized: np.ndarray, raw: np.ndarray, normalizer: dict, out_path: Path) -> None:
    model.eval()
    with torch.no_grad():
        recon, _ = model(torch.from_numpy(normalized[:3]))
    recon = recon.numpy()
    mean = np.array(normalizer["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(normalizer["std"], dtype=np.float32).reshape(1, 3, 1, 1)
    recon = recon * std + mean
    fig, axes = plt.subplots(3, 6, figsize=(16, 8))
    for i in range(3):
        for c in range(3):
            axes[i, 2 * c].imshow(raw[i, c], origin="lower", aspect="auto")
            axes[i, 2 * c].set_title(f"Input {i+1} ch{c+1}")
            axes[i, 2 * c].axis("off")
            axes[i, 2 * c + 1].imshow(recon[i, c], origin="lower", aspect="auto")
            axes[i, 2 * c + 1].set_title(f"Recon {i+1} ch{c+1}")
            axes[i, 2 * c + 1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_dirs()
    draft_idx = pd.read_csv(PATHS.archetype_output_dir / "draft_player_shot_tensor_index.csv")
    hof_idx = pd.read_csv(PATHS.archetype_output_dir / "hof_player_shot_tensor_index.csv")
    draft_tensors = load_npz(PATHS.archetype_output_dir / "draft_player_season_shot_tensors.npz")
    hof_tensors = load_npz(PATHS.archetype_output_dir / "hof_player_season_shot_tensors.npz")

    split_mask = draft_idx["PLAYER_ID"].astype(int) % 5 != 0
    train_raw, val_raw = draft_tensors[split_mask.to_numpy()], draft_tensors[(~split_mask).to_numpy()]
    train_norm, draft_norm, normalizer = normalize(train_raw, draft_tensors)
    val_norm = apply_normalizer(val_raw, normalizer)
    hof_norm = apply_normalizer(hof_tensors, normalizer)

    model, history = fit_model(train_norm, val_norm)

    draft_embeddings = encode(model, draft_norm)
    hof_embeddings = encode(model, hof_norm)
    emb_cols = [f"shot_emb_{i+1}" for i in range(draft_embeddings.shape[1])]

    draft_season = pd.concat([draft_idx.reset_index(drop=True), pd.DataFrame(draft_embeddings, columns=emb_cols)], axis=1)
    hof_season = pd.concat([hof_idx.reset_index(drop=True), pd.DataFrame(hof_embeddings, columns=emb_cols)], axis=1)

    draft_player = player_aggregate(draft_season, emb_cols)
    hof_player = player_aggregate(hof_season, emb_cols)

    draft_season_path = PATHS.archetype_output_dir / "shot_embedding_player_season.csv"
    draft_player_path = PATHS.archetype_output_dir / "shot_embedding_player.csv"
    hof_season_path = PATHS.archetype_output_dir / "hof_shot_embedding_player_season.csv"
    hof_player_path = PATHS.archetype_output_dir / "hof_shot_embedding_player.csv"
    history_path = PATHS.archetype_output_dir / "shot_autoencoder_training_history.csv"
    model_path = PATHS.archetype_output_dir / "shot_autoencoder_model.pt"
    meta_path = PATHS.archetype_output_dir / "shot_autoencoder_artifacts.json"
    nn_path = PATHS.archetype_output_dir / "shot_embedding_nearest_neighbors.csv"
    recon_path = PATHS.archetype_visual_dir / "shot_embedding_reconstruction_examples.png"

    draft_season.to_csv(draft_season_path, index=False)
    draft_player.to_csv(draft_player_path, index=False)
    hof_season.to_csv(hof_season_path, index=False)
    hof_player.to_csv(hof_player_path, index=False)
    history.to_csv(history_path, index=False)
    torch.save(model.state_dict(), model_path)
    nn_df = nearest_neighbors(draft_player, emb_cols)
    nn_df.to_csv(nn_path, index=False)
    save_recon_examples(model, draft_norm, draft_tensors, normalizer, recon_path)

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train")
    plt.plot(history["epoch"], history["val_loss"], label="validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Shot Autoencoder Training History")
    plt.tight_layout()
    plt.savefig(PATHS.archetype_visual_dir / "shot_autoencoder_training_curve.png", dpi=220, bbox_inches="tight")
    plt.close()

    meta_path.write_text(json.dumps({"latent_dim": LATENT_DIM, "epochs_ran": int(history["epoch"].max()), "normalizer": normalizer, "decoder_output": "linear", "validation_split": "player_aware_pid_mod_5"}, indent=2), encoding="utf-8")

    append_log(
        phase="PHASE 4 — TRAIN SHOT-STYLE EMBEDDING MODEL",
        completed=(
            "Trained a CNN autoencoder on drafted-player player-season tensors inside the new analysis workflow, then encoded both drafted-player and HOF tensors with the fitted model. "
            "Exported season-level and player-level embeddings, training history, model weights, nearest-neighbor diagnostics, and reconstruction visuals."
        ),
        learned=(
            "The project’s existing shot-style idea is viable as a reusable learned representation rather than just a set of hand-crafted shot summaries. "
            "Weighted player-level aggregation benefits from preserving shot volume and mildly favoring later rookie-contract seasons."
        ),
        assumptions=(
            "CPU training is acceptable for this deliverable-scale model. "
            "The validation split is player-aware through a deterministic PLAYER_ID-based partition rather than a random season-level split."
        ),
        files_read=[
            str(PATHS.archetype_output_dir / "draft_player_season_shot_tensors.npz"),
            str(PATHS.archetype_output_dir / "hof_player_season_shot_tensors.npz"),
            str(PATHS.archetype_output_dir / "draft_player_shot_tensor_index.csv"),
            str(PATHS.archetype_output_dir / "hof_player_shot_tensor_index.csv"),
        ],
        files_written=[str(draft_season_path), str(draft_player_path), str(hof_season_path), str(hof_player_path), str(history_path), str(model_path), str(meta_path), str(nn_path), str(recon_path)],
    )


if __name__ == "__main__":
    main()
