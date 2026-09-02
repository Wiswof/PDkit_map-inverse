#!/usr/bin/env python3
"""PyTorch one-hidden-layer GLP-1R QSAR with per-epoch loss curves."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Keep this script self-contained: it does not import the earlier sklearn script.
DESC_LIST = [(name, fn) for name, fn in Descriptors._descList if name != "Ipc"]


def canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def descriptor_row(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    values = []
    for _, fn in DESC_LIST:
        try:
            value = float(fn(mol))
            values.append(value if math.isfinite(value) else np.nan)
        except Exception:
            values.append(np.nan)
    return values


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    value = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    return value or smiles


def parse_exact_ec50(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.startswith((">", "<", "~")):
        return np.nan
    try:
        number = float(text)
        return number if number > 0 else np.nan
    except ValueError:
        return np.nan


def prepare_data(csv_path: Path, assay: int, conflict_log10: float = 0.30):
    target_col = f"Assay_{assay}_EC50_nM"
    replicate_col = f"Assay_{assay}_Number"
    source = pd.read_csv(csv_path)
    required = {"SMILES", target_col}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = source.copy()
    df["canonical_smiles"] = df["SMILES"].map(canonicalize)
    invalid_smiles = int(df["canonical_smiles"].isna().sum())
    df["ec50_nM"] = df[target_col].map(parse_exact_ec50)
    df = df.dropna(subset=["canonical_smiles", "ec50_nM"]).copy()
    df["pEC50"] = 9.0 - np.log10(df["ec50_nM"].astype(float))

    spread = df.groupby("canonical_smiles")["pEC50"].agg(lambda x: x.max() - x.min())
    conflicts = set(spread[spread > conflict_log10].index)
    conflict_rows = df[df["canonical_smiles"].isin(conflicts)].copy()
    clean = df[~df["canonical_smiles"].isin(conflicts)].copy()

    aggregations = {"pEC50": "mean", "ec50_nM": "median"}
    if replicate_col in clean.columns:
        clean[replicate_col] = pd.to_numeric(clean[replicate_col], errors="coerce")
        aggregations[replicate_col] = "max"
    clean = clean.groupby("canonical_smiles", as_index=False).agg(aggregations)
    clean["scaffold"] = clean["canonical_smiles"].map(scaffold)
    report = {
        "input_rows": int(len(source)),
        "invalid_smiles": invalid_smiles,
        "usable_exact_rows_before_conflict_filter": int(len(df)),
        "conflicting_canonical_structures_removed": int(len(conflicts)),
        "conflicting_rows_removed": int(len(conflict_rows)),
        "final_unique_structures": int(len(clean)),
        "unique_scaffolds": int(clean["scaffold"].nunique()),
    }
    return clean, conflict_rows, report


class PCAFirstLayerNet(nn.Module):
    """Raw RDKit descriptors -> fixed embedded preprocessing/PCA -> MLP."""

    def __init__(
        self,
        input_dim: int,
        pca_dim: int,
        hidden_dim: int = 16,
        pca_weight: np.ndarray | None = None,
        pca_bias: np.ndarray | None = None,
        impute_values: np.ndarray | None = None,
    ):
        super().__init__()
        self.register_buffer(
            "impute_values",
            torch.zeros(input_dim, dtype=torch.float32)
            if impute_values is None
            else torch.as_tensor(impute_values, dtype=torch.float32),
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim, pca_dim),
            nn.Linear(pca_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        if pca_weight is not None:
            with torch.no_grad():
                self.network[0].weight.copy_(torch.as_tensor(pca_weight, dtype=torch.float32))
                self.network[0].bias.copy_(torch.as_tensor(pca_bias, dtype=torch.float32))
        self.network[0].weight.requires_grad_(False)
        self.network[0].bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(torch.isfinite(x), x, self.impute_values)
        return self.network(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_preprocessor(pca_variance: float, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, svd_solver="full", random_state=seed)),
        ]
    )


def embedded_pca_parameters(preprocessor: Pipeline) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fold imputation metadata and StandardScaler into a raw-input PCA layer."""
    imputer = preprocessor.named_steps["impute"]
    scaler = preprocessor.named_steps["scale"]
    pca = preprocessor.named_steps["pca"]
    weight = pca.components_ / scaler.scale_[None, :]
    bias = -pca.components_ @ (scaler.mean_ / scaler.scale_ + pca.mean_)
    return (
        weight.astype(np.float32),
        bias.astype(np.float32),
        imputer.statistics_.astype(np.float32),
    )


def build_pca_model(preprocessor: Pipeline, hidden: int) -> PCAFirstLayerNet:
    weight, bias, impute_values = embedded_pca_parameters(preprocessor)
    return PCAFirstLayerNet(
        input_dim=weight.shape[1],
        pca_dim=weight.shape[0],
        hidden_dim=hidden,
        pca_weight=weight,
        pca_bias=bias,
        impute_values=impute_values,
    )


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    hidden: int,
    pca_variance: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
):
    set_seed(seed)
    preprocessor = make_preprocessor(pca_variance, seed)
    preprocessor.fit(X_train)
    X_train_t = X_train.astype(np.float32)
    X_val_t = X_val.astype(np.float32)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std()) or 1.0
    y_train_z = ((y_train - y_mean) / y_std).astype(np.float32)

    xt = torch.from_numpy(X_train_t)
    yt = torch.from_numpy(y_train_z)
    xv = torch.from_numpy(X_val_t)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(xt, yt),
        batch_size=min(batch_size, len(xt)),
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    model = build_pca_model(preprocessor, hidden)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = model(xt).numpy() * y_std + y_mean
            val_pred = model(xv).numpy() * y_std + y_mean
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean((train_pred - y_train) ** 2)),
                "val_loss": float(np.mean((val_pred - y_val) ** 2)),
            }
        )

    return val_pred, history


def train_final(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    hidden: int,
    pca_variance: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
):
    set_seed(seed)
    preprocessor = make_preprocessor(pca_variance, seed)
    preprocessor.fit(X)
    X_t = X.astype(np.float32)
    y_mean = float(y.mean())
    y_std = float(y.std()) or 1.0
    y_z = ((y - y_mean) / y_std).astype(np.float32)
    xt, yt = torch.from_numpy(X_t), torch.from_numpy(y_z)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(xt, yt),
        batch_size=min(batch_size, len(xt)),
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    model = build_pca_model(preprocessor, hidden)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model, preprocessor, y_mean, y_std


def plot_results(history: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray,
                 metrics: dict, output: Path) -> None:
    summary = history.groupby("epoch")[["train_loss", "val_loss"]].agg(["mean", "std"])
    epoch = summary.index.to_numpy()
    train_mean = summary[("train_loss", "mean")].to_numpy()
    train_std = summary[("train_loss", "std")].fillna(0).to_numpy()
    val_mean = summary[("val_loss", "mean")].to_numpy()
    val_std = summary[("val_loss", "std")].fillna(0).to_numpy()

    fig, (ax_loss, ax_pred) = plt.subplots(1, 2, figsize=(12, 5.2))
    ax_loss.plot(epoch, train_mean, label="Train MSE", color="#2878b5")
    ax_loss.fill_between(epoch, np.maximum(0, train_mean - train_std),
                         train_mean + train_std, color="#2878b5", alpha=0.13)
    ax_loss.plot(epoch, val_mean, label="Validation MSE", color="#e07b39")
    ax_loss.fill_between(epoch, np.maximum(0, val_mean - val_std),
                         val_mean + val_std, color="#e07b39", alpha=0.13)
    ax_loss.set(xlabel="Epoch", ylabel="MSE loss (pEC50²)",
                title="Mean loss across scaffold folds")
    ax_loss.set_yscale("log")
    ax_loss.grid(alpha=0.2)
    ax_loss.legend(frameon=False)

    lo = min(actual.min(), predicted.min()) - 0.25
    hi = max(actual.max(), predicted.max()) + 0.25
    line = np.linspace(lo, hi, 200)
    ax_pred.fill_between(line, line - 1, line + 1, color="#b9d8f2", alpha=0.35)
    ax_pred.plot(line, line, "--", color="#333333", linewidth=1.3)
    ax_pred.scatter(actual, predicted, s=38, alpha=0.78, color="#2878b5",
                    edgecolors="white", linewidths=0.5)
    ax_pred.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Measured pEC50",
                ylabel="Out-of-fold predicted pEC50", title="Cross-validation predictions")
    ax_pred.set_aspect("equal", adjustable="box")
    ax_pred.grid(alpha=0.2)
    ax_pred.text(
        0.98, 0.03,
        f"n = {len(actual)}\nMAE = {metrics['MAE_pEC50']:.2f}\n"
        f"RMSE = {metrics['RMSE_pEC50']:.2f}\nR² = {metrics['R2_pEC50']:.2f}",
        transform=ax_pred.transAxes, ha="right", va="bottom",
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#dddddd"},
    )
    fig.suptitle(f"PyTorch GLP-1R Assay {metrics['assay']} — one hidden layer",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--assay", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("model_output_torch"))
    # The validation curve for this small dataset begins to overfit after ~20 epochs.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.pca_variance < 1.0:
        parser.error("--pca-variance must be between 0 and 1")

    torch.set_num_threads(1)
    data, conflicts, report = prepare_data(args.csv, args.assay)
    X = np.asarray([descriptor_row(s) for s in data["canonical_smiles"]], dtype=float)
    y = data["pEC50"].to_numpy(dtype=float)
    groups = data["scaffold"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)

    oof = np.full(len(y), np.nan)
    all_history = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        pred, history = train_fold(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx], args.epochs,
            args.hidden, args.pca_variance, args.batch_size,
            args.lr, args.weight_decay, args.seed + fold,
        )
        oof[val_idx] = pred
        for row in history:
            row["fold"] = fold
            all_history.append(row)

    metrics = {
        "assay": args.assay,
        "framework": f"PyTorch {torch.__version__}",
        "architecture": (
            f"All RDKit descriptors -> fixed embedded PCA -> "
            f"Dense({args.hidden}, SiLU) -> Dense(1)"
        ),
        "activation": "SiLU",
        "pca_variance_retained": args.pca_variance,
        "endpoint": "pEC50 = 9 - log10(EC50_nM)",
        "cv": f"{n_splits}-fold GroupKFold by Bemis-Murcko scaffold",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "shuffle_each_epoch": True,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "MAE_pEC50": float(mean_absolute_error(y, oof)),
        "RMSE_pEC50": float(mean_squared_error(y, oof) ** 0.5),
        "R2_pEC50": float(r2_score(y, oof)),
        **report,
    }

    model, preprocessor, y_mean, y_std = train_final(
        X, y, args.epochs, args.hidden, args.pca_variance, args.batch_size,
        args.lr, args.weight_decay, args.seed
    )
    metrics["pca_input_dim"] = int(model.network[0].in_features)
    metrics["pca_output_dim"] = int(model.network[0].out_features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"glp1r_assay{args.assay}_torch"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": model.network[0].in_features,
            "pca_dim": model.network[0].out_features,
            "hidden_dim": args.hidden,
            "activation": "SiLU",
            "pca_first_layer_frozen": True,
            "pca_variance_retained": args.pca_variance,
            "y_mean": y_mean,
            "y_std": y_std,
            "descriptor_names": [name for name, _ in DESC_LIST],
            "descriptor_scale": preprocessor.named_steps["scale"].scale_.astype(np.float32),
        },
        f"{prefix}_model.pt",
    )
    joblib.dump(preprocessor, f"{prefix}_preprocessor.joblib")
    history_df = pd.DataFrame(all_history)
    history_df.to_csv(f"{prefix}_history.csv", index=False)
    data.assign(oof_pred_pEC50=oof).to_csv(f"{prefix}_oof_predictions.csv", index=False)
    conflicts.to_csv(f"{prefix}_conflicts.csv", index=False)
    Path(f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_results(history_df, y, oof, metrics, Path(f"{prefix}_training.png"))
    print(
        f"PCA dimension: {metrics['pca_input_dim']} -> "
        f"{metrics['pca_output_dim']} "
        f"(retained variance={args.pca_variance:.1%})"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
