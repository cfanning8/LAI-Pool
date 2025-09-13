from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as torchdata
from sklearn.base import BaseEstimator, ClassifierMixin

from utils import load_vectors, evaluate_and_save, save_model, nice
from data_loader import Paths  

def _xy_from_pids(pids, X, y, idx_map):
    keep = [p for p in pids if p in idx_map]
    rows = [idx_map[p] for p in keep]
    return X[rows].astype(np.float32), y[rows].astype(int)

def build_vector_sets(paths: Paths, split_std: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    # Load caches
    img_pids, img_y, img_X = load_vectors(paths.IMG_PKL)
    lan_pids, lan_y, lan_X = load_vectors(paths.LAN_PKL)

    # PID to row index
    img_idx = {pid: i for i, pid in enumerate(img_pids)}
    lan_idx = {pid: i for i, pid in enumerate(lan_pids)}

    train_pids = split_std["train_pids_balanced"].astype(object)
    val_pids   = split_std["val_pids_balanced"].astype(object)
    test_pids  = split_std["test_pids"].astype(object)

    # Image
    Xtr_img, ytr_img = _xy_from_pids(train_pids, img_X, img_y, img_idx)
    Xva_img, yva_img = _xy_from_pids(val_pids,   img_X, img_y, img_idx)
    Xte_img, yte_img = _xy_from_pids(test_pids,  img_X, img_y, img_idx)

    # Landscape
    Xtr_lan, ytr_lan = _xy_from_pids(train_pids, lan_X, lan_y, lan_idx)
    Xva_lan, yva_lan = _xy_from_pids(val_pids,   lan_X, lan_y, lan_idx)
    Xte_lan, yte_lan = _xy_from_pids(test_pids,  lan_X, lan_y, lan_idx)

    return {
        "image":     dict(Xtr=Xtr_img, ytr=ytr_img, Xva=Xva_img, yva=yva_img, Xte=Xte_img, yte=yte_img),
        "landscape": dict(Xtr=Xtr_lan, ytr=ytr_lan, Xva=Xva_lan, yva=yva_lan, Xte=Xte_lan, yte=yte_lan),
    }

@dataclass
class MLPConfig:
    epochs: int = 1000
    patience: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.1
    verbose: bool = True
    device: str | None = None

class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, cfg: MLPConfig = MLPConfig()):
        self.cfg = cfg
        self.model_ = None

    def _build(self, d):
        return nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(inplace=True), nn.Dropout(self.cfg.dropout),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(self.cfg.dropout),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(self.cfg.dropout),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, np.float32); y = np.asarray(y, np.int64).ravel()
        d = X.shape[1]
        self.model_ = self._build(d)
        dev = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.to(dev)

        crit = nn.BCELoss()
        opt  = torch.optim.AdamW(self.model_.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        def _loader(Xa, ya, shuffle):
            ds = torchdata.TensorDataset(torch.from_numpy(Xa.astype(np.float32)),
                                         torch.from_numpy(ya.astype(np.int64)))
            return torchdata.DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle, drop_last=False)

        ld_tr = _loader(X, y, shuffle=True)
        ld_va = None
        if X_val is not None and len(X_val):
            X_val = np.asarray(X_val, np.float32); y_val = np.asarray(y_val, np.int64).ravel()
            ld_va = _loader(X_val, y_val, shuffle=False)

        best_state, best_val, no_imp = None, float("inf"), 0
        for ep in range(self.cfg.epochs):
            # train
            self.model_.train()
            for xb, yb in ld_tr:
                xb, yb = xb.to(dev), yb.float().to(dev)
                opt.zero_grad(set_to_none=True)
                p = self.model_(xb).squeeze(1)
                loss = crit(p, yb)
                loss.backward(); opt.step()

            # validate
            if ld_va is not None:
                self.model_.eval()
                vloss, n = 0.0, 0
                with torch.no_grad():
                    for xb, yb in ld_va:
                        xb, yb = xb.to(dev), yb.float().to(dev)
                        p = self.model_(xb).squeeze(1)
                        l = crit(p, yb).item()
                        vloss += l * xb.size(0); n += xb.size(0)
                vloss /= max(1, n)
                if self.cfg.verbose:
                    print(f"MLP ep {ep+1:03d} val_loss={vloss:.4f} (best={best_val:.4f})", end="\r")

                if vloss + 1e-6 < best_val:
                    best_val = vloss
                    best_state = {k: v.detach().cpu() for k, v in self.model_.state_dict().items()}
                    no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= self.cfg.patience:
                    if self.cfg.verbose: print(f"\nEarly stop (patience {self.cfg.patience})")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.to("cpu").eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        if self.model_ is None:
            raise RuntimeError("Model not fit.")
        X = np.asarray(X, np.float32)
        bs, probs = 4096, []
        for i in range(0, len(X), bs):
            p = self.model_(torch.from_numpy(X[i:i+bs])).squeeze(1).numpy()
            probs.append(p)
        p1 = np.concatenate(probs) if probs else np.array([], dtype=np.float32)
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X, y):
        return (self.predict(X) == np.asarray(y).ravel()).mean()

def train_one(tag: str, Xtr, ytr, Xva, yva, Xte, yte, out_name: str, models_dir) -> None:
    path = models_dir / f"mlp_{out_name}_train.joblib"
    if path.exists():
        print(f"Found model, skipping: {nice(path)}")
        return
    print(f"Train MLP  {tag}")
    est = TorchMLPClassifier(MLPConfig())
    est.fit(Xtr, ytr, X_val=Xva, y_val=yva)
    acc_tr = est.score(Xtr, ytr)
    acc_te = est.score(Xte, yte)
    print(f"train_acc={acc_tr:.3f} test_acc={acc_te:.3f}")
    save_model(est, path)

def evaluate_pairs(pairs: Dict[str, Dict[str, np.ndarray]], models_dir, eval_dir, model_prefix="MLP_"):
    import joblib
    import numpy as np
    summaries = {}
    for name, pack in pairs.items():
        model_path = models_dir / f"mlp_{name}_train.joblib"
        if not model_path.exists():
            print(f"Missing model: {nice(model_path)}, skipping {name}")
            continue
        est = joblib.load(model_path)
        pva = est.predict_proba(pack["Xva"])[:, 1]
        pte = est.predict_proba(pack["Xte"])[:, 1]
        model_tag = f"{model_prefix}{name.upper()}"
        df = evaluate_and_save(
            model_tag,
            val_true=pack["yva"], val_prob=pva,
            test_true=pack["yte"], test_prob=pte,
            out_dir=eval_dir,
            base_threshold=0.5,
            n_boot=2000,
            ece_bins=15,
            ece_strategy="uniform",
            seed=14,
        )
        (eval_dir / f"{model_tag}__metrics_snapshot.csv").write_text(df.to_csv(index=False))
        summaries[name] = df
    return summaries

def run_training(paths: Paths, split_std: Dict[str, Any]) -> None:
    pairs = build_vector_sets(paths, split_std)

    # Train both models
    train_one("PI (train_bal → val_bal → test)",
              pairs["image"]["Xtr"], pairs["image"]["ytr"],
              pairs["image"]["Xva"], pairs["image"]["yva"],
              pairs["image"]["Xte"], pairs["image"]["yte"],
              out_name="image", models_dir=paths.MODELS_DIR)

    train_one("LAN (train_bal → val_bal → test)",
              pairs["landscape"]["Xtr"], pairs["landscape"]["ytr"],
              pairs["landscape"]["Xva"], pairs["landscape"]["yva"],
              pairs["landscape"]["Xte"], pairs["landscape"]["yte"],
              out_name="landscape", models_dir=paths.MODELS_DIR)

    # Full eval
    eval_dir = paths.RESULTS_DIR / "Evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    evaluate_pairs(pairs, paths.MODELS_DIR, eval_dir)