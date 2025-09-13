from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import math
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool as gmp, global_mean_pool as gap
from torch_geometric.utils import add_self_loops

from utils import evaluate_and_save, nice
from data_loader import Paths

from topo_pool import TopoPool

@dataclass
class TopoPoolConfig:
    hidden: int = 64
    batch_size: int = 32
    epochs: int = 1000
    patience: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warm_start_if_exists: bool = False 
    seed: int = 14

def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _cpu_state_dict(m: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in m.state_dict().items()}

def _save_ckpt_cpu(path: Path, m: nn.Module, meta: Dict[str, Any] | None = None):
    payload: Dict[str, Any] = {"model": _cpu_state_dict(m)}
    if meta:
        payload.update(meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path.as_posix())

def _load_ckpt_cpu(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    ck = torch.load(path.as_posix(), map_location="cpu")
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
        meta = {k: v for k, v in ck.items() if k != "model"}
        return state, meta
    return ck, {}

def _safe_weight(d) -> float:
    w = d.get("weight", 1.0)
    try:
        if w is None:
            return 1.0
        if isinstance(w, float) and math.isnan(w):
            return 1.0
        return float(w)
    except Exception:
        return 1.0

def build_pyg_dataset(ego_graphs: Dict[Any, nx.Graph], flags_df) -> Tuple[List[Data], List[int], List[str]]:
    all_types = set()
    for G in ego_graphs.values():
        for _, nd in G.nodes(data=True):
            t = nd.get("type", "unknown")
            all_types.add(t if t is not None else "unknown")
    all_types.add("unknown")
    type_list = sorted(all_types)
    type_to_idx = {t: i for i, t in enumerate(type_list)}

    data_list: List[Data] = []
    labels: List[int] = []

    for pid, G in ego_graphs.items():
        idx = {n: i for i, n in enumerate(G.nodes())}
        n = len(idx)

        deg = np.zeros(n, dtype=np.float32)
        wdeg = np.zeros(n, dtype=np.float32)
        type_oh = np.zeros((n, len(type_list)), dtype=np.float32)

        for u in G.nodes():
            ui = idx[u]
            deg[ui] = G.degree(u)
            wsum = 0.0
            for _, v, d in G.edges(u, data=True):
                wsum += _safe_weight(d)
            wdeg[ui] = wsum
            t = (G.nodes[u].get("type", "unknown") or "unknown")
            type_oh[ui, type_to_idx.get(t, type_to_idx["unknown"])] = 1.0

        X = np.concatenate([deg.reshape(-1, 1), wdeg.reshape(-1, 1), type_oh], axis=1)
        x = torch.tensor(X, dtype=torch.float)

        rows, cols, wts = [], [], []
        for u, v, d in G.edges(data=True):
            i, j = idx[u], idx[v]
            w = _safe_weight(d)
            rows += [i, j]
            cols += [j, i]
            wts  += [w, w]

        if len(rows) == 0:
            edge_index = torch.arange(n, dtype=torch.long).repeat(2, 1)
            edge_weight = torch.ones(n, dtype=torch.float)
        else:
            edge_index  = torch.tensor([rows, cols], dtype=torch.long)
            edge_weight = torch.tensor(wts, dtype=torch.float)

        edge_index, edge_weight = add_self_loops(
            edge_index,
            edge_attr=edge_weight,
            fill_value=1.0,
            num_nodes=n
        )

        y = int(flags_df.loc[pid, "hospital_expire_flag"]) if pid in flags_df.index else 0
        data = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))
        data.edge_weight = edge_weight
        data_list.append(data)
        labels.append(y)

    return data_list, labels, type_list

def split_loaders(
    data_list: List[Data],
    labels: List[int],
    batch_size: int,
    seed: int = 14
) -> Dict[str, DataLoader]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    y_all = np.array(labels)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(np.zeros(len(y_all)), y_all))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)  # 0.25 of trainm,  5% val
    train_sub, val_idx = next(sss2.split(np.zeros(len(train_idx)), y_all[train_idx]))
    train_idx = train_idx[train_sub]

    train_set = [data_list[i] for i in train_idx]
    val_set   = [data_list[i] for i in val_idx]
    test_set  = [data_list[i] for i in test_idx]

    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True),
        "val":   DataLoader(val_set,   batch_size=batch_size, shuffle=False),
        "test":  DataLoader(test_set,  batch_size=batch_size, shuffle=False),
    }

class TopoPoolNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 2, use_edge_weight: bool = True):
        super().__init__()
        self.use_edge_weight = use_edge_weight
        self.conv1 = GCNConv(in_dim, hidden, add_self_loops=False)
        self.conv2 = GCNConv(hidden, hidden, add_self_loops=False)
        self.pool  = TopoPool(in_channels=hidden, max_clusters=None, gen_edges=False)
        self.lin1  = nn.Linear(hidden * 2, hidden)
        self.lin2  = nn.Linear(hidden, n_classes)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        ew = getattr(data, "edge_weight", None) if self.use_edge_weight else None

        x = F.relu(self.conv1(x, edge_index, edge_weight=ew))
        x = F.relu(self.conv2(x, edge_index, edge_weight=ew))

        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, edge_attr=None, batch=batch)

        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = F.relu(self.lin1(x))
        out = self.lin2(x)
        return out

def _infer_in_dim(loader: DataLoader) -> int:
    sample = next(iter(loader))
    return int(sample.x.size(1))

def _run_epoch(model: nn.Module, loader: DataLoader, device, optimizer: Adam | None, crit) -> Tuple[float, float]:
    train = optimizer is not None
    model.train() if train else model.eval()
    losses, ys, ys_hat = [], [], []
    for batch in loader:
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(batch)
        loss = crit(logits, batch.y)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        ys.append(batch.y.detach().cpu())
        ys_hat.append(logits.detach().cpu().argmax(dim=1))
    y_true = torch.cat(ys).numpy() if ys else np.array([])
    y_pred = torch.cat(ys_hat).numpy() if ys_hat else np.array([])
    acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    return float(np.mean(losses) if losses else 0.0), acc

def train_topopool(
    *,
    paths: Paths,
    loaders: Dict[str, DataLoader],
    cfg: TopoPoolConfig
) -> Dict[str, Any]:
    device = _device()
    in_dim = _infer_in_dim(loaders["train"])
    n_classes = 2
    model = TopoPoolNet(in_dim=in_dim, hidden=cfg.hidden, n_classes=n_classes).to(device)

    models_dir = paths.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = models_dir / "topopool_std_best.pt"
    last_ckpt = models_dir / "topopool_std_last.pt"

    skip_training = False
    if best_ckpt.exists():
        try:
            state, _ = _load_ckpt_cpu(best_ckpt)
            model.load_state_dict(state); model.to(device)
            if cfg.warm_start_if_exists:
                print(f" Warm-started TopoPoolNet from {best_ckpt.name}")
            else:
                print(f"Loaded TopoPoolNet from {best_ckpt.name} skipping training.")
                skip_training = True
        except Exception as e:
            print(f"Could not load BEST checkpoint: {e}, training from scratch.")

    crit = nn.CrossEntropyLoss()
    opt  = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val, bad, best_state = 0.0, 0, None
    if not skip_training:
        for ep in range(1, cfg.epochs + 1):
            tr_loss, tr_acc = _run_epoch(model, loaders["train"], device, opt,  crit)
            va_loss, va_acc = _run_epoch(model, loaders["val"],   device, None, crit)

            _save_ckpt_cpu(last_ckpt, model, meta={"in_dim": in_dim, "hidden": cfg.hidden, "n_classes": n_classes})

            if va_acc > best_val:
                best_val, bad = va_acc, 0
                _save_ckpt_cpu(best_ckpt, model, meta={"in_dim": in_dim, "hidden": cfg.hidden, "n_classes": n_classes})
                best_state = _cpu_state_dict(model)
                print(f"[TopoPool] epoch {ep:03d} NEW BEST val_acc={va_acc:.3f} {best_ckpt.name}")
            else:
                bad += 1

            print(f"[TopoPool {ep:03d}] train {tr_loss:.4f}/{tr_acc:.3f} "
                  f"val {va_loss:.4f}/{va_acc:.3f}, best {best_val:.3f} (pat {bad}/{cfg.patience})")

            if bad >= cfg.patience:
                print("Early stopping.")
                break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state); model.to(device)
    elif best_ckpt.exists():
        state, _ = _load_ckpt_cpu(best_ckpt)
        model.load_state_dict(state); model.to(device)
        print(f"Loaded BEST weights from {best_ckpt.name}")

    te_loss, te_acc = _run_epoch(model, loaders["test"], device, None, crit)
    print(f"TopoPool · test_acc={te_acc:.3f} (best val={best_val:.3f})")
    print(f"Checkpoints: BEST: {best_ckpt.as_posix()}, LAST: {last_ckpt.as_posix()}")

    return {
        "model": model,
        "best_val": best_val,
        "test_acc": te_acc,
        "best_ckpt": best_ckpt,
        "last_ckpt": last_ckpt,
    }

def _expected_in_dim_tp(m: nn.Module, train_loader: DataLoader) -> int:
    # GCNConv conv1.lin.weight: [out, in]
    c1 = getattr(m, "conv1", None)
    lin = getattr(c1, "lin", None) if c1 is not None else None
    if lin is not None and hasattr(lin, "weight"):
        return int(lin.weight.shape[1])
    sample = next(iter(train_loader))
    return int(sample.x.size(1))

@torch.no_grad()
def _ytrue_yprob_tp(m: nn.Module, loader: DataLoader, train_loader: DataLoader, device) -> Tuple[np.ndarray, np.ndarray]:
    m.eval()
    want = _expected_in_dim_tp(m, train_loader)
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        have = x.size(1)
        if have != want:
            if have > want:
                x = x[:, -want:]
            else:
                pad = torch.zeros((x.size(0), want - have), dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=1)
            batch = batch.clone()
            batch.x = x

        logits = m(batch)
        if logits.dim() == 2 and logits.size(1) == 2:
            prob1 = torch.softmax(logits, dim=1)[:, 1]
        else:
            prob1 = torch.sigmoid(logits.squeeze(-1).float())
        ys.append(batch.y.detach().cpu()); ps.append(prob1.detach().cpu())
    y_true = torch.cat(ys).numpy() if ys else np.array([])
    y_prob = torch.cat(ps).numpy() if ps else np.array([])
    return y_true, y_prob

def _metrics(tp_true: np.ndarray, tp_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (tp_prob >= thr).astype(int)
    acc  = accuracy_score(tp_true, y_pred) if tp_true.size else float("nan")
    try:
        auc = roc_auc_score(tp_true, tp_prob) if tp_true.size and len(np.unique(tp_true)) == 2 else np.nan
    except Exception:
        auc = np.nan
    tn, fp, fn, tp = confusion_matrix(tp_true, y_pred, labels=[0,1]).ravel() if tp_true.size else (0,0,0,0)
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    return dict(accuracy=acc, auc=auc, specificity=spec, sensitivity=sens)

def full_eval_topopool(
    *,
    paths: Paths,
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    model_tag: str = "TopoPool",
    seed: int = 14
) -> Any:
    device = _device()
    y_val, p_val = _ytrue_yprob_tp(model, loaders["val"],  loaders["train"], device)
    y_tst, p_tst = _ytrue_yprob_tp(model, loaders["test"], loaders["train"], device)

    if y_val.size == 0 or y_tst.size == 0:
        raise RuntimeError("Empty VAL/TEST predictions for TopoPool; check loaders and model outputs.")

    eval_dir = paths.RESULTS_DIR / "Evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    df = evaluate_and_save(
        model_tag,
        val_true=y_val,  val_prob=p_val,
        test_true=y_tst, test_prob=p_tst,
        out_dir=eval_dir,
        base_threshold=0.5,
        n_boot=2000,
        ece_bins=15,
        ece_strategy="uniform",
        seed=seed,
    )
    snap_csv = eval_dir / f"{model_tag}__metrics_snapshot.csv"
    df.to_csv(snap_csv, index=False)
    return df

def run_topopool_pipeline(
    *,
    paths: Paths,
    ego_graphs: Dict[Any, nx.Graph],
    flags_df,
    cfg: TopoPoolConfig = TopoPoolConfig(),
) -> Dict[str, Any]:
    data_list, labels, type_list = build_pyg_dataset(ego_graphs, flags_df)
    print(f"Built {len(data_list)} PyG graphs. Label counts 0: {labels.count(0)}, 1: {labels.count(1)}")
    loaders = split_loaders(data_list, labels, batch_size=cfg.batch_size, seed=cfg.seed)
    print(f"Split sizes train:{len(loaders['train'].dataset)} "
          f"val:{len(loaders['val'].dataset)} test:{len(loaders['test'].dataset)}")

    train_art = train_topopool(paths=paths, loaders=loaders, cfg=cfg)
    model = train_art["model"]
    df = full_eval_topopool(paths=paths, model=model, loaders=loaders, model_tag="TopoPool", seed=cfg.seed)

    return {
        "type_list": type_list,
        "loaders": loaders,
        "train_artifacts": train_art,
        "eval_df": df,
    }