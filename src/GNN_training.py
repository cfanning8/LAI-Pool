from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.utils import data as torchdata
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, SAGEConv, GINConv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm 

from utils import evaluate_and_save, nice
from data_loader import Paths 

@dataclass
class GNNConfig:
    hidden: int = 64
    dropout: float = 0.5
    batch_size: int = 32
    epochs: int = 1000
    patience: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warm_start_if_exists: bool = False
    device: str | None = None 

def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}

def _save_ckpt_cpu(path: Path, model: nn.Module):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": _cpu_state_dict(model)}, path.as_posix())

def _load_ckpt_cpu(path: Path) -> Dict[str, torch.Tensor]:
    ck = torch.load(path.as_posix(), map_location="cpu")
    return ck["model"] if isinstance(ck, dict) and "model" in ck else ck

def _require_label(pid: int, flags_df) -> int:
    return int(flags_df.loc[pid, "hospital_expire_flag"])

def _build_vocab(ego_graphs) -> Tuple[List[str], Dict[str, int]]:
    seen = set()
    for G in ego_graphs.values():
        for _, nd in G.nodes(data=True):
            seen.add(nd.get("type", "unknown"))
    vocab = sorted(seen)
    if "unknown" not in vocab:
        vocab.append("unknown")
    idx = {t: i for i, t in enumerate(vocab)}
    return vocab, idx

def _graph_to_data(G, y: int, vocab, IDX) -> Data:
    X = np.zeros((G.number_of_nodes(), len(vocab)), dtype=np.float32)
    n2i = {n: i for i, n in enumerate(G.nodes())}
    for n, nd in G.nodes(data=True):
        X[n2i[n], IDX.get(nd.get("type", "unknown"), IDX["unknown"])] = 1.0
    edges = []
    for u, v in G.edges():
        ui, vi = n2i[u], n2i[v]
        edges.append([ui, vi]); edges.append([vi, ui])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return Data(x=torch.from_numpy(X), edge_index=edge_index, y=torch.tensor(int(y), dtype=torch.long))

def _strict_pid_lists(split_std: Dict[str, Any], ego_graphs, flags_df):
    tr = list(map(int, split_std["train_pids_balanced"]))
    va = list(map(int, split_std["val_pids_balanced"]))
    te = list(map(int, split_std["test_pids"]))

    have_graph = set(ego_graphs.keys())
    have_label = set(flags_df.index)

    def _keep(pids):
        kept = [p for p in pids if (p in have_graph and p in have_label)]
        return kept

    tr_keep = _keep(tr)
    va_keep = _keep(va)
    te_keep = _keep(te)

    if not tr_keep or not va_keep or not te_keep:
        raise RuntimeError("Empty split after filtering for graph/label presence.")

    return tr_keep, va_keep, te_keep

def build_loaders(
    *,
    ego_graphs,
    flags_df,
    split_std: Dict[str, Any],
    batch_size: int = 32,
) -> Dict[str, DataLoader]:
    vocab, IDX = _build_vocab(ego_graphs)
    tr_pids, va_pids, te_pids = _strict_pid_lists(split_std, ego_graphs, flags_df)

    train_list = [_graph_to_data(ego_graphs[p], _require_label(p, flags_df), vocab, IDX) for p in tr_pids]
    val_list   = [_graph_to_data(ego_graphs[p], _require_label(p, flags_df), vocab, IDX) for p in va_pids]
    test_list  = [_graph_to_data(ego_graphs[p], _require_label(p, flags_df), vocab, IDX) for p in te_pids]

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_list,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_list,  batch_size=batch_size, shuffle=False)

    return {"train": train_loader, "val": val_loader, "test": test_loader}

def _init_head(hidden: int, n_classes: int, dropout: float) -> nn.Sequential:
    head = nn.Sequential(
        nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden, n_classes),
    )
    for m in head:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in = m.weight.size(1)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
    return head

class GCNKW(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, improved=False)
        self.conv2 = GCNConv(hidden, hidden, improved=False)
        self.dropout = dropout
        self.head = _init_head(hidden, n_classes, dropout)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x)

class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = dropout
        self.head = _init_head(hidden, n_classes, dropout)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x)

def _mlp(in_c, hid_c, out_c):
    return nn.Sequential(nn.Linear(in_c, hid_c), nn.ReLU(inplace=True), nn.Linear(hid_c, out_c))

class GINNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.gin1 = GINConv(_mlp(in_dim, hidden, hidden), train_eps=True)
        self.bn1  = BatchNorm(hidden)
        self.gin2 = GINConv(_mlp(hidden, hidden, hidden), train_eps=True)
        self.bn2  = BatchNorm(hidden)
        self.dropout = dropout
        self.head = _init_head(hidden, n_classes, dropout)

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gin1(x, edge_index); x = self.bn1(x); x = F.relu(x); x = F.dropout(x, self.dropout, self.training)
        x = self.gin2(x, edge_index); x = self.bn2(x); x = F.relu(x); x = F.dropout(x, self.dropout, self.training)
        x = global_mean_pool(x, batch)
        return self.head(x)

def _infer_dims(loader) -> Tuple[int, int]:
    sample = next(iter(loader))
    in_dim = int(sample.x.size(-1))
    y = sample.y
    n_classes = int(y.max().item() + 1) if y.ndim > 0 else int(y.item()) + 1
    return in_dim, n_classes

def _device(cfg: GNNConfig) -> torch.device:
    if cfg.device:
        return torch.device(cfg.device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_epoch(model, loader, optimizer, loss_fn, train: bool, device):
    model.train() if train else model.eval()
    losses, y_true, y_pred = [], [], []
    for batch in tqdm(loader, leave=False, desc="train" if train else "val"):
        batch = batch.to(device)
        if train:
            optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch.y)
        if train:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        y_true.append(batch.y.detach().cpu())
        y_pred.append(logits.detach().cpu().argmax(dim=1))
    y_true = torch.cat(y_true).numpy() if y_true else np.array([])
    y_pred = torch.cat(y_pred).numpy() if y_pred else np.array([])
    acc = float(accuracy_score(y_true, y_pred)) if y_true.size else 0.0
    return float(np.mean(losses) if losses else 0.0), acc

def evaluate(model, loader, loss_fn, device):
    return run_epoch(model, loader, optimizer=None, loss_fn=loss_fn, train=False, device=device)

def train_gnn(
    *,
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    ckpt_prefix: str,
    models_dir: Path,
    cfg: GNNConfig,
):
    device = _device(cfg)
    best_path = models_dir / f"{ckpt_prefix}_best.pt"
    last_path = models_dir / f"{ckpt_prefix}_last.pt"

    # If a best checkpoint exists…
    if best_path.exists():
        if cfg.warm_start_if_exists:
            try:
                state = _load_ckpt_cpu(best_path)
                model.load_state_dict(state)
                model.to(device)  # ensure correct device
            except Exception:
                pass
        else:
            try:
                state = _load_ckpt_cpu(best_path)
                model.load_state_dict(state)
                model.to(device)  # ensure correct device
                return {
                    "best": best_path, "last": last_path, "best_val_acc": float("nan"),
                    "history": [], "skipped_training": True
                }
            except Exception:
                pass

    model.to(device)

    opt  = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss()
    history: List[Tuple[int, float, float, float, float]] = []
    best_val, bad_epochs = 0.0, 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], opt, crit, True, device)
        va_loss, va_acc = run_epoch(model, loaders["val"],   opt, crit, False, device)
        history.append((epoch, tr_loss, tr_acc, va_loss, va_acc))

        _save_ckpt_cpu(last_path, model)
        if va_acc > best_val:
            best_val, bad_epochs = va_acc, 0
            _save_ckpt_cpu(best_path, model)
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            break

    return {"best": best_path, "last": last_path, "best_val_acc": best_val, "history": history}

def load_weights(model: nn.Module, path: Path, cfg: GNNConfig) -> nn.Module:
    state = _load_ckpt_cpu(path)
    model.load_state_dict(state)
    return model.to(_device(cfg))

def _expected_in_dim(model: nn.Module, train_loader: DataLoader) -> int:
    c1 = getattr(model, "conv1", None)
    if c1 is not None:
        lin = getattr(c1, "lin", None)
        if lin is not None and hasattr(lin, "weight"):
            return int(lin.weight.shape[1])
        for name in ("lin_l", "lin_src", "lin_r", "lin_dst"):
            l = getattr(c1, name, None)
            if l is not None and hasattr(l, "weight"):
                return int(l.weight.shape[1])
    gin1 = getattr(model, "gin1", None)
    mlp  = getattr(gin1, "nn", None) or getattr(gin1, "apply_func", None)
    if mlp is not None:
        for layer in mlp.modules():
            if isinstance(layer, nn.Linear):
                return int(layer.in_features)
    sample = next(iter(train_loader))
    return int(sample.x.size(1))

@torch.no_grad()
def _ytrue_yprob(model: nn.Module, loader: DataLoader, device, want_dim: int | None, train_loader: DataLoader):
    model.eval()
    if want_dim is None:
        want_dim = _expected_in_dim(model, train_loader)
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        x = batch.x
        have = x.size(1)
        if have != want_dim:
            if have > want_dim:
                x = x[:, -want_dim:]
            else:
                pad = torch.zeros((x.size(0), want_dim - have), dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=1)
            batch = batch.clone()
            batch.x = x
        logits = model(batch)
        if logits.dim() == 2 and logits.size(1) == 2:
            prob1 = torch.softmax(logits, dim=1)[:, 1]
        else:
            prob1 = torch.sigmoid(logits.squeeze(-1).float())
        ys.append(batch.y.detach().cpu()); ps.append(prob1.detach().cpu())
    y_true = torch.cat(ys).numpy() if ys else np.array([])
    y_prob = torch.cat(ps).numpy() if ps else np.array([])
    return y_true, y_prob

def _basic_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    acc  = accuracy_score(y_true, y_pred) if y_true.size else float("nan")
    try:
        auc = roc_auc_score(y_true, y_prob) if y_true.size and len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        auc = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel() if y_true.size else (0,0,0,0)
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    return dict(accuracy=acc, auc=auc, specificity=spec, sensitivity=sens)

def build_models(in_dim: int, n_classes: int, cfg: GNNConfig) -> Dict[str, nn.Module]:
    return {
        "GCN": GCNKW(in_dim, cfg.hidden, n_classes, cfg.dropout),
        "GraphSAGE": GraphSAGE(in_dim, cfg.hidden, n_classes, cfg.dropout),
        "GIN": GINNet(in_dim, cfg.hidden, n_classes, cfg.dropout),
    }

def train_all_gnns(
    *,
    paths: Paths,
    loaders: Dict[str, DataLoader],
    cfg: GNNConfig,
) -> Dict[str, Dict[str, Any]]:
    in_dim, n_classes = _infer_dims(loaders["train"])
    models = build_models(in_dim, n_classes, cfg)
    results = {}
    for name, model in models.items():
        res = train_gnn(
            model=model,
            loaders=loaders,
            ckpt_prefix=name.lower(),
            models_dir=paths.MODELS_DIR,
            cfg=cfg,
        )
        results[name] = res
    return results

def test_accuracy_summary(
    *,
    paths: Paths,
    loaders: Dict[str, DataLoader],
    cfg: GNNConfig,
) -> Dict[str, float]:
    in_dim, n_classes = _infer_dims(loaders["train"])
    models = build_models(in_dim, n_classes, cfg)
    device = _device(cfg)
    out: Dict[str, float] = {}
    for name, model in models.items():
        best = paths.MODELS_DIR / f"{name.lower()}_best.pt"
        if best.exists():
            model = load_weights(model, best, cfg)
            _, acc = evaluate(model, loaders["test"], nn.CrossEntropyLoss(), device)
            out[name] = acc
    return out

def full_eval_all_gnns(
    *,
    paths: Paths,
    loaders: Dict[str, DataLoader],
    cfg: GNNConfig,
) -> Dict[str, Any]:
    in_dim, n_classes = _infer_dims(loaders["train"])
    models = build_models(in_dim, n_classes, cfg)
    device = _device(cfg)
    eval_dir = paths.RESULTS_DIR / "Evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for name, model in models.items():
        best = paths.MODELS_DIR / f"{name.lower()}_best.pt"
        if not best.exists():
            continue
        model = load_weights(model, best, cfg)

        want = _expected_in_dim(model, loaders["train"])
        y_val, p_val = _ytrue_yprob(model, loaders["val"], device, want, loaders["train"])
        y_tst, p_tst = _ytrue_yprob(model, loaders["test"], device, want, loaders["train"])

        if y_val.size == 0 or y_tst.size == 0:
            continue

        model_tag = f"GNN_{name}"
        df = evaluate_and_save(
            model_tag,
            val_true=y_val,  val_prob=p_val,
            test_true=y_tst, test_prob=p_tst,
            out_dir=eval_dir,
            base_threshold=0.5,
            n_boot=2000,
            ece_bins=15,
            ece_strategy="uniform",
            seed=14,
        )
        snap_csv = eval_dir / f"{model_tag}__metrics_snapshot.csv"
        df.to_csv(snap_csv, index=False)
        summaries[name] = df
    return summaries

def run_gnn_pipeline(
    *,
    paths: Paths,
    ego_graphs,
    flags_df,
    split_std: Dict[str, Any],
    cfg: GNNConfig = GNNConfig(),
) -> Dict[str, Any]:
    loaders = build_loaders(ego_graphs=ego_graphs, flags_df=flags_df, split_std=split_std, batch_size=cfg.batch_size)
    train_results = train_all_gnns(paths=paths, loaders=loaders, cfg=cfg)
    test_acc = test_accuracy_summary(paths=paths, loaders=loaders, cfg=cfg)
    eval_summaries = full_eval_all_gnns(paths=paths, loaders=loaders, cfg=cfg)
    return dict(loaders=loaders, train_results=train_results, test_acc=test_acc, eval_summaries=eval_summaries)