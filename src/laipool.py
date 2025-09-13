from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.nn import global_max_pool as gmp, global_mean_pool as gap

def build_pid2idx(pd_std: List[Tuple[Any, Any, dict]]) -> Dict[Any, int]:
    return {pid: i for i, (pid, *_rest) in enumerate(pd_std)}

def validate_pid2idx(
    pid2idx: Dict[Any, int],
    pd_std: List[Tuple[Any, Any, dict]],
    *,
    ego_graphs: Optional[dict] = None,
    flags_df=None,
    split_std: Optional[dict] = None,
) -> dict:
    report = {
        "n_pd_std": len(pd_std),
        "n_pid2idx": len(pid2idx),
        "duplicates": [],
        "missing_in_graphs": [],
        "missing_in_flags": [],
        "missing_in_pid2idx_from_split": [],
        "examples": {},
    }

    seen, dups = set(), []
    for pid, *_ in pd_std:
        if pid in seen:
            dups.append(pid)
        seen.add(pid)
    report["duplicates"] = sorted(set(dups))

    if ego_graphs is not None:
        report["missing_in_graphs"] = [pid for pid in pid2idx if pid not in ego_graphs]
    if flags_df is not None and hasattr(flags_df, "index"):
        report["missing_in_flags"] = [pid for pid in pid2idx if pid not in flags_df.index]
    if split_std is not None:
        all_split = []
        for k in ('train_pids_balanced', 'val_pids_balanced', 'test_pids',
                  'train_pids', 'val_pids'):
            if k in split_std:
                all_split += list(map(int, split_std[k]))
        miss = [p for p in set(all_split) if p not in pid2idx]
        report["missing_in_pid2idx_from_split"] = sorted(miss)

    report["examples"] = {
        "first_pd_std": pd_std[0][0] if pd_std else None,
        "first_pid2idx": next(iter(pid2idx.items())) if pid2idx else None,
    }
    return report

def pid_to_idx(pid, pid2idx: Dict[Any, int], default: int = -1) -> int:
    return pid2idx.get(pid, default)

def build_type_vocab(ego_graphs: dict) -> Tuple[List[str], Dict[str, int]]:
    ALL_TYPES = set()
    for G in ego_graphs.values():
        for _, d in G.nodes(data=True):
            t = d.get("type", "unknown") or "unknown"
            ALL_TYPES.add(t)
    ALL_TYPES.add("unknown")
    TYPE_LIST  = sorted(ALL_TYPES)
    TYPE_TO_IDX = {t: i for i, t in enumerate(TYPE_LIST)}
    return TYPE_LIST, TYPE_TO_IDX

def _safe_weight(d) -> float:
    try:
        w = d.get("weight", 1.0)
        if w is None or (isinstance(w, float) and math.isnan(w)):
            return 1.0
        return float(w)
    except Exception:
        return 1.0

def to_pyg(
    pid: int,
    G,
    y: int,
    TYPE_LIST: List[str],
    TYPE_TO_IDX: Dict[str, int],
    pid2idx: Dict[Any, int],
) -> Data:
    nodes = list(G.nodes()); idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    deg   = np.zeros(n, np.float32)
    wdeg  = np.zeros(n, np.float32)
    Xtype = np.zeros((n, len(TYPE_LIST)), np.float32)

    for u in nodes:
        ui = idx[u]
        deg[ui] = G.degree(u)
        wsum = 0.0
        for _, _, d in G.edges(u, data=True):
            wsum += _safe_weight(d)
        wdeg[ui] = wsum
        t = (G.nodes[u].get("type", "unknown") or "unknown")
        Xtype[ui, TYPE_TO_IDX.get(t, TYPE_TO_IDX["unknown"])] = 1.0

    X = np.concatenate([deg[:, None], wdeg[:, None], Xtype], axis=1)
    x = torch.tensor(X, dtype=torch.float)

    rows, cols, wts = [], [], []
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        w = _safe_weight(d)
        rows += [i, j]; cols += [j, i]; wts += [w, w]
    edge_index = torch.tensor([rows, cols], dtype=torch.long) if rows else torch.empty((2, 0), dtype=torch.long)
    edge_weight = torch.tensor(wts, dtype=torch.float) if wts else None

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor(int(y), dtype=torch.long),
        pid_idx=torch.tensor(int(pid2idx.get(pid, -1)), dtype=torch.long),
    )
    if edge_weight is not None:
        data.edge_weight = edge_weight
    return data

def build_std_loaders(
    split_std: dict,
    ego_graphs: dict,
    flags_df,
    pid2idx: Dict[Any, int],
    TYPE_LIST: List[str],
    TYPE_TO_IDX: Dict[str, int],
    *,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
):
    def _keep_pid(p):
        p = int(p)
        return (p in ego_graphs) and (p in flags_df.index) and (p in pid2idx)

    def _graph_to_data_with_pid(pid):
        pid = int(pid)
        return to_pyg(
            pid,
            ego_graphs[pid],
            int(flags_df.loc[pid, "hospital_expire_flag"]),
            TYPE_LIST, TYPE_TO_IDX, pid2idx
        )

    tr_pids = [int(p) for p in split_std["train_pids_balanced"] if _keep_pid(p)]
    va_pids = [int(p) for p in split_std["val_pids"] if _keep_pid(p)]
    te_pids = [int(p) for p in split_std["test_pids"] if _keep_pid(p)]

    train_list = [_graph_to_data_with_pid(p) for p in tr_pids]
    val_list   = [_graph_to_data_with_pid(p) for p in va_pids]
    test_list  = [_graph_to_data_with_pid(p) for p in te_pids]

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=shuffle_train,
                              pin_memory=torch.cuda.is_available(), num_workers=num_workers)
    val_loader   = DataLoader(val_list,   batch_size=batch_size, shuffle=False,
                              pin_memory=torch.cuda.is_available(), num_workers=num_workers)
    test_loader  = DataLoader(test_list,  batch_size=batch_size, shuffle=False,
                              pin_memory=torch.cuda.is_available(), num_workers=num_workers)

    feat_dim = int(train_list[0].x.size(1)) if train_list else None
    info = dict(n_train=len(train_list), n_val=len(val_list), n_test=len(test_list), feat_dim=feat_dim)
    return train_loader, val_loader, test_loader, info

class SAPILayer(nn.Module):
    def __init__(self, *, res=(32, 32), sigma=2.0, decay="exp", d=32, debug=False, eps=1e-6):
        super().__init__()
        self.H, self.W = res
        self.log_sigma  = nn.Parameter(torch.log(torch.tensor(float(sigma))))
        self.log_lambda = nn.Parameter(torch.zeros(1))
        self.decay_type = decay
        self.W_Q = nn.Linear(2, d, bias=False)
        self.W_K = nn.Linear(2, d, bias=False)
        self.W_V = nn.Linear(2, d, bias=False)
        self._debug = debug
        self._dbg_printed = False
        self.eps = eps

    @staticmethod
    def _phi(x):  return F.elu(x, alpha=1.0) + 1.0
    def _psi(self, y):
        lam = self.log_lambda.exp().clamp(min=1e-6)
        if   self.decay_type == "gauss": return y * torch.exp(-(y ** 2) / lam ** 2)
        elif self.decay_type == "exp":   return y * torch.exp(-y / lam)
        elif self.decay_type == "ramp":  return y / (lam + y)
        else: raise ValueError("decay must be gauss, exp, ramp")

    def forward(self, diags, *, return_weights=False, debug=None):
        if debug is None: debug = self._debug
        B = len(diags); dev = diags[0].device if B>0 else torch.device("cpu")
        out = torch.zeros(B, 1, self.H, self.W, device=dev)
        weights_out = []
        sigma2   = self.log_sigma.exp().pow(2)
        ys_l = torch.linspace(0,1,self.H,device=dev)
        xs_l = torch.linspace(0,1,self.W,device=dev)
        GY, GX = torch.meshgrid(ys_l, xs_l, indexing="ij")
        for b, pd in enumerate(diags):
            if pd.numel() == 0:
                weights_out.append(torch.empty(0, device=dev)); continue
            pd = pd[pd[:,0] <= 1]
            if pd.numel() == 0:
                weights_out.append(torch.empty(0, device=dev)); continue
            x = pd[:,1]; y = pd[:,2]-pd[:,1]
            pts = torch.stack([x,y],1)
            q = self.W_Q(pts); k = self.W_K(pts); v = self.W_V(pts)
            qφ = self._phi(q);  kφ = self._phi(k)
            kv   = kφ.transpose(0,1) @ v
            ksum = kφ.sum(dim=0)
            num   = qφ @ kv
            denom = (qφ @ ksum.unsqueeze(1)).squeeze(1) + self.eps
            z     = num / denom.unsqueeze(1)
            w_vec = z.norm(dim=1) * self._psi(y)
            g = torch.exp(-((GX[None]-x[:,None,None])**2 + (GY[None]-y[:,None,None])**2) / (2*sigma2))
            out[b,0] = (w_vec[:,None,None] * g).sum(0)
            weights_out.append(w_vec)
            if debug and not self._dbg_printed:
                print(f"[SAPILayer] sigma={self.log_sigma.exp():.3f}, lambda={self.log_lambda.exp():.3f}")
                self._dbg_printed = True
        return (out, weights_out) if return_weights else out

class TopoTokens(nn.Module):
    def __init__(self, grid=(8,8), dtok=64, dropout=0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(grid)
        self.proj = nn.Linear(1, dtok, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, Pi):  # [B,1,H,W]
        pooled = self.pool(Pi)
        B, _, Th, Tw = pooled.shape
        T = Th * Tw
        tokens = pooled.view(B, 1, T).transpose(1, 2)  # [B,T,1]
        return self.drop(self.proj(tokens))            # [B,T,dtok]

class TopoPoolSAPI(nn.Module):
    def __init__(self, in_channels: int, ratio: float = 0.5, dtok: int = 64, datt: int = 64):
        super().__init__()
        self.q = nn.Linear(in_channels, datt, bias=False)
        self.k = nn.Linear(dtok,        datt, bias=False)
        self.pool = TopKPooling(in_channels, ratio=ratio)

    def forward(self, x, edge_index, edge_attr=None, batch=None, topo_tokens=None):
        if topo_tokens is not None and topo_tokens.numel() > 0:
            t_bar = topo_tokens.mean(dim=1) # [B, dtok]
            qx = self.q(x) # [N, datt]
            kb = self.k(t_bar)[batch] # [N, datt]
            attn = (qx * kb).sum(dim=1) / math.sqrt(max(1, qx.size(1))) # [N]
            x, edge_index, edge_attr, batch, perm, score = self.pool(
                x, edge_index, edge_attr=edge_attr, batch=batch, attn=attn
            )
        else:
            x, edge_index, edge_attr, batch, perm, score = self.pool(
                x, edge_index, edge_attr=edge_attr, batch=batch
            )
        return x, edge_index, edge_attr, batch, perm, score

class TopoPoolSAPINet_Tokens(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2,
                 hidden: int = 64, dropout: float = 0.0,
                 pi_res=(32,32), sigma_pi=2.0, decay="exp",
                 tok_grid=(8,8), dtok=64, datt=64, pool_ratio=0.5):
        super().__init__()
        self.pi = SAPILayer(res=pi_res, sigma=sigma_pi, decay=decay, d=32,
                            debug=bool(int(torch.getenv("DEBUG_PI") or 0)))
        self.tokens = TopoTokens(grid=tok_grid, dtok=dtok, dropout=0.10)

        self.conv1 = GCNConv(in_dim, hidden, add_self_loops=False)
        self.conv2 = GCNConv(hidden, hidden, add_self_loops=False)
        self.act   = nn.ReLU()
        self.drop  = nn.Dropout(dropout)

        self.pool  = TopoPoolSAPI(in_channels=hidden, ratio=pool_ratio, dtok=dtok, datt=datt)

        self.lin1  = nn.Linear(hidden * 2, hidden)
        self.lin2  = nn.Linear(hidden, n_classes)

    def forward(self, data, pd_list=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        ew = getattr(data, "edge_weight", None)

        if pd_list is None:
            pid_idx = data.pid_idx.detach().cpu().tolist()
            diags = [torch.zeros((0,3), device=x.device) for _ in pid_idx]
        else:
            diags = pd_list

        Pi, _ = self.pi(diags, return_weights=True) # [B,1,H,W]
        toks  = self.tokens(Pi) # [B,T,dtok]

        # Pre-pool GCN
        x = self.act(self.conv1(x, edge_index, edge_weight=ew))
        x = self.drop(self.act(self.conv2(x, edge_index, edge_weight=ew)))

        # Token-guided pooling
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, edge_attr=None, batch=batch, topo_tokens=toks)

        # Readout
        pooled = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        h = F.relu(self.lin1(pooled))
        out = self.lin2(h)
        return out

@dataclass
class LaiPoolConfig:
    epochs: int = 1000
    patience: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.0
    poly_power: float = 0.9
    warm_start: bool = True
    ckpt_prefix: str = "topopool_sapi_tokens"

def _cpu_state_dict(m): return {k: v.detach().cpu() for k, v in m.state_dict().items()}

def _save_ckpt_cpu(path: Path, m: nn.Module, extra: Optional[dict] = None):
    payload = {"model": _cpu_state_dict(m)}
    if extra: payload.update(extra)
    torch.save(payload, path.as_posix())

def _load_ckpt_cpu(path: Path):
    ck = torch.load(path.as_posix(), map_location="cpu")
    return ck

def _metrics_binary(y_true: np.ndarray, y_prob: np.ndarray):
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred) if y_true.size else float("nan")
    try:
        auc = roc_auc_score(y_true, y_prob) if y_true.size else float("nan")
    except Exception:
        auc = float("nan")
    if y_true.size:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
    else:
        spec = sens = float("nan")
    return acc, auc, spec, sens

@torch.no_grad()
def evaluate_topopool_tokens(model: nn.Module, loader: DataLoader, *, pd_cache: List[torch.Tensor], device) -> dict:
    model.eval()
    probs_all, ys_all = [], []
    for batch in loader:
        batch = batch.to(device)
        pid_idx = batch.pid_idx.detach().cpu().tolist()
        pd_list = [pd_cache[int(i)].to(device) if int(i) >= 0 else torch.zeros((0,3), device=device) for i in pid_idx]
        logits  = model(batch, pd_list=pd_list)
        probs = F.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        probs_all.append(probs)
        ys_all.append(batch.y.detach().cpu().numpy())
    y_prob = np.concatenate(probs_all) if probs_all else np.array([])
    y_true = np.concatenate(ys_all) if ys_all else np.array([])
    acc, auc, spec, sens = _metrics_binary(y_true, y_prob)
    return dict(acc=acc, auc=auc, spec=spec, sens=sens, n=len(y_true))

def train_topopool_tokens(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    pd_cache: List[torch.Tensor],
    models_dir: Path,
    cfg: LaiPoolConfig,
    device,
) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    BEST = models_dir / f"{cfg.ckpt_prefix}_best.pt"
    LAST = models_dir / f"{cfg.ckpt_prefix}_last.pt"

    # Warm start
    if cfg.warm_start and BEST.exists():
        try:
            state = _load_ckpt_cpu(BEST)
            sd = state["model"] if isinstance(state, dict) and "model" in state else state
            model.load_state_dict(sd); model.to(device)
            print(f"[LAI-Pool] Warm start: loaded {BEST.name}")
        except Exception as e:
            print(f"[LAI-Pool] Warm start failed: {e}")

    opt   = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit  = nn.CrossEntropyLoss()

    def _poly_lambda(epoch, total=cfg.epochs, power=cfg.poly_power):
        return max(0.0, (1.0 - epoch/float(total))**power)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: _poly_lambda(e))

    best_score, bad = -1e9, 0

    def _epoch(loader, train: bool):
        model.train() if train else model.eval()
        losses, all_y, all_p = [], [], []
        for batch in loader:
            batch = batch.to(device)
            if train: opt.zero_grad()
            pid_idx = batch.pid_idx.detach().cpu().tolist()
            pd_list = [pd_cache[int(i)].to(device) if int(i) >= 0 else torch.zeros((0,3), device=device) for i in pid_idx]
            logits  = model(batch, pd_list=pd_list)
            loss    = crit(logits, batch.y)
            if train:
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))
            all_y.append(batch.y.detach().cpu().numpy())
            all_p.append(F.softmax(logits, dim=1)[:,1].detach().cpu().numpy())
        y_true = np.concatenate(all_y) if all_y else np.array([])
        y_prob = np.concatenate(all_p) if all_p else np.array([])
        acc, auc, spec, sens = _metrics_binary(y_true, y_prob)
        return float(np.mean(losses) if losses else 0.0), dict(acc=acc, auc=auc, spec=spec, sens=sens)

    history = []
    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr = _epoch(train_loader, True)
        va_loss, va = _epoch(val_loader, False) 
        sched.step()

        # save LAST every epoch
        _save_ckpt_cpu(LAST, model, extra={"hparams": dict(hidden=64, dropout=0.0)})

        score = va["auc"]  # optimize the AUC
        improved = (score > best_score + 1e-6) or (ep == 1)
        if improved:
            best_score, bad = score, 0
            _save_ckpt_cpu(BEST, model, extra={"hparams": dict(hidden=64, dropout=0.0)})
            print(f"[LAI-Pool] epoch {ep:03d} NEW BEST: val AUC={va['auc']:.3f}: {BEST.name}")
        else:
            bad += 1

        history.append(dict(epoch=ep, tr_loss=tr_loss, tr=tr, va=va, best=best_score))
        if bad >= cfg.patience:
            print("[LAI-Pool] Early stopping."); break

    return {
        "best": BEST, "last": LAST,
        "best_val_auc": best_score,
        "history": history,
    }