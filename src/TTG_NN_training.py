from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Optional

import math
import random
import pickle
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from data_loader import Paths
from utils import evaluate_and_save, nice

@dataclass
class TTGConfig:
    # split / training
    use_balanced_split: bool = True
    batch_size: int = 32
    iters_per_epoch: int = 50
    epochs: int = 1000
    patience: int = 20
    # model / PI
    num_layers: int = 3
    num_mlp_layers: int = 2
    hidden_dim: int = 64
    final_dropout: float = 0.5
    tensor_layer_type: str = "TCL"  # "TCL" or "TRL"
    node_pooling: bool = True
    sublevel_filtration_methods: Tuple[str, ...] = ("degree","betweenness","eigenvector","closeness")
    PI_dim: int = 50
    # opt
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # misc
    seed: int = 14
    warm_start_if_exists: bool = False 
    force_train: bool = False   
    # repo path 
    ttg_repo_dir: Optional[Path] = None  

def _import_ttg_modules(ttg_repo_dir: Path):
    import sys
    ttg_repo_dir = ttg_repo_dir.resolve()
    if str(ttg_repo_dir) not in sys.path:
        sys.path.append(str(ttg_repo_dir))
        
    from cnn import CNN
    from mlp import MLP, MLP_output
    from diagram import sum_diag_from_point_cloud
    try:
        from tltorch import TRL, TCL 
        _has_tltorch = True
    except Exception:
        _has_tltorch = False

    if _has_tltorch:
        from tensorgcn import TenGCN
        return SimpleNamespace(CNN=CNN, MLP=MLP, MLP_output=MLP_output,
                               sum_diag_from_point_cloud=sum_diag_from_point_cloud,
                               TenGCN=TenGCN, has_tltorch=True)
    else:
        class TuckerLayer(nn.Module):
            def __init__(self, in_shape, out_shape):
                super().__init__()
                I1,I2,I3 = in_shape; O1,O2,O3 = out_shape
                self.U1 = nn.Parameter(torch.empty(I1, O1))
                self.U2 = nn.Parameter(torch.empty(I2, O2))
                self.U3 = nn.Parameter(torch.empty(I3, O3))
                for U in (self.U1, self.U2, self.U3):
                    nn.init.xavier_uniform_(U)
            def forward(self, X):
                Y = torch.einsum('...abc,aA->...Abc', X, self.U1)
                Y = torch.einsum('...Abc,bB->...ABc', Y, self.U2)
                Y = torch.einsum('...ABc,cC->...ABC', Y, self.U3)
                return F.relu(Y).contiguous()

        try:
            from torch_geometric.nn.pool.topk_pool import topk, filter_adj
        except Exception:
            from torch_geometric.utils import subgraph as _pyg_subgraph

            def topk(scores: torch.Tensor, ratio, batch: torch.Tensor) -> torch.Tensor:
                assert scores.dim() == 1
                out_idx = []
                for g in torch.unique(batch):
                    mask = (batch == g)
                    idx = torch.nonzero(mask, as_tuple=True)[0]
                    s = scores[idx]
                    k = (max(1, int(math.ceil(ratio * idx.numel()))) if isinstance(ratio, float)
                         else min(max(1, int(ratio)), idx.numel()))
                    _, perm = torch.topk(s, k, largest=True, sorted=False)
                    out_idx.append(idx[perm])
                return torch.cat(out_idx, dim=0)

            def filter_adj(edge_index, edge_attr, perm, num_nodes: int):
                ei, ea = _pyg_subgraph(perm, edge_index, edge_attr=edge_attr,
                                       relabel_nodes=True, num_nodes=num_nodes)
                return ei, ea

        from torch_geometric.nn import GCNConv

        class TenGCN(nn.Module):
            def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim,
                         final_dropout, tensor_layer_type, node_pooling, PI_dim,
                         sublevel_filtration_methods, device):
                super().__init__()
                self.device = device
                self.num_layers = num_layers
                self.num_neighbors = 5
                self.hidden_dim = hidden_dim
                self.score_node_layer = GCNConv(input_dim, self.num_neighbors * 2)
                self.node_pooling = node_pooling

                self.GCNs = torch.nn.ModuleList()
                self.mlps = torch.nn.ModuleList()
                for l in range(self.num_layers - 1):
                    self.GCNs.append(GCNConv(input_dim if l==0 else hidden_dim**2, hidden_dim))
                    self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim**2))

                self.GCN_tensor_layer = TuckerLayer((self.num_layers-1, hidden_dim, hidden_dim),
                                                    (hidden_dim, hidden_dim, hidden_dim))

                self.cnn = CNN(len(sublevel_filtration_methods), hidden_dim, kernel_size=2, stride=2)
                cnn_out = self.cnn.cnn_output_dim(PI_dim)
                self.PI_tensor_layer = TuckerLayer((hidden_dim, cnn_out, cnn_out),
                                                   (hidden_dim, hidden_dim, hidden_dim))

                self.attend = nn.Linear(2*hidden_dim, 1)
                self.output_tensor_layer = TuckerLayer((2*hidden_dim, hidden_dim, hidden_dim),
                                                       (2*hidden_dim, hidden_dim, hidden_dim))
                self.output = MLP_output(hidden_dim, output_dim, final_dropout)

            def compute_batch_feat(self, batch_graph):
                edge_attr = None
                edge_mat_list, pooled_x, sizes = [], [], []
                start_idx = [0]
                for i, g in enumerate(batch_graph):
                    x = g.x.to(self.device); edge_index = g.edge_index.to(self.device)
                    if self.node_pooling:
                        node_emb = self.score_node_layer(x, edge_index)
                        pc = node_emb.view(-1, self.num_neighbors, 2)
                        scores = torch.tensor([sum_diag_from_point_cloud(pc[j].detach().cpu().numpy())
                                               for j in range(pc.size(0))], dtype=torch.float32, device=self.device)
                        batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                        keep = topk(scores, 0.5, batch_vec)
                        x = x[keep]
                        edge_index, _ = filter_adj(edge_index, edge_attr, keep, num_nodes=g.x.size(0))
                    start_idx.append(start_idx[i] + x.size(0))
                    edge_mat_list.append(edge_index + start_idx[i])
                    pooled_x.append(x); sizes.append(x.size(0))
                return torch.cat(edge_mat_list, 1).to(self.device), torch.cat(pooled_x, 0).to(self.device), sizes

            def GCN_layer(self, h, edge_index, l):
                return self.mlps[l](self.GCNs[l](h, edge_index))

            def forward(self, batch_graph, batch_PI):
                edge_index, X, sizes = self.compute_batch_feat(batch_graph)
                PI_emb = self.cnn(batch_PI.to(self.device))
                PI_hidden = self.PI_tensor_layer(PI_emb)

                hidden_rep = []
                h = X
                for l in range(self.num_layers-1):
                    h = self.GCN_layer(h, edge_index, l)
                    hidden_rep.append(h)
                hidden_rep = torch.stack(hidden_rep).transpose(0,1)  # [N, L-1, h^2]

                B = len(sizes)
                graph_tensor = torch.zeros(B, 2*self.hidden_dim, self.hidden_dim, self.hidden_dim, device=self.device)
                chunks = torch.split(hidden_rep, sizes, dim=0)
                for i in range(B):
                    nodes = chunks[i].view(-1, self.num_layers-1, self.hidden_dim, self.hidden_dim)
                    g_tensor = self.GCN_tensor_layer(nodes).mean(dim=0)    # (h,h,h)
                    pi_tensor = PI_hidden[i]
                    graph_tensor[i] = torch.cat([g_tensor, pi_tensor], dim=0)

                graph_tensor = self.output_tensor_layer(graph_tensor).transpose(1,3)  # (B,h,h,2h)
                attn_map = self.attend(graph_tensor).squeeze()
                return self.output(attn_map)

        return SimpleNamespace(CNN=CNN, MLP=MLP, MLP_output=MLP_output,
                               sum_diag_from_point_cloud=sum_diag_from_point_cloud,
                               TenGCN=TenGCN, has_tltorch=False)

def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _safe_weight(d) -> float:
    w = d.get("weight", 1.0)
    try:
        if w is None: return 1.0
        if isinstance(w, float) and math.isnan(w): return 1.0
        return float(w)
    except Exception:
        return 1.0

def _build_vocab(ego_graphs: Dict[Any, nx.Graph]) -> Tuple[List[str], Dict[str,int]]:
    seen = set()
    for G in ego_graphs.values():
        for _, nd in G.nodes(data=True):
            seen.add(nd.get("type") or "unknown")
    seen.add("unknown")
    types = sorted(seen)
    return types, {t:i for i,t in enumerate(types)}

def _to_pyg(pid: int, G: nx.Graph, y: int, types: List[str], T2I: Dict[str,int]) -> Data:
    idx = {n:i for i,n in enumerate(G.nodes())}; n = len(idx)
    deg  = np.zeros(n, dtype=np.float32)
    wdeg = np.zeros(n, dtype=np.float32)
    type_ = np.zeros((n, len(types)), dtype=np.float32)
    for u in G.nodes():
        ui = idx[u]
        deg[ui] = G.degree(u)
        wsum = 0.0
        for _, v, d in G.edges(u, data=True):
            wsum += _safe_weight(d)
        wdeg[ui] = wsum
        t = (G.nodes[u].get("type") or "unknown")
        type_[ui, T2I.get(t, T2I["unknown"])] = 1.0
    X = np.concatenate([deg[:,None], wdeg[:,None], type_], axis=1)
    x = torch.tensor(X, dtype=torch.float32)

    rows, cols = [], []
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        rows += [i, j]; cols += [j, i]
    edge_index = (torch.tensor([rows, cols], dtype=torch.long)
                  if rows else torch.arange(n, dtype=torch.long).repeat(2, 1))

    data = Data(x=x, edge_index=edge_index, y=torch.tensor(int(y), dtype=torch.long))
    data.pid = int(pid)
    return data

def graphs_by_pid(ego_graphs: Dict[Any, nx.Graph], flags_df) -> Tuple[Dict[int, Data], List[str]]:
    types, T2I = _build_vocab(ego_graphs)
    out = {}
    for pid, G in ego_graphs.items():
        if pid in flags_df.index:
            y = int(flags_df.loc[pid, "hospital_expire_flag"])
            out[int(pid)] = _to_pyg(int(pid), G, y, types, T2I)
    return out, types

def graphs_from_split(
    split_std: Dict[str, Any],
    graph_by_pid: Dict[int, Data],
    use_balanced: bool = True,
) -> Tuple[List[Data], List[Data], List[Data]]:
    if use_balanced:
        tr = list(map(int, split_std["train_pids_balanced"]))
        va = list(map(int, split_std["val_pids_balanced"]))
    else:
        tr = list(map(int, split_std["train_pids"]))
        va = list(map(int, split_std["val_pids"]))
    te = list(map(int, split_std["test_pids"]))
    have = set(graph_by_pid.keys())
    train = [graph_by_pid[p] for p in tr if p in have]
    val   = [graph_by_pid[p] for p in va if p in have]
    test  = [graph_by_pid[p] for p in te if p in have]
    print(f"split_std (TTG), train={len(train)}  val={len(val)}  test={len(test)}, balanced={use_balanced}")
    if not train or not val or not test:
        raise RuntimeError("Empty split after filtering by data availability.")
    return train, val, test

def _PI_cache_path(proc_dir: Path, prefix: str, dim: int, methods: Tuple[str,...]) -> Path:
    return proc_dir / f"PI_{prefix}_d{dim}_{'-'.join(methods)}.pt"

def compute_PI_tensor(
    graphs: List[Data],
    PI_dim: int,
    methods: Tuple[str,...],
    show_progress: bool = True,
) -> torch.Tensor:
    B, C = len(graphs), len(methods)
    out = []
    pbar = tqdm(total=B*C, desc=f"PI {PI_dim}×{PI_dim}", disable=not show_progress)
    import gudhi as gd
    def _sublevel_pd(A: np.ndarray, method: str, max_scale: float = 50.0) -> np.ndarray:
        assert method in ['degree','betweenness','communicability','eigenvector','closeness']
        G = nx.from_numpy_array(A)
        if method == 'degree':
            feat = np.asarray(A.sum(axis=1)).ravel()
        elif method == 'betweenness':
            feat = np.fromiter(nx.betweenness_centrality(G).values(), dtype=float)
        elif method == 'communicability':
            feat = np.fromiter(nx.communicability_betweenness_centrality(G).values(), dtype=float)
        elif method == 'eigenvector':
            feat = np.fromiter(nx.eigenvector_centrality(G, max_iter=10000).values(), dtype=float)
        else:
            feat = np.fromiter(nx.closeness_centrality(G).values(), dtype=float)

        stb = gd.SimplexTree()
        xs, ys = np.where(np.triu(A) > 0)
        n = A.shape[0]
        for j in range(n): stb.insert([int(j)], filtration=-1e10)
        for u, v in zip(xs, ys): stb.insert([int(u), int(v)], filtration=-1e10)
        for j in range(n): stb.assign_filtration([int(j)], float(feat[j]))
        stb.make_filtration_non_decreasing()

        pd = []
        for _, (b, d) in stb.persistence():
            pd.append((float(b), float(max_scale) if np.isinf(d) else float(d)))
        return np.asarray(pd, dtype=float)

    def _persistence_image(dgm: np.ndarray, resolution=(50, 50), normalization=True, bandwidth=1.0, power=1.0) -> np.ndarray:
        dgm = np.asarray(dgm, dtype=float)
        if dgm.size == 0: return np.zeros(resolution, dtype=float)
        PXs, PYs = dgm[:,0], dgm[:,1]
        xm,xM,ym,yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
        if xM==xm: xM = xm + 1e-6
        if yM==ym: yM = ym + 1e-6
        x = np.linspace(xm, xM, resolution[0]); y = np.linspace(ym, yM, resolution[1])
        X, Y = np.meshgrid(x, y); X, Y = X[...,None], Y[...,None]
        P0, P1 = dgm[:,0][None,None,:], dgm[:,1][None,None,:]
        weight = np.abs(P1 - P0)
        distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)
        Z = (weight ** power) * np.exp(-distpts ** 2 / bandwidth)
        outI = Z.sum(axis=2)
        if normalization:
            M,m = np.max(outI), np.min(outI)
            if M != m: outI = (outI - m) / (M - m)
        return outI

    for i, g in enumerate(graphs):
        Gx = to_networkx(g, to_undirected=True)
        A = nx.to_numpy_array(Gx)
        chans = []
        for m in methods:
            pd = _sublevel_pd(A, m)
            pi = torch.tensor(_persistence_image(pd, resolution=(PI_dim, PI_dim)), dtype=torch.float32)
            chans.append(pi)
            if show_progress:
                pbar.update(1); pbar.set_postfix(graph=f"{i+1}/{B}", method=m, nodes=A.shape[0])
        out.append(torch.stack(chans, dim=0))
    if show_progress: pbar.close()
    return torch.stack(out, dim=0)  # [B, C, H, W]

def ensure_PI_cache(
    graphs: List[Data],
    prefix: str,
    cfg: TTGConfig,
    proc_dir: Path,
    device: torch.device,
) -> List[torch.Tensor]:
    methods = tuple(cfg.sublevel_filtration_methods)
    cache = _PI_cache_path(proc_dir, prefix, cfg.PI_dim, methods)
    if cache.exists():
        PIs = torch.load(cache.as_posix(), map_location="cpu").to(device)
        print(f"Loaded {prefix} PI cache to {cache.name}  ({tuple(PIs.shape)})")
    else:
        print(f"Computing {prefix} PI tensor ({len(graphs)} graphs)…")
        PIs = compute_PI_tensor(graphs, cfg.PI_dim, methods, show_progress=True).to(device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(PIs.detach().cpu(), cache.as_posix())
        print(f"Saved {prefix} PI to {cache.name}")
    return [PIs[i] for i in range(PIs.size(0))]

_criterion = nn.CrossEntropyLoss()

def _train_one_epoch(args: SimpleNamespace, model, device, train_graphs, train_PIs, optimizer):
    model.train()
    loss_accum = 0.0
    for _ in range(args.iters_per_epoch):
        idx = np.random.permutation(len(train_graphs))[:args.batch_size]
        batch_graph = [train_graphs[i] for i in idx]
        batch_PI    = torch.stack([train_PIs[i] for i in idx]).to(device)  # [B, C, H, W]
        logits = model(batch_graph, batch_PI)
        labels = torch.tensor([int(g.y) for g in batch_graph], dtype=torch.long, device=device)
        loss = _criterion(logits, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_accum += float(loss.detach().cpu())
    return loss_accum / max(1, args.iters_per_epoch)

@torch.no_grad()
def _logits_batches(model, graphs, PIs, device, mb: int = 64):
    model.eval()
    outs = []
    for i in range(0, len(graphs), mb):
        bgraphs = graphs[i:i+mb]
        if not bgraphs: continue
        bPIs = torch.stack(PIs[i:i+mb]).to(device)
        outs.append(model(bgraphs, bPIs).detach())
    return torch.cat(outs, dim=0) if outs else torch.empty(0)

@torch.no_grad()
def _acc_of(model, device, graphs, PIs) -> float:
    out = _logits_batches(model, graphs, PIs, device)
    if out.numel() == 0: return 0.0
    pred = out.argmax(dim=1).cpu().numpy()
    y = np.array([int(g.y) for g in graphs], dtype=int)
    return float((pred == y).mean())

@torch.no_grad()
def _ytrue_yprob(model, graphs, PIs, device) -> Tuple[np.ndarray, np.ndarray]:
    out = _logits_batches(model, graphs, PIs, device)
    if out.numel() == 0: return np.array([]), np.array([])
    prob1 = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
    y = np.array([int(g.y) for g in graphs], dtype=int)
    return y, prob1

def _metrics(y_true, y_prob, thr=0.5):
    if y_true.size == 0: return (float("nan"),)*4
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        auc = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    spec = tn/(tn+fp) if (tn+fp) else np.nan
    sens = tp/(tp+fn) if (tp+fn) else np.nan
    return acc, auc, spec, sens

def run_ttg_pipeline(
    *,
    paths: Paths,
    ego_graphs: Dict[Any, nx.Graph],
    flags_df,
    split_std: Dict[str, Any],
    cfg: TTGConfig = TTGConfig(),
) -> Dict[str, Any]:
    assert cfg.ttg_repo_dir is not None and Path(cfg.ttg_repo_dir).exists(), \
        "cfg.ttg_repo_dir must point to the folder containing TTG-NN files."
    mod = _import_ttg_modules(Path(cfg.ttg_repo_dir))

    _seed_everything(cfg.seed)
    device = _device()

    g_by_pid, type_list = graphs_by_pid(ego_graphs, flags_df)
    train_graphs, val_graphs, test_graphs = graphs_from_split(
        split_std, g_by_pid, use_balanced=cfg.use_balanced_split
    )

    PROC_DIR = paths.DATA_DIR / "Processed"
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train_PIs = ensure_PI_cache(train_graphs, "train", cfg, PROC_DIR, device)
    val_PIs   = ensure_PI_cache(val_graphs,   "val",   cfg, PROC_DIR, device)
    test_PIs  = ensure_PI_cache(test_graphs,  "test",  cfg, PROC_DIR, device)

    in_dim = int(train_graphs[0].x.size(1))
    num_classes = int(max(int(g.y) for g in (train_graphs + val_graphs + test_graphs)) + 1)

    args = SimpleNamespace(
        batch_size=cfg.batch_size,
        iters_per_epoch=cfg.iters_per_epoch,
        epochs=cfg.epochs,
        patience=cfg.patience,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        PI_dim=cfg.PI_dim,
        num_layers=cfg.num_layers,
        num_mlp_layers=cfg.num_mlp_layers,
        hidden_dim=cfg.hidden_dim,
        final_dropout=cfg.final_dropout,
        tensor_layer_type=cfg.tensor_layer_type,
        node_pooling=cfg.node_pooling,
        sublevel_filtration_methods=tuple(cfg.sublevel_filtration_methods),
    )

    prefix = f"ttg_{cfg.tensor_layer_type.lower()}_pi{cfg.PI_dim}_L{cfg.num_layers}_H{cfg.hidden_dim}_seed{cfg.seed}"
    MODELS_DIR = paths.MODELS_DIR
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TTG_BEST = MODELS_DIR / f"{prefix}_best.pt"
    TTG_LAST = MODELS_DIR / f"{prefix}_last.pt"
    TTG_META = MODELS_DIR / f"{prefix}_meta.pkl"

    def _cpu_state_dict(m: nn.Module):
        return {k: v.detach().cpu() for k, v in m.state_dict().items()}
    def _save_ckpt(path: Path, model, best_val=None):
        torch.save(
            {"model": _cpu_state_dict(model), "in_dim": in_dim, "n_classes": num_classes, "best_val": best_val},
            path.as_posix()
        )
    def _load_ckpt(path: Path):
        return torch.load(path.as_posix(), map_location="cpu")

    #  build model
    def _make_model(in_dim_):
        return mod.TenGCN(
            args.num_layers, args.num_mlp_layers, in_dim_, args.hidden_dim, num_classes,
            args.final_dropout, args.tensor_layer_type, args.node_pooling, args.PI_dim,
            list(args.sublevel_filtration_methods), device
        ).to(device)

    ck_loaded = False
    model: nn.Module

    if (not cfg.force_train) and TTG_BEST.exists():
        try:
            ck = _load_ckpt(TTG_BEST)
            model = _make_model(int(ck.get("in_dim", in_dim) or in_dim))
            model.load_state_dict(ck["model"])
            model.eval()
            ck_loaded = True
            print(f"Loaded TTG best at {TTG_BEST.name} (best_val={ck.get('best_val')})")
        except Exception as e:
            print(f"Could not load BEST checkpoint: {e} so training from scratch.")
            model = _make_model(in_dim)
    else:
        model = _make_model(in_dim)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Train with early-stopping on VAL acc
    best_val, bad, best_state = 0.0, 0, None
    if not ck_loaded or cfg.warm_start_if_exists or cfg.force_train:
        for ep in range(1, args.epochs + 1):
            tr_loss = _train_one_epoch(args, model, device, train_graphs, train_PIs, optimizer)
            val_acc = _acc_of(model, device, val_graphs, val_PIs)

            _save_ckpt(TTG_LAST, model, best_val=best_val)

            if val_acc > best_val + 1e-6:
                best_val, bad = val_acc, 0
                _save_ckpt(TTG_BEST, model, best_val=best_val)
                best_state = {k: v.clone() for k, v in _cpu_state_dict(model).items()}
                print(f"[TTG] ep {ep:03d} NEW BEST val_acc={val_acc:.3f} to {TTG_BEST.name}")
            else:
                bad += 1

            print(f"[TTG {ep:03d}] train_loss={tr_loss:.4f}, val_acc={val_acc:.3f}, best={best_val:.3f} (pat {bad}/{args.patience})")
            if bad >= args.patience:
                print("Early stopping on VAL.")
                break
    else:
        print("Skipping training (best checkpoint already loaded and warm_start_if_exists=False).")

    # Restore best for TEST
    if isinstance(best_state, dict) and len(best_state):
        model.load_state_dict(best_state); model.to(device)
    elif TTG_BEST.exists():
        ck = _load_ckpt(TTG_BEST)
        model.load_state_dict(ck["model"]); model.to(device)
        print(f"Loaded BEST from {TTG_BEST.name}")

    # TEST metrics
    y_true, y_prob = _ytrue_yprob(model, test_graphs, test_PIs, device)
    acc, auc, spec, sens = _metrics(y_true, y_prob, thr=0.5)
    print(f"TTG ({cfg.tensor_layer_type.upper()}) · Acc={acc:.3f}  AUC={auc:.3f}  Spec={spec:.3f}  Sens={sens:.3f}")

    # Persist meta
    meta = dict(prefix=prefix, best=TTG_BEST.as_posix(), last=TTG_LAST.as_posix(),
                in_dim=in_dim, n_classes=num_classes, best_val=best_val,
                split_balanced=cfg.use_balanced_split, seed=cfg.seed,
                hparams=dict(lr=cfg.lr, weight_decay=cfg.weight_decay, patience=cfg.patience,
                             batch_size=cfg.batch_size, iters_per_epoch=cfg.iters_per_epoch,
                             PI_dim=cfg.PI_dim, num_layers=cfg.num_layers, hidden_dim=cfg.hidden_dim,
                             tensor_layer=cfg.tensor_layer_type))
    with open(TTG_META, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved TTG meta to {TTG_META.name}")

    eval_dir = paths.RESULTS_DIR / "Evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    y_val, p_val = _ytrue_yprob(model, val_graphs,  val_PIs,  device)
    y_tst, p_tst = _ytrue_yprob(model, test_graphs, test_PIs, device)
    if y_val.size == 0 or y_tst.size == 0:
        raise RuntimeError("Empty VAL/TEST predictions for TTG and check data and model outputs.")

    model_tag = f"TTG_{cfg.tensor_layer_type.upper()}__{Path(TTG_BEST).stem}"
    df = evaluate_and_save(
        model_tag,
        val_true=y_val,  val_prob=p_val,
        test_true=y_tst, test_prob=p_tst,
        out_dir=eval_dir,
        base_threshold=0.5,
        n_boot=2000,
        ece_bins=15,
        ece_strategy="uniform",
        seed=cfg.seed,
    )
    snap = eval_dir / f"{model_tag}__metrics_snapshot.csv"
    df.to_csv(snap, index=False)
    print(f"Wrote snapshot to {nice(snap)}")

    return {
        "type_list": type_list,
        "graphs": dict(train=train_graphs, val=val_graphs, test=test_graphs),
        "pis": dict(train=train_PIs, val=val_PIs, test=test_PIs),
        "test_metrics": dict(acc=acc, auc=auc, specificity=spec, sensitivity=sens),
        "best_ckpt": TTG_BEST,
        "last_ckpt": TTG_LAST,
        "eval_df": df,
    }