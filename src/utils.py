# utils.py
from __future__ import annotations
from pathlib import Path
from contextlib import contextmanager
from typing import Tuple, Dict, Callable, Optional, Any, List
import json, math, pickle
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_score, recall_score, f1_score,
    brier_score_loss
)

def nice(p: Path, root: Path | str = "..") -> str:
    root = Path(root)
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return str(p)

def build_pid_universe(
    *,
    pd_std: List[Tuple[Any, Optional[int], Dict[int, np.ndarray]]],
    ego_graphs: Dict[Any, Any],
    flags_df: pd.DataFrame,
    require_vectors: bool = True,
    img_std: Optional[List[Tuple[Any, Optional[int], np.ndarray]]] = None,
    lan_std: Optional[List[Tuple[Any, Optional[int], np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    
    pd_index = {pid for pid, _y, _dims in pd_std}
    common_all = np.array(sorted(set(ego_graphs.keys()) & set(flags_df.index) & pd_index), dtype=object)
    y_all = flags_df.loc[common_all, "hospital_expire_flag"].astype(int).to_numpy()

    if not require_vectors:
        return common_all, y_all

    if img_std is None or lan_std is None:
        raise ValueError("require_vectors=True but img_std/lan_std were not provided.")
    img_pids = np.array([pid for pid, _y, _ in img_std], dtype=object)
    lan_pids = np.array([pid for pid, _y, _ in lan_std], dtype=object)
    both = np.intersect1d(img_pids, lan_pids)
    mask = np.isin(common_all, both)
    return common_all[mask], y_all[mask]


def split_80_10_10(
    pids: np.ndarray, y: np.ndarray, seed: int = 14
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    # test = 10%
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    trval_idx, test_idx = next(sss1.split(pids, y))
    p_trval, y_trval = pids[trval_idx], y[trval_idx]
    p_test,  y_test  = pids[test_idx],  y[test_idx]
    val_rel = 0.10 / 0.90
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
    tr_idx, val_idx = next(sss2.split(p_trval, y_trval))
    p_train, y_train = p_trval[tr_idx], y_trval[tr_idx]
    p_val,   y_val   = p_trval[val_idx], y_trval[val_idx]
    return (p_train, y_train), (p_val, y_val), (p_test, y_test)


def upsample_to_balance(
    pids: np.ndarray, y: np.ndarray, seed: int = 14
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pos = pids[y == 1]
    neg = pids[y == 0]
    if len(pos) == 0 or len(neg) == 0 or len(pos) == len(neg):
        return pids.copy()
    maj, mino = (pos, neg) if len(pos) > len(neg) else (neg, pos)
    dup = rng.choice(mino, size=(len(maj) - len(mino)), replace=True)
    balanced = np.concatenate([maj, mino, dup])
    rng.shuffle(balanced)
    return balanced


def compute_or_load_balanced_split(
    *,
    split_cache: Path,
    ego_flags_df: pd.DataFrame,   # needed later for sanity checks
    common_pids: np.ndarray,
    y: np.ndarray,
    seed: int = 14,
    require_vectors: bool = True,
    img_std: Optional[List[Tuple[Any, Optional[int], np.ndarray]]] = None,
    lan_std: Optional[List[Tuple[Any, Optional[int], np.ndarray]]] = None,
    attach_vector_indices: bool = False,
    img_cache: Optional[Path] = None,
    lan_cache: Optional[Path] = None,
) -> Dict[str, Any]:
    if split_cache.exists():
        split_std = pickle.load(open(split_cache, "rb"))
        if attach_vector_indices:
            if img_cache is None or lan_cache is None:
                raise ValueError("attach_vector_indices=True requires img_cache and lan_cache paths.")
            split_std = _attach_vector_indices(split_std, img_cache, lan_cache)
        return split_std

    (p_train, y_train), (p_val, y_val), (p_test, y_test) = split_80_10_10(common_pids, y, seed=seed)
    train_pids_bal = upsample_to_balance(p_train, y_train, seed=seed)
    val_pids_bal   = p_val  # no balancing

    split_std: Dict[str, Any] = {
        "seed": seed,
        "train_pids": p_train, "val_pids": p_val, "test_pids": p_test,
        "train_pids_balanced": train_pids_bal, "val_pids_balanced": val_pids_bal,
    }

    if require_vectors:
        if img_std is None or lan_std is None:
            raise ValueError("require_vectors=True but img_std/lan_std were not provided.")
        img_index = {pid: i for i, (pid, _, _) in enumerate(img_std)}
        lan_index = {pid: i for i, (pid, _, _) in enumerate(lan_std)}
        def _idxs(pids_arr):
            return (
                np.array([img_index[p] for p in pids_arr], dtype=int),
                np.array([lan_index[p] for p in pids_arr], dtype=int)
            )
        split_std["img_idx"] = {}
        split_std["lan_idx"] = {}
        for key in ("train_pids", "val_pids", "test_pids", "train_pids_balanced", "val_pids_balanced"):
            ii, ll = _idxs(split_std[key])
            split_std["img_idx"][key] = ii
            split_std["lan_idx"][key] = ll

    split_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(split_cache, "wb") as f:
        pickle.dump(split_std, f, protocol=pickle.HIGHEST_PROTOCOL)

    if attach_vector_indices:
        if img_cache is None or lan_cache is None:
            raise ValueError("attach_vector_indices=True requires img_cache and lan_cache paths.")
        split_std = _attach_vector_indices(split_std, img_cache, lan_cache)

    # sanity checks
    _ = _pos_neg_counts(ego_flags_df, split_std["train_pids_balanced"])
    _ = _pos_neg_counts(ego_flags_df, split_std["val_pids"])
    _ = _pos_neg_counts(ego_flags_df, split_std["test_pids"])
    return split_std


def _pos_neg_counts(flags_df: pd.DataFrame, pids: np.ndarray) -> Tuple[int, int]:
    yv = flags_df.loc[pids, "hospital_expire_flag"].astype(int).to_numpy()
    return int((yv == 1).sum()), int((yv == 0).sum())
    

def load_vectors(pkl_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = pickle.load(open(pkl_path, "rb"))
    if not rows:
        return np.array([], dtype=object), np.array([], dtype=int), np.empty((0, 0), dtype=float)
    pids, ys, vecs = zip(*rows)
    y = np.asarray(ys, dtype=float)
    mask = np.isin(y, [0, 1])
    pids = np.asarray(pids, dtype=object)[mask]
    y = y[mask].astype(int)
    X = (np.stack([np.asarray(v, float).ravel() for v in np.asarray(vecs, dtype=object)[mask]], axis=0)
         if len(vecs) else np.empty((0, 0), dtype=float))
    return pids, y, X


def _attach_vector_indices(split: Dict[str, Any], img_pkl: Path, lan_pkl: Path) -> Dict[str, Any]:
    img_rows = pickle.load(open(img_pkl, "rb"))
    lan_rows = pickle.load(open(lan_pkl, "rb"))
    img_index = {pid: i for i, (pid, _, _) in enumerate(img_rows)}
    lan_index = {pid: i for i, (pid, _, _) in enumerate(lan_rows)}

    img_idx, lan_idx = {}, {}
    keys = [k for k in split.keys() if k.endswith("_pids") or k.endswith("_pids_balanced")]
    for k in keys:
        pids = split[k]
        img_idx[k] = np.array([img_index[p] for p in pids], dtype=int)
        lan_idx[k] = np.array([lan_index[p] for p in pids], dtype=int)
    split["img_idx"] = img_idx
    split["lan_idx"] = lan_idx
    return split


def save_model(model, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path

@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)
    old_cb = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()

def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    z = 1.959963984540054  # ~N^{-1}(1 - alpha/2) for alpha=0.05
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(y_true, y_scores):
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, float)
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return np.nan, np.nan
    X = np.concatenate([pos, neg])
    T = _compute_midrank(X)
    Tpos = _compute_midrank(pos)
    Tneg = _compute_midrank(neg)
    auc = (T[:m].sum() - m*(m+1)/2) / (m*n)
    v01 = (T[:m] - Tpos) / n
    v10 = 1.0 - (T[m:] - Tneg) / m
    s01 = np.var(v01, ddof=1)
    s10 = np.var(v10, ddof=1)
    var_auc = s01/m + s10/n
    return auc, var_auc

def auc_with_ci(y_true, y_score, alpha: float = 0.05) -> Tuple[float, float, float]:
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        return (np.nan, np.nan, np.nan)
    auc_d, var_auc = _fast_delong(y_true, y_score)
    if not np.isfinite(var_auc):
        return (auc, np.nan, np.nan)
    se = math.sqrt(max(var_auc, 0.0))
    z = 1.959963984540054
    return (auc, max(0.0, auc - z*se), min(1.0, auc + z*se))

def ece_score(y_true, y_prob, n_bins: int = 15, strategy: str = "uniform"):
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, float).clip(1e-8, 1-1e-8)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins+1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0, 1, n_bins+1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")
    inds = np.minimum(np.digitize(p, edges) - 1, n_bins-1)

    ece = 0.0
    bins = []
    for b in range(n_bins):
        mask = inds == b
        nb = int(mask.sum())
        if nb == 0:
            bins.append(dict(lower=edges[b], upper=edges[b+1], conf=np.nan, acc=np.nan, count=0))
            continue
        conf = float(p[mask].mean())
        acc  = float(y_true[mask].mean())
        ece += (nb / len(p)) * abs(acc - conf)
        bins.append(dict(lower=edges[b], upper=edges[b+1], conf=conf, acc=acc, count=nb))
    return float(ece), bins

def youden_j(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    J = tpr - fpr
    j_idx = int(np.argmax(J))
    return dict(threshold=float(thr[j_idx]), J=float(J[j_idx]), TPR=float(tpr[j_idx]), FPR=float(fpr[j_idx]))

def evaluate_and_save(
    model_tag: str,
    *,
    val_true, val_prob,
    test_true, test_prob,
    out_dir: Path,
    base_threshold: float = 0.5,
    n_boot: int = 2000,
    ece_bins: int = 15,
    ece_strategy: str = "uniform",
    seed: int = 14
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _bootstrap_ci(func: Callable, y_true, y_prob_or_pred, n_boot=2000, seed=14, stratified=True):
        rng = np.random.RandomState(seed)
        y_true_arr = np.asarray(y_true).astype(int)
        vals = []
        idx_pos = np.where(y_true_arr == 1)[0]
        idx_neg = np.where(y_true_arr == 0)[0]
        n_pos, n_neg = len(idx_pos), len(idx_neg)
        N = len(y_true_arr)
        for _ in range(n_boot):
            if stratified and n_pos > 0 and n_neg > 0:
                samp_pos = rng.choice(idx_pos, size=n_pos, replace=True)
                samp_neg = rng.choice(idx_neg, size=n_neg, replace=True)
                idx = np.concatenate([samp_pos, samp_neg])
            else:
                idx = rng.choice(np.arange(N), size=N, replace=True)
            vals.append(func(idx))
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    def _conf_counts(y_true_arr, y_pred_arr):
        tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0,1]).ravel()
        return int(tn), int(fp), int(fn), int(tp)

    def _with_ci_proportions(y_true_arr, y_pred_arr):
        tn, fp, fn, tp = _conf_counts(y_true_arr, y_pred_arr)
        n = tn + fp + fn + tp
        acc = (tp + tn) / n if n else np.nan
        sen = tp / (tp + fn) if (tp + fn) else np.nan
        spe = tn / (tn + fp) if (tn + fp) else np.nan
        ppv = tp / (tp + fp) if (tp + fp) else np.nan
        acc_ci = _wilson_ci(tp+tn, n) if n else (np.nan, np.nan)
        sen_ci = _wilson_ci(tp, tp+fn) if (tp+fn) else (np.nan, np.nan)
        spe_ci = _wilson_ci(tn, tn+fp) if (tn+fp) else (np.nan, np.nan)
        ppv_ci = _wilson_ci(tp, tp+fp) if (tp+fp) else (np.nan, np.nan)
        return {
            "accuracy": (acc, acc_ci),
            "sensitivity": (sen, sen_ci),
            "specificity": (spe, spe_ci),
            "precision": (ppv, ppv_ci),
        }

    def _summarize_split(split_name, y_true_arr, y_prob_arr, applied_thr: float):
        y_true_arr = np.asarray(y_true_arr).astype(int)
        y_prob_arr = np.asarray(y_prob_arr, float)
        y_pred_arr = (y_prob_arr >= applied_thr).astype(int)

        prop = _with_ci_proportions(y_true_arr, y_pred_arr)
        auc, auc_lo, auc_hi = auc_with_ci(y_true_arr, y_prob_arr)

        f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
        f1_lo, f1_hi = _bootstrap_ci(lambda idx: f1_score(y_true_arr[idx], y_pred_arr[idx], zero_division=0),
                                     y_true_arr, y_pred_arr, n_boot=n_boot, seed=seed)

        brier = brier_score_loss(y_true_arr, y_prob_arr)
        b_lo, b_hi = _bootstrap_ci(lambda idx: brier_score_loss(y_true_arr[idx], y_prob_arr[idx]),
                                   y_true_arr, y_prob_arr, n_boot=n_boot, seed=seed)

        ece, bins = ece_score(y_true_arr, y_prob_arr, n_bins=ece_bins, strategy=ece_strategy)
        e_lo, e_hi = _bootstrap_ci(
            lambda idx: ece_score(y_true_arr[idx], y_prob_arr[idx], n_bins=ece_bins, strategy=ece_strategy)[0],
            y_true_arr, y_prob_arr, n_boot=n_boot, seed=seed
        )

        fpr, tpr, thr = roc_curve(y_true_arr, y_prob_arr)
        bundle = dict(
            model_tag=model_tag, split=split_name,
            y_true=y_true_arr.astype(int),
            y_prob=y_prob_arr.astype(float),
            fpr=fpr.astype(float), tpr=tpr.astype(float), thresholds=thr.astype(float),
            auc=float(auc), auc_ci=(float(auc_lo), float(auc_hi))
        )
        np.savez_compressed(out_dir / f"{model_tag}__{split_name}__roc.npz",
                            **{k: (v if isinstance(v, np.ndarray) else np.array(v)) for k, v in bundle.items()})

        def row_for(name, val, ci):
            return dict(
                model_tag=model_tag, split=split_name, metric=name,
                value=float(val),
                ci_low=(None if np.isnan(ci[0]) else float(ci[0])),
                ci_high=(None if np.isnan(ci[1]) else float(ci[1])),
                threshold=float(applied_thr)
            )

        return [
            row_for("AUC", auc, (auc_lo, auc_hi)),
            row_for("Accuracy", *prop["accuracy"]),
            row_for("Sensitivity", *prop["sensitivity"]),
            row_for("Specificity", *prop["specificity"]),
            row_for("Precision", *prop["precision"]),
            row_for("F1", f1, (f1_lo, f1_hi)),
            row_for("Brier", brier, (b_lo, b_hi)),
            row_for("ECE", ece, (e_lo, e_hi)),
        ]

    j = youden_j(val_true, val_prob)
    thr_j = float(j["threshold"])

    with open(out_dir / f"{model_tag}__youdenJ.json", "w") as f:
        json.dump(
            dict(model_tag=model_tag, threshold_J=thr_j, J=float(j["J"]),
                 val_TPR=float(j["TPR"]), val_FPR=float(j["FPR"])),
            f, indent=2
        )

    rows = []
    rows += _summarize_split("val@0.5", val_true, val_prob, base_threshold)
    rows += _summarize_split("val@J",   val_true, val_prob, thr_j)
    rows += _summarize_split("test@J",  test_true, test_prob, thr_j)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "metrics_summary.csv"
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    return df