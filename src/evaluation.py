from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F

def infer_dims_from_loaders(
    loaders: Dict[str, Any]
) -> Tuple[int, int]:
    for key in ("val", "test", "train"):
        ld = loaders.get(key)
        if ld is None:
            continue
        for batch in ld:
            in_dim = int(batch.x.size(1))
            classes = sorted(torch.unique(batch.y).cpu().tolist())
            n_classes = max(classes) + 1 if classes and classes[0] == 0 else len(classes)
            return in_dim, n_classes
    raise RuntimeError("Could not infer dims: all loaders are empty.")

def _load_ckpt_cpu(path: Path):
    ck = torch.load(path.as_posix(), map_location="cpu")
    return ck

@torch.no_grad()
def _probs_over_loader(m, loader, device, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    m.eval()
    probs_all, ys_all = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = m(batch)
        p = F.softmax(logits, dim=1).detach().cpu().numpy()
        probs_all.append(p)
        ys_all.append(batch.y.detach().cpu().numpy())
    probs = np.concatenate(probs_all) if probs_all else np.empty((0, n_classes))
    y_true = np.concatenate(ys_all) if ys_all else np.array([])
    return y_true.astype(int), probs

def run_topopool_full_eval(
    *,
    model_class: type,                     
    best_ckpt: Path,                     
    val_loader,                        
    test_loader,                        
    evaluate_and_save,                   
    device: torch.device,
    results_dir: Optional[Path] = None,   
    train_loader=None,                   
    default_hparams: Optional[Dict[str, Any]] = None, 
) -> Dict[str, Any]:

    if results_dir is None:
        results_dir = Path("..") / "Results"
    eval_dir = results_dir / "Evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not best_ckpt.exists():
        raise FileNotFoundError(f"BEST checkpoint not found at: {best_ckpt}")

    # Infer dims
    loaders_for_dims = {"val": val_loader, "test": test_loader}
    if train_loader is not None:
        loaders_for_dims["train"] = train_loader
    IN_DIM, N_CLASSES = infer_dims_from_loaders(loaders_for_dims)

    # Load ckpt (support {'model': state_dict, 'hparams': {...}} or raw state_dict)
    _ck = _load_ckpt_cpu(best_ckpt)
    hparams = (_ck.get("hparams") if isinstance(_ck, dict) else None) or {}
    if default_hparams is None:
        default_hparams = dict(
            hidden=64,
            dropout=0.0,
            pi_res=(32, 32),
            sigma_pi=2.0,
            decay="gauss",     
            tok_grid=(8, 8),
            dtok=64,
            datt=64,
            pool_ratio=0.5,
        )

    model = model_class(
        in_dim=IN_DIM,
        n_classes=N_CLASSES,
        hidden=hparams.get("hidden", default_hparams["hidden"]),
        dropout=hparams.get("dropout", default_hparams["dropout"]),
        pi_res=hparams.get("pi_res", default_hparams["pi_res"]),
        sigma_pi=hparams.get("sigma_pi", default_hparams["sigma_pi"]),
        decay=hparams.get("decay", default_hparams["decay"]),
        tok_grid=hparams.get("tok_grid", default_hparams["tok_grid"]),
        dtok=hparams.get("dtok", default_hparams["dtok"]),
        datt=hparams.get("datt", default_hparams["datt"]),
        pool_ratio=hparams.get("pool_ratio", default_hparams["pool_ratio"]),
    ).to(device)

    state = _ck["model"] if isinstance(_ck, dict) and "model" in _ck else _ck
    model.load_state_dict(state, strict=True)
    model.to(device)

    y_val_all, P_val = _probs_over_loader(model, val_loader, device, N_CLASSES)
    y_tst_all, P_tst = _probs_over_loader(model, test_loader, device, N_CLASSES)
    if y_val_all.size == 0 or y_tst_all.size == 0:
        raise RuntimeError("Empty VAL/TEST predictions; check your loaders and model outputs.")

    # Tag
    model_tag_base = f"TopoPool+SAPI_tokens__{best_ckpt.stem}"
    snapshots: List[Tuple[str, Any]] = []

    # Write eval artifacts
    if N_CLASSES == 2:
        pva = P_val[:, 1].ravel()
        pte = P_tst[:, 1].ravel()
        df = evaluate_and_save(
            model_tag_base,
            val_true=y_val_all, val_prob=pva,
            test_true=y_tst_all, test_prob=pte,
            out_dir=eval_dir,
            base_threshold=0.5,
            n_boot=2000,
            ece_bins=15,
            ece_strategy="uniform",
            seed=14,
        )
        snap_csv = eval_dir / f"{model_tag_base}__metrics_snapshot.csv"
        df.to_csv(snap_csv, index=False)
        snapshots.append((model_tag_base, df))
    else:
        # One-vs-rest per class k
        for k in range(N_CLASSES):
            y_val = (y_val_all == k).astype(int)
            y_tst = (y_tst_all == k).astype(int)
            pva_k = P_val[:, k].ravel()
            pte_k = P_tst[:, k].ravel()
            tag_k = f"{model_tag_base}__class{k}"
            df_k = evaluate_and_save(
                tag_k,
                val_true=y_val, val_prob=pva_k,
                test_true=y_tst, test_prob=pte_k,
                out_dir=eval_dir,
                base_threshold=0.5,
                n_boot=2000,
                ece_bins=15,
                ece_strategy="uniform",
                seed=14,
            )
            snap_csv = eval_dir / f"{tag_k}__metrics_snapshot.csv"
            df_k.to_csv(snap_csv, index=False)
            snapshots.append((tag_k, df_k))

    artifacts = []
    for tag, _df in snapshots:
        for split in ("val@0.5", "val@J", "test@J"):
            npz = eval_dir / f"{tag}__{split}__roc.npz"
            artifacts.append(npz)

    return {
        "in_dim": IN_DIM,
        "n_classes": N_CLASSES,
        "snapshots": snapshots,
        "artifacts": artifacts,
        "eval_dir": eval_dir,
        "model_tag_base": model_tag_base,
    }