from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from data_loader import Paths
from utils import evaluate_and_save

from .perslay import PerslayModel

@dataclass
class PerslayConfig:
    img_size: Tuple[int, int] = (32, 32)
    bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.0, 1.0), (0.0, 1.0))  # [[bmin,bmax],[dmin,dmax]]
    variance: float = 0.01
    power: float = 1.0
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1000
    batch_size: int = 256
    patience: int = 20
    cpu_only: bool = False
    force_train: bool = False

def _fit_minmax(di_list: List[np.ndarray]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if not di_list:
        return ((0.0, 1.0), (0.0, 1.0))
    births = np.concatenate([d[:, 0] for d in di_list if d.size], axis=0)
    deaths = np.concatenate([d[:, 1] for d in di_list if d.size], axis=0)
    births = births[np.isfinite(births)]
    deaths = deaths[np.isfinite(deaths)]
    if births.size == 0 or deaths.size == 0:
        return ((0.0, 1.0), (0.0, 1.0))
    return (float(births.min()), float(births.max() or 1.0)), (float(deaths.min()), float(deaths.max() or 1.0))

def _scale_di(di: np.ndarray, bnds: Tuple[Tuple[float,float], Tuple[float,float]]) -> np.ndarray:
    if di.size == 0:
        return di
    (bmin, bmax), (dmin, dmax) = bnds
    eps = 1e-12
    out = di.astype(np.float32).copy()
    out[:, 0] = (out[:, 0] - bmin) / max(bmax - bmin, eps)
    out[:, 1] = (out[:, 1] - dmin) / max(dmax - dmin, eps)
    return np.clip(out, 0.0, 1.0)

def _scale_list(di_list: List[np.ndarray], bnds) -> List[np.ndarray]:
    return [_scale_di(d, bnds) for d in di_list]

def _pack_diagrams(di_list: List[np.ndarray], max_pts: int) -> np.ndarray:
    B = len(di_list)
    out = np.zeros((B, max_pts, 3), dtype=np.float32)
    for i, d in enumerate(di_list):
        k = min(len(d), max_pts)
        if k > 0:
            out[i, :k, :2] = d[:k, :2]
            out[i, :k, 2] = 1.0
    return out

def _as_bd_stacked(third) -> np.ndarray:
    if isinstance(third, dict):
        parts = []
        for d in (0, 1):
            bd = third.get(d)
            if bd is not None and np.asarray(bd).size:
                parts.append(np.asarray(bd, dtype=np.float32))
        if parts:
            return np.vstack(parts).astype(np.float32, copy=False)
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(third)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, :2].astype(np.float32, copy=False)
    return np.zeros((0, 2), dtype=np.float32)

def _prep_sets_from_split(pd_list, split_obj):
    pd_by_pid = {int(pid): (int(y), third) for pid, y, third in pd_list}
    def _one(key):
        pids = split_obj[key].astype(int)
        ys   = np.array([pd_by_pid[p][0] for p in pids], dtype=int)
        di   = [_as_bd_stacked(pd_by_pid[p][1]) for p in pids]
        return pids, ys, di
    tr = _one("train_pids_balanced")
    va = _one("val_pids_balanced")
    te = _one("test_pids")
    return tr, va, te

def _build_perslay_model(*, max_pts: int, cfg: PerslayConfig) -> keras.Model:
    perslay_parameters = [dict(
        pweight="power",
        pweight_init=lambda shape: tf.ones(shape, dtype=tf.float32),
        pweight_train=False,
        pweight_power=cfg.power,

        layer="Image",
        layer_train=True,
        lvariance_init=lambda shape=None: tf.constant([cfg.variance], dtype=tf.float32),
        image_size=list(cfg.img_size),
        image_bnds=[list(cfg.bounds[0]), list(cfg.bounds[1])],  # [[bmin,bmax],[dmin,dmax]]
        perm_op="sum",
        keep=None,
        final_model="identity",
    )]

    rho = keras.Sequential([
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dropout(cfg.dropout),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(cfg.dropout),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(cfg.dropout),
        keras.layers.Dense(1, activation="sigmoid", dtype="float32"),
    ], name="rho_head")

    pl = PerslayModel(name="perslay", diagdim=2, perslay_parameters=perslay_parameters, rho=rho)

    # Inputs
    inp_diags = keras.Input(shape=(max_pts, 3), name="pd_masked", dtype="float32")  # [birth,death,mask]
    inp_feats = keras.Input(shape=(1,), name="dummy_feats", dtype="float32")

    out = pl([[inp_diags], inp_feats], training=True)

    model = keras.Model([inp_diags, inp_feats], out, name="PERSLAY_GITHUB")

    try:
        opt = keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    except Exception:
        try:
            opt = keras.optimizers.experimental.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
        except Exception:
            opt = keras.optimizers.Adam(learning_rate=cfg.lr)

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def _metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        auc = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    return dict(accuracy=float(acc), auc=float(auc), specificity=float(spec), sensitivity=float(sens))

def run_perslay_e2e(
    *,
    paths: Paths,
    pd_std,
    split_std,
    model_name: str = "perslay_e2e_std",
    cfg: PerslayConfig = PerslayConfig(),
):
    
    gpus = tf.config.list_physical_devices("GPU")
    if not cfg.cpu_only and gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        device = "/GPU:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        device = "/CPU:0"

    weights_prefix = paths.MODELS_DIR / model_name          
    meta_path      = paths.MODELS_DIR / f"{model_name}_meta.pkl"
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    (tr_pids, y_tr, di_tr), (va_pids, y_va, di_va), (te_pids, y_te, di_te) = _prep_sets_from_split(pd_std, split_std)

    bnds = _fit_minmax(di_tr)
    di_tr_sc = _scale_list(di_tr, bnds)
    di_va_sc = _scale_list(di_va, bnds)
    di_te_sc = _scale_list(di_te, bnds)

    max_pts = max((len(d) for d in di_tr_sc + di_va_sc + di_te_sc), default=1)
    X_tr = _pack_diagrams(di_tr_sc, max_pts)
    X_va = _pack_diagrams(di_va_sc, max_pts)
    X_te = _pack_diagrams(di_te_sc, max_pts)

    Z_tr = np.zeros((len(X_tr), 1), dtype=np.float32)
    Z_va = np.zeros((len(X_va), 1), dtype=np.float32)
    Z_te = np.zeros((len(X_te), 1), dtype=np.float32)

    have_ckpt = Path(str(weights_prefix) + ".index").exists() and meta_path.exists()

    if have_ckpt and not cfg.force_train:
        with tf.device(device):
            model = _build_perslay_model(max_pts=max_pts, cfg=cfg)
            model.load_weights(str(weights_prefix))
            y_prob = model.predict([X_te, Z_te], batch_size=max(32, cfg.batch_size), verbose=0).reshape(-1)
        metrics = _metrics_from_probs(y_te, y_prob)
        return metrics, dict(weights=weights_prefix, meta=meta_path)

    with tf.device(device):
        model = _build_perslay_model(max_pts=max_pts, cfg=cfg)

        meta = dict(
            max_pts=int(max_pts),
            img_size=cfg.img_size, bounds=cfg.bounds,
            variance=float(cfg.variance), power=float(cfg.power),
            bnds=bnds,
        )
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        ckpt = keras.callbacks.ModelCheckpoint(
            filepath=str(weights_prefix),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=0
        )
        es = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
            verbose=0
        )

        model.fit(
            x=[X_tr, Z_tr], y=y_tr,
            validation_data=([X_va, Z_va], y_va),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
            callbacks=[ckpt, es]
        )

        model.save_weights(str(weights_prefix))
        y_prob = model.predict([X_te, Z_te], batch_size=max(32, cfg.batch_size), verbose=0).reshape(-1)

    metrics = _metrics_from_probs(y_te, y_prob)
    return metrics, dict(weights=weights_prefix, meta=meta_path)


def full_eval_val_test(
    *,
    paths: Paths,
    pd_std,
    split_std,
    model_name: str = "perslay_e2e_std",
    cfg: PerslayConfig = PerslayConfig(),
    eval_dir: Optional[Path] = None,
    model_tag_prefix: str = "PersLayE2E__",
):
    test_metrics, artifacts = run_perslay_e2e(
        paths=paths, pd_std=pd_std, split_std=split_std, model_name=model_name, cfg=cfg
    )

    eval_dir = eval_dir or (paths.RESULTS_DIR / "Evaluation")
    eval_dir.mkdir(parents=True, exist_ok=True)

    meta = pickle.load(open(artifacts["meta"], "rb"))
    bnds = meta.get("bnds", ((0.0, 1.0), (0.0, 1.0)))
    max_pts = int(meta.get("max_pts", 1))

    (_, _, di_tr), (_, y_va, di_va), (_, y_te, di_te) = _prep_sets_from_split(pd_std, split_std)
    di_va_sc = _scale_list(di_va, bnds)
    di_te_sc = _scale_list(di_te, bnds)

    max_pts2 = max(max_pts, max((len(d) for d in di_va_sc + di_te_sc), default=1))

    X_va = _pack_diagrams(di_va_sc, max_pts2)
    X_te = _pack_diagrams(di_te_sc, max_pts2)
    Z_va = np.zeros((len(X_va), 1), dtype=np.float32)
    Z_te = np.zeros((len(X_te), 1), dtype=np.float32)

    gpus = tf.config.list_physical_devices("GPU")
    if not cfg.cpu_only and gpus:
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        device = "/GPU:0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        device = "/CPU:0"

    with tf.device(device):
        model = _build_perslay_model(max_pts=max_pts2, cfg=cfg)
        model.load_weights(str(artifacts["weights"]))
        pva = model.predict([X_va, Z_va], batch_size=max(32, cfg.batch_size), verbose=0).reshape(-1)
        pte = model.predict([X_te, Z_te], batch_size=max(32, cfg.batch_size), verbose=0).reshape(-1)

    model_tag = f"{model_tag_prefix}{Path(artifacts['weights']).stem}"
    df = evaluate_and_save(
        model_tag,
        val_true=y_va,  val_prob=pva,
        test_true=y_te, test_prob=pte,
        out_dir=eval_dir,
        base_threshold=0.5,
        n_boot=2000,
        ece_bins=15,
        ece_strategy="uniform",
        seed=14,
    )
    return df, test_metrics, artifacts