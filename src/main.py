from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

import numpy as np
import torch

from data_loader import (
    get_paths,
    load_graphs_and_flags,
    ensure_edge_weights,
    load_or_compute_pds_forward,
    load_or_compute_vectors_forward,
)
from utils import (
    build_pid_universe,
    compute_or_load_balanced_split,
    evaluate_and_save,
)

from laipool import (
    build_pid2idx, validate_pid2idx, pid_to_idx,
    build_type_vocab, build_std_loaders,
    TopoPoolSAPINet_Tokens, LaiPoolConfig,
    train_topopool_tokens, evaluate_topopool_tokens,
)

from evaluation import run_topopool_full_eval

import visualizations as viz

def echo(msg: str):
    print(f"[main] {msg}")

def delete_if_exists(path: Path):
    try:
        if path.is_dir():
            shutil.rmtree(path)
            echo(f"Removed dir: {path}")
        elif path.exists():
            path.unlink()
            echo(f"Removed file: {path}")
    except Exception as e:
        echo(f"Warning: could not remove {path}: {e}")

def step_data(paths):
    echo("Loading graphs + flags…")
    ego_graphs, flags_df = load_graphs_and_flags(paths)
    ensure_edge_weights(ego_graphs)
    return ego_graphs, flags_df

def step_persistence(paths, ego_graphs, flags_df, *, force: bool = False):
    if force:
        for cand in ("PERSIST_PKL", "PERSIST_CACHE", "PD_PKL", "PROC_DIR"):
            p = getattr(paths, cand, None)
            if isinstance(p, Path) and p.exists():
                delete_if_exists(p)
    echo("Computing / loading PDs…")
    pd_std = load_or_compute_pds_forward(paths, ego_graphs, flags_df, progress=True)
    return pd_std

def step_vectors(paths, pd_std, *, force: bool = False):
    if force:
        for cand in ("IMG_PKL", "LAN_PKL"):
            p = getattr(paths, cand, None)
            if isinstance(p, Path) and p.exists():
                delete_if_exists(p)
    echo("Computing / loading vectorizations (PI / PL)…")
    img_std, lan_std = load_or_compute_vectors_forward(paths, pd_std, progress=True)
    return img_std, lan_std

def step_split(paths, ego_graphs, flags_df, pd_std, img_std, lan_std, *, force: bool = False):
    echo("Building eligible PID universe…")
    common_pids, y = build_pid_universe(
        pd_std=pd_std, ego_graphs=ego_graphs, flags_df=flags_df,
        require_vectors=True, img_std=img_std, lan_std=lan_std,
    )
    split_cache = paths.SPLITS_DIR / "ego_split_train_val_test_balanced.pkl"
    if force:
        delete_if_exists(split_cache)
    echo("Computing / loading balanced split…")
    split_std = compute_or_load_balanced_split(
        split_cache=split_cache,
        ego_flags_df=flags_df,
        common_pids=common_pids,
        y=y,
        seed=14,
        require_vectors=True,
        img_std=img_std,
        lan_std=lan_std,
        attach_vector_indices=False,
        img_cache=getattr(paths, "IMG_PKL", None),
        lan_cache=getattr(paths, "LAN_PKL", None),
    )
    return split_std

def step_lai_pool_train(paths, ego_graphs, flags_df, pd_std, split_std, *, warm_start: bool, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = paths.RESULTS_DIR / "Models"
    models_dir.mkdir(parents=True, exist_ok=True)

    PID2IDX = build_pid2idx(pd_std)
    rep = validate_pid2idx(PID2IDX, pd_std, ego_graphs=ego_graphs, flags_df=flags_df, split_std=split_std)
    echo(f"PID2IDX size: {rep['n_pid2idx']}")

    TYPE_LIST, TYPE_TO_IDX = build_type_vocab(ego_graphs)
    train_loader, val_loader, test_loader, info = build_std_loaders(
        split_std, ego_graphs, flags_df, PID2IDX, TYPE_LIST, TYPE_TO_IDX,
        batch_size=32, shuffle_train=True
    )
    echo(f"Loaders, {info}")

    sample = next(iter(train_loader))
    in_dim = int(sample.x.size(1))
    n_classes = int(sample.y.max().item() + 1)

    model = TopoPoolSAPINet_Tokens(
        in_dim=in_dim, n_classes=n_classes,
        hidden=64, dropout=0.0,
        pi_res=(32,32), sigma_pi=2.0, decay="exp",
        tok_grid=(8,8), dtok=64, datt=64, pool_ratio=0.5
    ).to(device)

    cfg = LaiPoolConfig(
        epochs=epochs, patience=20, lr=1e-4, weight_decay=0.0,
        poly_power=0.9, warm_start=warm_start, ckpt_prefix="topopool_sapi_tokens"
    )

    PD_CACHE = []
    for pid, _y, dims in pd_std:
        parts = []
        for d in (0, 1):
            bd = np.asarray(dims.get(d, np.empty((0,2))), dtype=float)
            if bd.size:
                dcol = np.full((len(bd), 1), float(d), dtype=np.float32)
                parts.append(np.concatenate([dcol, bd.astype(np.float32)], axis=1))
        arr = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float32)
        PD_CACHE.append(torch.from_numpy(arr))

    res = train_topopool_tokens(
        model, train_loader, val_loader,
        pd_cache=PD_CACHE, models_dir=models_dir, cfg=cfg, device=device
    )
    echo(f"Best (val): AUC={res['best_val_auc']:.3f}")
    return dict(
        model=model, loaders=(train_loader, val_loader, test_loader),
        models_dir=models_dir, device=device, pd_cache=PD_CACHE, train_result=res
    )

def step_lai_pool_eval(paths, assets: dict):
    device = assets["device"]
    models_dir = assets["models_dir"]
    train_loader, val_loader, test_loader = assets["loaders"]
    best_ckpt = models_dir / "topopool_sapi_tokens_best.pt"

    res = run_topopool_full_eval(
        model_class=TopoPoolSAPINet_Tokens,
        best_ckpt=best_ckpt,
        val_loader=val_loader,
        test_loader=test_loader,
        evaluate_and_save=evaluate_and_save,
        device=device,
        results_dir=paths.RESULTS_DIR,
        train_loader=train_loader,
        default_hparams=dict(
            hidden=64, dropout=0.0,
            pi_res=(32, 32), sigma_pi=2.0, decay="gauss",
            tok_grid=(8, 8), dtok=64, datt=64, pool_ratio=0.5,
        ),
    )
    echo("Evaluation complete; artifacts written.")
    return res

def step_visualizations(paths, ego_graphs, pd_std):

    viz.show_filtration_grid(ego_graphs, n_preview=1, max_steps=5)

    pid = viz.pick_representative_pid(ego_graphs, q=0.94)
    _, pos = viz.show_filtration_for_pid(ego_graphs, pid, max_steps=5)
    viz.show_graph_with_weights(ego_graphs[pid], pos=pos)

    fig, cache = viz.panel_pd_pl_pi_for_pid(pd_std, pid)
    _saved = viz.save_pl_pd_pi_images(
        cache["bd_all"], cache["dgm"], cache["img_mat"],
        out_dir=viz.FIG_DIR,  
        pid=pid
    )
    echo(f"Saved PL/PD/PI: {list(_saved.values())}")

    pdf, png = viz.roc_grid_panel(eval_dir=paths.RESULTS_DIR / "Evaluation", out_dir=viz.FIG_DIR)
    echo(f"Saved ROC grid: {pdf}, {png}")

def build_argparser():
    p = argparse.ArgumentParser(
        description="Recreate / run LAI-Pool experiment end-to-end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--project-root", default="..", help="Project root for data/Results (same as notebook).")
    sub = p.add_subparsers(dest="cmd", required=True)

    s_setup = sub.add_parser("setup", help="Load data, compute/load PDs & vectors, build split.")
    s_setup.add_argument("--force", action="store_true", help="Rebuild caches (PDs/vectors/split).")

    # train laipool
    s_train = sub.add_parser("train-laipool", help="Train (or warm-start) LAI-Pool + SAPI(tokens).")
    s_train.add_argument("--epochs", type=int, default=1000)
    s_train.add_argument("--no-warm-start", action="store_true", help="Train from scratch even if BEST exists.")

    # eval laipool
    sub.add_parser("eval-laipool", help="Load BEST and evaluate; write Evaluation artifacts.")

    # viz
    sub.add_parser("viz", help="Make key figures (filtration previews, PD/PL/PI, ROC grid).")

    # all-in-one
    s_all = sub.add_parser("recreate", help="Recreate the full experiment end-to-end.")
    s_all.add_argument("--epochs", type=int, default=1000, help="Epochs for LAI-Pool training.")

    return p

def main(argv=None):
    args = build_argparser().parse_args(argv)
    paths = get_paths(args.project_root)

    ego_graphs, flags_df = step_data(paths)

    if args.cmd in ("setup", "recreate"):
        force = getattr(args, "force", False) or (args.cmd == "recreate")
        pd_std = step_persistence(paths, ego_graphs, flags_df, force=force)
        img_std, lan_std = step_vectors(paths, pd_std, force=force)
        split_std = step_split(paths, ego_graphs, flags_df, pd_std, img_std, lan_std, force=force)
        echo("Setup complete.")
        if args.cmd == "setup":
            return 0
    else:
        pd_std = load_or_compute_pds_forward(paths, ego_graphs, flags_df, progress=False)
        img_std, lan_std = load_or_compute_vectors_forward(paths, pd_std, progress=False)
        split_std = compute_or_load_balanced_split(
            split_cache=paths.SPLITS_DIR / "ego_split_train_val_test_balanced.pkl",
            ego_flags_df=flags_df,
            common_pids=build_pid_universe(pd_std=pd_std, ego_graphs=ego_graphs, flags_df=flags_df,
                                           require_vectors=True, img_std=img_std, lan_std=lan_std)[0],
            y=build_pid_universe(pd_std=pd_std, ego_graphs=ego_graphs, flags_df=flags_df,
                                 require_vectors=True, img_std=img_std, lan_std=lan_std)[1],
            seed=14, require_vectors=True, img_std=img_std, lan_std=lan_std,
            attach_vector_indices=False,
            img_cache=getattr(paths, "IMG_PKL", None),
            lan_cache=getattr(paths, "LAN_PKL", None),
        )

    # Train
    if args.cmd in ("train-laipool", "recreate"):
        assets = step_lai_pool_train(
            paths=paths,
            ego_graphs=ego_graphs,
            flags_df=flags_df,
            pd_std=pd_std,
            split_std=split_std,
            warm_start=not getattr(args, "no_warm_start", False),
            epochs=getattr(args, "epochs", 1000),
        )
    else:
        PID2IDX = build_pid2idx(pd_std)
        TYPE_LIST, TYPE_TO_IDX = build_type_vocab(ego_graphs)
        loaders = build_std_loaders(
            split_std, ego_graphs, flags_df, PID2IDX, TYPE_LIST, TYPE_TO_IDX,
            batch_size=32, shuffle_train=False
        )
        assets = dict(
            loaders=loaders,
            models_dir=paths.RESULTS_DIR / "Models",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    # Eval
    if args.cmd in ("eval-laipool", "recreate"):
        _ = step_lai_pool_eval(paths, assets)

    # Viz
    if args.cmd in ("viz", "recreate"):
        step_visualizations(paths, ego_graphs, pd_std)

    echo("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
