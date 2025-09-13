from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Polygon

_RC_BASE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Palatino", "Charter"],
    "mathtext.fontset": "stix",
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,
    "figure.autolayout": False,
}

COL_NODE = {
    "patient":"#1f77b4","admission":"#2ca02c","diagnosis":"#d62728","provider":"#ff7f0e",
    "medication":"#9467bd","procedure":"#8c564b","icu_stay":"#e377c2","icu_unit":"#7f7f7f",
}
_NODE_SIZE, _EDGE_WIDTH = 120, 2.2

RESULTS_DIR = Path("..") / "Results"
FIG_DIR     = RESULTS_DIR / "Figures"

def apply_base_style():
    plt.rcParams.update(_RC_BASE)

def _safe_w(d) -> Optional[float]:
    w = d.get("weight", d.get("time", None))
    try:
        w = float(w) if w is not None else None
        return w if (w is not None and np.isfinite(w)) else None
    except Exception:
        return None

def _unique_sorted_weights(G: nx.Graph) -> List[float]:
    return sorted({w for *_e,d in G.edges(data=True) if (w := _safe_w(d)) is not None})

def _threshold_graph(G: nx.Graph, thr: float) -> nx.Graph:
    H = nx.Graph(); H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = _safe_w(d)
        if w is not None and w <= thr:
            H.add_edge(u, v, **d)
    return H

def _tri_faces(H: nx.Graph):
    tris = []
    for clique in nx.enumerate_all_cliques(H):
        if len(clique) == 3: tris.append(tuple(clique))
        elif len(clique) > 3: break
    return tris

def _draw_complex(ax, H: nx.Graph, pos: Dict, *, node_size=_NODE_SIZE, edge_w=_EDGE_WIDTH):
    node_colors = [COL_NODE.get(H.nodes[n].get("type","unknown"), "lightgray") for n in H]
    if pos:
        all_pos = np.array(list(pos.values()))
        x_min, x_max = all_pos[:,0].min(), all_pos[:,0].max()
        y_min, y_max = all_pos[:,1].min(), all_pos[:,1].max()
        x_pad = max(1e-6, (x_max - x_min) * 0.12); y_pad = max(1e-6, (y_max - y_min) * 0.12)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    for a,b,c in _tri_faces(H):
        if a in pos and b in pos and c in pos:
            ax.add_patch(Polygon([pos[a],pos[b],pos[c]], closed=True,
                                 alpha=0.20, linewidth=0.0, facecolor="tab:cyan"))
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="0.4", width=edge_w, alpha=0.85)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=node_colors,
                           node_size=node_size, linewidths=0.8, edgecolors="k", alpha=0.95)
    ax.axis("off"); ax.set_aspect("equal")

def _smart_sample_weights(wts: List[float], max_steps: int) -> List[float]:
    if len(wts) <= max_steps: return wts
    selected = {0, len(wts)-1}
    remaining = max_steps - 2
    if remaining > 0:
        percentiles = np.linspace(0, 1, remaining + 2)[1:-1]
        idx = [int(p * (len(wts)-1)) for p in percentiles]
        selected.update(idx)
    return [wts[i] for i in sorted(selected)]

def _arrow_between_axes(ax_from, ax_to, fig, color="0.25"):
    from matplotlib.patches import FancyArrowPatch
    a = ax_from.get_position(); b = ax_to.get_position()
    x0 = a.x1 + 0.005; x1 = b.x0 - 0.005
    y0 = (a.y0 + a.y1)/2.0; y1 = (b.y0 + b.y1)/2.0
    fig.add_artist(FancyArrowPatch(
        (x0, y0), (x1, y1),
        transform=fig.transFigure, arrowstyle='-|>', mutation_scale=14,
        lw=2.0, color=color, shrinkA=0, shrinkB=0, zorder=10
    ))

def _percentile_index(n: int, q=0.75) -> int:
    if n <= 1: return 0
    return int(np.ceil(q * (n - 1)))

def show_filtration_grid(ego_graphs: Dict, *, n_preview=1, max_steps=5, seed=100):
    apply_base_style()
    pids = list(ego_graphs.keys())[:max(0, int(n_preview))]
    rows_info = []
    for i, pid in enumerate(pids):
        G = ego_graphs[pid]
        pos = nx.spring_layout(G, seed=seed+i, weight=None, k=0.8, iterations=50)
        wts = _unique_sorted_weights(G)
        wts = _smart_sample_weights(wts, max_steps) if (max_steps is not None) else wts
        frames = [_threshold_graph(G, float(thr)) for thr in wts]
        rows_info.append(dict(pid=pid, pos=pos, frames=frames))
    max_cols = max((len(r["frames"]) for r in rows_info), default=0)
    if max_cols == 0: return None

    nrows = len(rows_info)
    step_w, sp_w = 2.4, 0.28
    ncols  = 2*max_cols - 1
    widths = [step_w if j%2==0 else sp_w for j in range(ncols)]

    fig = plt.figure(figsize=(sum(widths), max(2.2*nrows, 3.0)), dpi=300)
    gs  = plt.GridSpec(nrows, ncols, figure=fig, hspace=0.25, wspace=0.05,
                       left=0.05, right=0.98, bottom=0.08, top=0.90, width_ratios=widths)
    for ri, rinfo in enumerate(rows_info):
        axs = []
        ci = 0
        for k in range(max_cols):
            ax = fig.add_subplot(gs[ri, ci])
            if k < len(rinfo["frames"]): _draw_complex(ax, rinfo["frames"][k], rinfo["pos"])
            else: ax.axis("off")
            axs.append(ax); ci += 1
            if k < max_cols - 1: fig.add_subplot(gs[ri, ci]).axis("off"); ci += 1
        for a, b in zip(axs[:-1], axs[1:]): _arrow_between_axes(a, b, fig, color="0.25")
    plt.show()
    return fig

def pick_representative_pid(ego_graphs: Dict, *, exclude_type="medication", q=0.94) -> int:
    pids_all = list(ego_graphs.keys())
    if not pids_all: raise RuntimeError("ego_graphs is empty.")
    types_seen = set()
    for pid_ in pids_all:
        for n in ego_graphs[pid_].nodes():
            t = ego_graphs[pid_].nodes[n].get("type","unknown")
            if t in COL_NODE and t != exclude_type: types_seen.add(t)
    req = set(types_seen)
    candidates = []
    for pid_ in pids_all:
        G_ = ego_graphs[pid_]
        types_here = {G_.nodes[n].get("type","unknown") for n in G_.nodes()}
        if req.issubset(types_here): candidates.append((pid_, G_.number_of_nodes()))
    if candidates:
        candidates.sort(key=lambda x: (x[1], str(x[0])))
        return candidates[_percentile_index(len(candidates), q=q)][0]
    # fallback
    best_pid, best_cov, best_size = None, -1, -1
    for pid_ in pids_all:
        G_ = ego_graphs[pid_]
        types_here = {G_.nodes[n].get("type","unknown") for n in G_.nodes()}
        cov = len(req.intersection(types_here)); size = G_.number_of_nodes()
        if (cov > best_cov) or (cov == best_cov and size > best_size) or \
           (cov == best_cov and size == best_size and (best_pid is None or str(pid_) < str(best_pid))):
            best_pid, best_cov, best_size = pid_, cov, size
    return best_pid

def show_filtration_for_pid(ego_graphs: Dict, pid: int, *, max_steps=5, seed=0):
    apply_base_style()
    G = ego_graphs[pid]
    pos = nx.spring_layout(G, seed=seed, weight=None, k=0.8, iterations=50)
    wts = _smart_sample_weights(_unique_sorted_weights(G), max_steps) if max_steps else _unique_sorted_weights(G)
    frames = [_threshold_graph(G, float(thr)) for thr in wts]
    step_w, sp_w = 2.4, 0.28
    ncols  = 2*len(frames) - 1
    widths = [step_w if j%2==0 else sp_w for j in range(ncols)]
    fig = plt.figure(figsize=(sum(widths), 3.2), dpi=300)
    gs  = plt.GridSpec(1, ncols, figure=fig, hspace=0.25, wspace=0.05,
                       left=0.05, right=0.98, bottom=0.16, top=0.92, width_ratios=widths)
    axs, ci = [], 0
    for k, H in enumerate(frames):
        ax = fig.add_subplot(gs[0, ci]); _draw_complex(ax, H, pos); axs.append(ax); ci += 1
        if k < len(frames) - 1: fig.add_subplot(gs[0, ci]).axis("off"); ci += 1
    for a, b in zip(axs[:-1], axs[1:]): _arrow_between_axes(a, b, fig, color="0.25")
    plt.show()
    return fig, pos

# ego network with labels
def show_graph_with_weights(G: nx.Graph, *, pos: Optional[Dict]=None):
    apply_base_style()
    if pos is None or set(pos.keys()) != set(G.nodes()):
        pos = nx.spring_layout(G, seed=0, weight=None, k=0.8, iterations=50)
    edge_labels = { (u,v): f"{w:.2g}" for u,v,d in G.edges(data=True) if (w := _safe_w(d)) is not None }
    node_colors = [COL_NODE.get(G.nodes[n].get("type","unknown"), "lightgray") for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=300)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="0.4", width=1.8, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=140, edgecolors="black", linewidths=0.7, alpha=0.97)
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=edge_labels, font_size=8, label_pos=0.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
        )
    ax.axis("off"); ax.set_aspect("equal", adjustable="datalim")
    plt.show()
    return fig

def export_gexf_minimal(ego_graphs: Dict, pid: int, out_dir: Path, *,
                        keep_zero_constant_weight=True, epsilon_for_zero=1e-6) -> Path:
    def _hex_to_rgb255(hex_color: str):
        s = hex_color.lstrip("#"); s = "".join([c*2 for c in s]) if len(s)==3 else s
        return int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)

    G_src = ego_graphs[pid]
    G_out = nx.Graph(); G_out.graph["name"] = f"ego_graph_{pid}"
    for n, dat_in in G_src.nodes(data=True):
        t = dat_in.get("type", "unknown")
        hexcol = COL_NODE.get(t, "#cccccc")
        r, g, b = _hex_to_rgb255(hexcol)
        G_out.add_node(n, type=t, viz={"color": {"r": int(r), "g": int(g), "b": int(b)}})
    for u, v, d_in in G_src.edges(data=True):
        w = _safe_w(d_in)
        if w is None: continue
        w = float(np.clip(w, 0.0, 1.0))
        if keep_zero_constant_weight:
            G_out.add_edge(u, v, w=w, weight=1.0)
        else:
            gephi_w = w if w > 0.0 else float(epsilon_for_zero)
            G_out.add_edge(u, v, weight=gephi_w, w=w)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ego_graph_min_{pid}.gexf"
    nx.write_gexf(G_out, out_path)
    return out_path

def _cap_essentials(bd, pad=1e-6):
    bd = np.asarray(bd, float)
    if bd.size == 0: return bd
    finite = bd[np.isfinite(bd[:,1])]
    if finite.size == 0:
        cap = 1.0 + pad
        return np.column_stack([bd[:,0], np.full(len(bd), cap)])
    cap = float(finite[:,1].max() + pad)
    nonfin = bd[~np.isfinite(bd[:,1])]
    if nonfin.size:
        capped = np.column_stack([nonfin[:,0], np.full(len(nonfin), cap)])
        return np.vstack([finite, capped])
    return finite

def _dimsdict_to_bd_all(dims):
    parts = []
    for d in (0,1):
        bd = np.asarray(dims.get(d, np.empty((0,2))), float)
        if bd.size: parts.append(_cap_essentials(bd))
    return np.vstack(parts) if parts else np.zeros((0,2), float)

def _pd_points_with_dim(dims):
    pts = []
    for dim in (0,1):
        bd = np.asarray(dims.get(dim, np.empty((0,2))), float)
        if bd.size:
            for (b,dv) in bd: pts.append([b, dv, dim])
    return np.array(pts) if pts else np.zeros((0,3), float)

def _plot_pd(ax, dgm):
    if dgm is None or dgm.size == 0: ax.axis('off'); return
    dgm = dgm[np.isin(dgm[:,2], [0,1])]
    if dgm.size == 0: ax.axis('off'); return
    finite = dgm[np.isfinite(dgm[:,1])]; infs = dgm[~np.isfinite(dgm[:,1])]
    vals = dgm[:,:2].ravel(); vals = vals[np.isfinite(vals)]
    lo, hi = (vals.min(), vals.max()) if vals.size else (0.0, 1.0)
    span = max(1e-8, hi-lo)
    lo -= 0.05*span; hi += 0.05*span
    ax.plot([lo,hi],[lo,hi],'--',color='gray',alpha=0.7,linewidth=1)
    styles = {0:dict(marker='o', edgecolor='#1f77b4', facecolor='none'),
              1:dict(marker='s', edgecolor='#d62728', facecolor='none')}
    for d in (0,1):
        pts = finite[finite[:,2]==d]
        if pts.size:
            ax.scatter(pts[:,0], pts[:,1], s=20, linewidths=1,
                       edgecolors=styles[d]['edgecolor'], facecolors=styles[d]['facecolor'],
                       marker=styles[d]['marker'])
    if infs.size:
        ymax = hi + 0.06*span
        for d,c in [(0,'#1f77b4'),(1,'#d62728')]:
            ptsi = infs[infs[:,2]==d]
            if ptsi.size: ax.scatter(ptsi[:,0], [ymax]*len(ptsi), marker='x', s=28, linewidths=1.2, color=c)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title(''); ax.set_xlabel(''); ax.set_ylabel('')

def _show_pi(ax, img_mat):
    if img_mat is None or getattr(img_mat, "size", 0) == 0: ax.axis('off'); return
    im = ax.imshow(img_mat, origin='lower', interpolation='bilinear', cmap='viridis')
    ax.set_title(''); ax.set_xticks([]); ax.set_yticks([])
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax); cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def _show_pl(ax, lan_mat):
    if lan_mat is None or getattr(lan_mat, "size", 0) == 0: ax.axis('off'); return
    x = np.linspace(0, 1, lan_mat.shape[1])
    cols = plt.cm.viridis(np.linspace(0, 0.85, lan_mat.shape[0]))
    for k in range(lan_mat.shape[0]): ax.plot(x, lan_mat[k], lw=1.5, color=cols[k])
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_title(''); ax.set_xlabel(''); ax.set_ylabel('')

def compute_pi_pl(bd_all, *, pi_res=(128,128), pi_bw=1.0, lan_num=15, lan_res=256):
    from gudhi.representations.vector_methods import PersistenceImage, Landscape
    if bd_all is None or getattr(bd_all, "size", 0) == 0:
        return None, None
    img_mat = PersistenceImage(bandwidth=pi_bw, resolution=list(pi_res)).fit_transform([bd_all])[0].reshape(pi_res)
    lan_mat = Landscape(num_landscapes=lan_num, resolution=lan_res).fit_transform([bd_all])[0].reshape(lan_num, lan_res)
    return img_mat, lan_mat

def panel_pd_pl_pi_for_pid(pd_std: List[Tuple[int,int,dict]], pid: int, *,
                           pi_res=(128,128), pi_bw=1.0, lan_num=15, lan_res=256):
    apply_base_style()
    dims = None
    for _pid, _y, d in pd_std:
        if int(_pid)==int(pid): dims = d; break
    if dims is None: raise RuntimeError(f"pid {pid} not found in pd_std.")
    bd_all = _dimsdict_to_bd_all(dims)
    dgm = _pd_points_with_dim(dims)
    img_mat, lan_mat = compute_pi_pl(bd_all, pi_res=pi_res, pi_bw=pi_bw, lan_num=lan_num, lan_res=lan_res)

    panel_w, sp_w = 3.0, 0.45
    widths = [panel_w, sp_w, panel_w, sp_w, panel_w]
    fig = plt.figure(figsize=(sum(widths), 2.9), dpi=300)
    gs  = plt.GridSpec(1, 5, figure=fig, hspace=0.35, wspace=0.05,
                       left=0.06, right=0.98, bottom=0.08, top=0.90, width_ratios=widths)
    ax_pl = fig.add_subplot(gs[0, 0]); _show_pl(ax_pl, lan_mat)
    fig.add_subplot(gs[0, 1]).axis('off')
    ax_pd = fig.add_subplot(gs[0, 2]); _plot_pd(ax_pd, dgm)
    fig.add_subplot(gs[0, 3]).axis('off')
    ax_pi = fig.add_subplot(gs[0, 4]); _show_pi(ax_pi, img_mat)
    _arrow_between_axes(ax_pd, ax_pl, fig, color="0.25")
    _arrow_between_axes(ax_pd, ax_pi, fig, color="0.25")
    plt.show()
    return fig, dict(bd_all=bd_all, dgm=dgm, img_mat=img_mat, lan_mat=lan_mat)

def save_pl_pd_pi_images(
    bd_all, dgm, img_mat, *, out_dir: Path, pid: int,
    pi_res: Tuple[int,int]=(128,128), pi_bw: float=1.0, lan_num: int=15, lan_res: int=256
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pid_str = str(int(pid))
    # compute missing PI/PL
    img_c, lan_c = compute_pi_pl(bd_all, pi_res=pi_res, pi_bw=pi_bw, lan_num=lan_num, lan_res=lan_res)
    img_use = img_mat if (img_mat is not None and getattr(img_mat, "size", 0)) else img_c
    lan_use = lan_c

    # PL
    fig_pl, ax_pl = plt.subplots(figsize=(3.0, 2.1), dpi=300)
    _show_pl(ax_pl, lan_use)
    p_pl = out_dir / f"ego_pid{pid_str}_PL.png"
    fig_pl.savefig(p_pl, dpi=600, bbox_inches="tight"); plt.close(fig_pl)

    # PD
    fig_pd, ax_pd = plt.subplots(figsize=(3.0, 2.1), dpi=300)
    _plot_pd(ax_pd, dgm)
    p_pd = out_dir / f"ego_pid{pid_str}_PD.png"
    fig_pd.savefig(p_pd, dpi=600, bbox_inches="tight"); plt.close(fig_pd)

    # PI
    fig_pi, ax_pi = plt.subplots(figsize=(3.0, 2.1), dpi=300)
    _show_pi(ax_pi, img_use)
    p_pi = out_dir / f"ego_pid{pid_str}_PI.png"
    fig_pi.savefig(p_pi, dpi=600, bbox_inches="tight"); plt.close(fig_pi)

    return {"PL": p_pl, "PD": p_pd, "PI": p_pi}

def roc_grid_panel(eval_dir: Path, *, out_dir: Path = FIG_DIR) -> Tuple[Path, Path]:
    eval_dir = Path(eval_dir)
    spec_rows = [
        [("Image",     "MLP_IMAGE*__test@J__roc.npz",            None,                ()),
         ("Landscape", "MLP_LANDSCAPE*__test@J__roc.npz",        None,                ()),
         ("PersLay",   "PersLayE2E*__test@J__roc.npz",           None,                ())],
        [("GCN",       "GNN_GCN*__test@J__roc.npz",              None,                ()),
         ("GraphSAGE", "GNN_GraphSAGE*__test@J__roc.npz",        None,                ()),
         ("GIN",       "GNN_GIN*__test@J__roc.npz",              None,                ())],
        [("TopoPool",  "TopoPool*__test@J__roc.npz",             "TopoPool",          ("+SAPI",)),
         ("TTG-NN",    "TTG_*__test@J__roc.npz",                 None,                ()),
         ("LAI-Pool",  "TopoPool+SAPI_tokens*__test@J__roc.npz", "TopoPool+SAPI_tokens", ())]
    ]
    def best_match(glob_pat, must_start=None, must_not_contains=()):
        files = sorted(eval_dir.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            name = f.name
            if must_start and not name.startswith(must_start): continue
            if any(bad in name for bad in must_not_contains): continue
            return f
        return files[0] if files else None

    grid_paths = [[(lab, best_match(pat, ms, mnc)) for (lab, pat, ms, mnc) in row] for row in spec_rows]

    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Computer Modern Roman"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    def fmt_auc_title(auc, lo, hi):
        if np.isnan(auc) or np.isnan(lo) or np.isnan(hi): return f"AUC {auc:.2f}"
        hw = (hi - lo) / 2.0
        return f"AUC {auc:.2f} ± {hw:.02f}"

    with plt.rc_context(rc):
        fig_w, fig_h = 7.0, 7.2
        fig, axes = plt.subplots(3, 3, figsize=(fig_w, fig_h))
        axes = axes.ravel()
        for ax in axes:
            ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_box_aspect(1.0)
            ax.grid(True, which="both", linewidth=0.5, alpha=0.25)
            ax.plot([0,1],[0,1], linestyle=(0,(2,2)), linewidth=0.9, color="0.6")
            ax.tick_params(direction="out", length=3, width=0.8)
            ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks(np.linspace(0,1,6))
            for s in ("top","right"): ax.spines[s].set_visible(False)

        flat = [item for row in grid_paths for item in row]
        for i, ax in enumerate(axes):
            label, path = flat[i]
            if path is None or not path.exists():
                ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=9)
                ax.set_title(label, pad=2)
            else:
                data = np.load(path, allow_pickle=True)
                fpr, tpr = data["fpr"], data["tpr"]
                auc = float(data["auc"]) if "auc" in data else np.nan
                lo, hi = (np.nan, np.nan)
                if "auc_ci" in data and len(data["auc_ci"]) == 2:
                    lo, hi = map(float, data["auc_ci"])
                ax.plot(fpr, tpr, linewidth=1.8, color="0.0", solid_capstyle="round")
                ax.set_title(f"{label}: {fmt_auc_title(auc, lo, hi)}", pad=2)
            r, c = divmod(i, 3)
            if c == 0: ax.set_ylabel("TPR")
            else: ax.set_yticklabels([])
            if r == 2: ax.set_xlabel("FPR")
            else: ax.set_xticklabels([])
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = out_dir / "roc_grid.pdf"
        png = out_dir / "roc_grid.png"
        fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
        plt.show()
    return pdf, png