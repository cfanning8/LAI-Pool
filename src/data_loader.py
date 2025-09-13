from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional
import sys, importlib, pickle
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from tqdm.auto import tqdm
from gtda.homology import FlagserPersistence
from gudhi.representations.vector_methods import PersistenceImage, Landscape

class Paths:
    def __init__(self, root: Path):
        self.ROOT        = root
        self.DATA_DIR    = self.ROOT / "Data"
        self.RAW_DIR     = self.DATA_DIR / "Raw"
        self.PROC_DIR    = self.DATA_DIR / "Processed"
        self.RESULTS_DIR = self.ROOT / "Results"
        self.MODELS_DIR  = self.RESULTS_DIR / "Models"
        self.SPLITS_DIR  = self.RESULTS_DIR / "Splits"
        for p in (self.PROC_DIR, self.MODELS_DIR, self.SPLITS_DIR):
            p.mkdir(parents=True, exist_ok=True)

        self.EGO_PKL   = self.RAW_DIR / "ego_graphs.pkl"
        self.FLAGS_CSV = self.RAW_DIR / "patient_summary_flags.csv"

        self.PD_STD_PKL = self.PROC_DIR / "ego_persistence.pkl"

        self.IMG_PKL = self.PROC_DIR / "ego_image.pkl"
        self.LAN_PKL = self.PROC_DIR / "ego_landscape.pkl"

        self.SPLIT_PKL     = self.SPLITS_DIR / "ego_split_train_test.pkl"
        self.SPLIT_BAL_PKL = self.SPLITS_DIR / "ego_split_train_val_test_balanced.pkl"

def get_paths(root: Path  str = "..") -> Paths:
    return Paths(Path(root))

def nice(p: Path, root: Path  str = "..") -> str:
    root = Path(root)
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return str(p)

def safe_pickle_load(pkl_path: Path):
    try:
        from numpy.core import multiarray as _mu  
        if not hasattr(np, "scalar"):  
            from numpy.core import multiarray as _mu2  
            np.scalar = _mu2.scalar  
    except Exception:
        pass

    class _CompatUnpickler(pickle.Unpickler):
        NAME_MAP = {
            ("numpy", "scalar"): "numpy.core.multiarray.scalar",
            ("numpy", "int"): "builtins.int",
            ("numpy", "float"): "builtins.float",
            ("numpy", "bool"): "builtins.bool",
            ("numpy", "object"): "builtins.object",
        }
        def find_class(self, module, name):
            if module.startswith("numpy._"):
                module = module.replace("numpy._core", "numpy.core").replace("numpy._", "numpy.")
            key = (module, name)
            if key in self.NAME_MAP:
                modpath, attr = self.NAME_MAP[key].rsplit(".", 1)
                return getattr(importlib.import_module(modpath), attr)
            return super().find_class(module, name)

    with open(pkl_path, "rb") as f:
        try:
            return _CompatUnpickler(f).load()
        except ModuleNotFoundError as e:
            name = getattr(e, "name", "")
            if name and name.startswith("numpy._"):
                sys.modules[name] = importlib.import_module(
                    name.replace("numpy._core", "numpy.core").replace("numpy._", "numpy.")
                )
                f.seek(0)
                return _CompatUnpickler(f).load()
            raise

def load_graphs_and_flags(paths: Paths) -> Tuple[Dict[Any, nx.Graph], pd.DataFrame]:
    ego_graphs: Dict[Any, nx.Graph] = safe_pickle_load(paths.EGO_PKL)
    flags_df = pd.read_csv(paths.FLAGS_CSV).set_index("subject_id")
    return ego_graphs, flags_df

def ensure_edge_weights(ego_graphs: Dict[Any, nx.Graph]) -> None:
    if not ego_graphs:
        return
    sample = next(iter(ego_graphs.values()))
    if any(("weight" in d) for *_e, d in sample.edges(data=True)):
        return

    times: List[float] = [
        float(d["time"])
        for G in ego_graphs.values()
        for *_e, d in G.edges(data=True)
        if "time" in d and pd.notna(d["time"])
    ]
    if not times:
        return

    tmin, tmax = min(times), max(times)
    if tmax == tmin:
        scaler = lambda t: 0.0
    else:
        scaler = lambda t: (t - tmin) / (tmax - tmin) 

    for G in ego_graphs.values():
        for _u, _v, d in G.edges(data=True):
            if "time" in d and pd.notna(d["time"]):
                d["weight"] = float(scaler(float(d["time"])))
            else:
                d["weight"] = np.nan

MAX_FILTRATION = 1.0

_flagser = FlagserPersistence(
    homology_dimensions=[0, 1],
    directed=False,
    filtration="max",
    coeff=2,
    reduced_homology=True,
)

def nx_to_weighted_sparse(G: nx.Graph) -> sp.csr_matrix:
    idx = {n: i for i, n in enumerate(G.nodes)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for u, v, d in G.edges(data=True):
        w = d.get("weight")
        if w is None or (isinstance(w, float) and np.isnan(w)):
            continue
        i, j = idx[u], idx[v]
        w = float(w)
        rows += [i, j]
        cols += [j, i]
        data += [w, w]
    n = len(idx)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

def _cap_inf_deaths_to_one(bd: np.ndarray) -> np.ndarray:
    if bd.size == 0:
        return bd
    out = bd.copy()
    infmask = ~np.isfinite(out[:, 1])
    if np.any(infmask):
        out[infmask, 1] = MAX_FILTRATION
    return out

def _split_dims_cap_to_one(dgm: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    if dgm.ndim == 2 and dgm.shape[1] >= 3:
        for d in (0, 1):
            bd = dgm[dgm[:, 2] == d][:, :2]
            if bd.size == 0:
                continue
            bd = bd[np.isfinite(bd[:, 0])]
            if bd.size == 0:
                continue
            out[d] = _cap_inf_deaths_to_one(bd)
    return out

def compute_pds_forward(
    ego_graphs: Dict[Any, nx.Graph],
    flags_df: pd.DataFrame,
    progress: bool = False,
) -> List[Tuple[Any, Optional[int], Dict[int, np.ndarray]]]:
    """Return list of (pid, label, {dim: bd_array}) with ∞ deaths capped to 1.0."""
    items = list(ego_graphs.items())
    iterator = tqdm(items, desc="PDs (forward)") if progress else items
    diagrams: List[Tuple[Any, Optional[int], Dict[int, np.ndarray]]] = []
    for pid, G in iterator:
        y: Optional[int] = None
        if pid in flags_df.index:
            val = flags_df.loc[pid, "hospital_expire_flag"]
            y = int(val) if pd.notna(val) else None
        W = nx_to_weighted_sparse(G)
        dgm = _flagser.fit_transform([W])[0]  # (n,3): [birth, death, dim]
        dims = _split_dims_cap_to_one(dgm)
        diagrams.append((pid, y, dims))
    return diagrams

def load_or_compute_pds_forward(
    paths: Paths,
    ego_graphs: Dict[Any, nx.Graph],
    flags_df: pd.DataFrame,
    cache_path: Optional[Path] = None,
    progress: bool = False,
) -> List[Tuple[Any, Optional[int], Dict[int, np.ndarray]]]:
    cache = cache_path if cache_path is not None else paths.PD_STD_PKL
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    diagrams = compute_pds_forward(ego_graphs, flags_df, progress=progress)
    with open(cache, "wb") as f:
        pickle.dump(diagrams, f, protocol=pickle.HIGHEST_PROTOCOL)
    return diagrams

def _finite_only_bd(bd: np.ndarray) -> np.ndarray:
    if bd is None or bd.size == 0:
        return np.empty((0, 2), dtype=float)
    bd = np.asarray(bd, dtype=float)
    mask = np.isfinite(bd[:, 0]) & np.isfinite(bd[:, 1]) & (bd[:, 1] > bd[:, 0])
    return bd[mask]

def vectorize_forward(
    diagrams: List[Tuple[Any, Optional[int], Dict[int, np.ndarray]]],
    pi_bandwidth: float = 1.0,
    pi_res: Tuple[int, int] = (50, 50),
    land_num: int = 5,
    land_res: int = 100,
    progress: bool = False,
) -> Tuple[List[Tuple[Any, Optional[int], np.ndarray]], List[Tuple[Any, Optional[int], np.ndarray]]]:
    pi_vec = {d: PersistenceImage(bandwidth=pi_bandwidth, resolution=list(pi_res)) for d in (0, 1)}
    pl_vec = {d: Landscape(num_landscapes=land_num, resolution=land_res) for d in (0, 1)}

    per_dim_lists = {d: [] for d in (0, 1)}
    for _pid, _y, dims in diagrams:
        for d in (0, 1):
            per_dim_lists[d].append(_finite_only_bd(dims.get(d, np.empty((0, 2)))))

    for d in (0, 1):
        try:
            pi_vec[d].fit(per_dim_lists[d])
            pl_vec[d].fit(per_dim_lists[d])
        except Exception:
            pass

    iterable = tqdm(diagrams, desc="Vectorize (forward)") if progress else diagrams
    img_rows: List[Tuple[Any, Optional[int], np.ndarray]] = []
    lan_rows: List[Tuple[Any, Optional[int], np.ndarray]] = []

    for pid, y, dims in iterable:
        imgs, lans = [], []
        for d in (0, 1):
            bd = _finite_only_bd(dims.get(d, np.empty((0, 2))))
            if bd.size:
                imgs.append(pi_vec[d].transform([bd])[0].ravel())
                lans.append(pl_vec[d].transform([bd])[0].ravel())
            else:
                imgs.append(np.zeros(pi_res[0] * pi_res[1], dtype=float))
                lans.append(np.zeros(land_num * land_res, dtype=float))
        img_rows.append((pid, y, np.concatenate(imgs)))
        lan_rows.append((pid, y, np.concatenate(lans)))

    return img_rows, lan_rows

def load_or_compute_vectors_forward(
    paths: Paths,
    diagrams: List[Tuple[Any, Optional[int], Dict[int, np.ndarray]]],
    img_cache: Optional[Path] = None,
    lan_cache: Optional[Path] = None,
    pi_bandwidth: float = 1.0,
    pi_res: Tuple[int, int] = (50, 50),
    land_num: int = 5,
    land_res: int = 100,
    progress: bool = False,
) -> Tuple[List[Tuple[Any, Optional[int], np.ndarray]], List[Tuple[Any, Optional[int], np.ndarray]]]:
    img_path = img_cache if img_cache is not None else paths.IMG_PKL
    lan_path = lan_cache if lan_cache is not None else paths.LAN_PKL

    if img_path.exists() and lan_path.exists():
        with open(img_path, "rb") as f:
            img_rows = pickle.load(f)
        with open(lan_path, "rb") as f:
            lan_rows = pickle.load(f)
        return img_rows, lan_rows

    img_rows, lan_rows = vectorize_forward(
        diagrams,
        pi_bandwidth=pi_bandwidth,
        pi_res=pi_res,
        land_num=land_num,
        land_res=land_res,
        progress=progress,
    )

    with open(img_path, "wb") as f:
        pickle.dump(img_rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(lan_path, "wb") as f:
        pickle.dump(lan_rows, f, protocol=pickle.HIGHEST_PROTOCOL)

    return img_rows, lan_rows

__all__ = [
    "Paths", "get_paths", "nice",
    "safe_pickle_load",
    "load_graphs_and_flags", "ensure_edge_weights",
    "compute_pds_forward", "load_or_compute_pds_forward",
    "vectorize_forward", "load_or_compute_vectors_forward",
    "nx_to_weighted_sparse",
]