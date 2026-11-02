"""
Spatially restricted pair-biased attention.

Restricts attention in Invariant Point Attention to a neighborhood of each
residue, selected from the *current noisy structure* at every sampling step.
This is an inference-time intervention on a pretrained dense model: no weights
are changed and no retraining is involved.

The restriction is applied as a multiplicative gate on the pairwise validity
mask that IPA already builds immediately before its softmax, so a neighborhood
covering every residue reproduces the dense model exactly.

Three neighborhood modes are provided. The two non-spatial ones are controls:
without them a result like "k=32 preserves designability" cannot distinguish
"the model needs spatial neighbors" from "the model needs some neighbors".

    knn     k nearest residues by distance in the current noisy structure
    random  k residues chosen uniformly at random (same budget, no geometry)
    seq     k nearest residues in sequence (local in chain, not in space)
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class SparsityConfig:
    """
    Configuration for neighborhood-restricted attention.

    Attributes:
        enabled: Master switch. When False every call is a no-op.
        k: Neighborhood size (number of residues attended per query,
            including self). A value >= N reproduces dense attention.
            Ignored when k_shape is not "const".
        k_shape: How the budget varies over the trajectory: "const", "linear",
            "cosine", "geom" or "step". See `k_at`.
        k_hi: Budget at t/T = 1 (pure noise), for every non-const shape.
        k_lo: Budget at t/T = 0 (final structure).
        k_frac: Read k_hi and k_lo as fractions of the chain length instead of
            absolute counts, since the measured budget scales with N.
        mode: One of "knn", "random", "seq".
        symmetrize: Whether to take the union of the mask and its transpose.
            Raw k-NN is asymmetric, which makes the effective budget
            per row exceed k; symmetrizing makes the attended graph
            undirected at the cost of a slightly larger effective k.
        t_lo: Lower bound of the timestep fraction (t / T) at which the
            restriction is active.
        t_hi: Upper bound of the timestep fraction at which the restriction
            is active. Defaults cover the whole trajectory.
        layers: Indices of structure layers to restrict. Empty means all.
        seed: Base seed for the "random" mode, for reproducibility.
        eps: Mass bound for the "thresh" mode.
        term: Logit the "thresh" mode selects on: "full" or "geom".
        shared: "thresh" keeps one key set for all heads, as kNN does.
        alpha: Entmax parameter for the "entmax" mode (1 = softmax, 2 = sparsemax).
        target: What the rule restricts: "ipa" (structure-module attention),
            "tri" (the third residue in the trunk's triangle updates), or
            "ipa,tri" for both.
        radius: Cutoff in Angstrom for the "radius" mode.
        union: Triangle updates keep a mediator k near i or near j, instead
            of near both.
        tri_t_lo, tri_t_hi: Timestep window for the triangle gate. None falls
            back to t_lo / t_hi, so attention and triangles can be scheduled
            separately when both are targets.
    """

    enabled: bool = False
    k: float = 32
    k_shape: str = "const"
    k_hi: float = 0.0
    k_lo: float = 0.0
    k_frac: bool = False
    mode: str = "knn"
    symmetrize: bool = True
    t_lo: float = 0.0
    t_hi: float = 1.0
    layers: List[int] = field(default_factory=list)
    seed: int = 0
    eps: float = 0.01
    term: str = "full"
    shared: bool = False
    alpha: float = 1.5
    target: str = "ipa"
    radius: float = 8.0
    union: bool = False
    tri_t_lo: Optional[float] = None
    tri_t_hi: Optional[float] = None

    def k_at(self, t_frac: Optional[float], n: int) -> int:
        """Budget for this timestep. Fig. 5 measured the need falling steeply
        over the first half of sampling, so a fixed k is oversized late and far
        too small early; these shapes spend the budget where it is needed."""
        if self.k_shape == "const":
            # k_hi, like every other shape, so a fractional const condition is
            # configured the same way as a decaying one. Reading self.k here
            # instead silently produced k=1 whenever a caller put the fraction
            # in k_hi and left k at 0.
            return max(int(round(self.k_hi * n)), 1) if self.k_frac else int(self.k)
        hi, lo = self.k_hi, self.k_lo
        if self.k_frac:
            hi, lo = hi * n, lo * n
        t = 0.0 if t_frac is None else min(max(float(t_frac), 0.0), 1.0)
        if self.k_shape == "linear":
            k = lo + (hi - lo) * t
        elif self.k_shape == "cosine":
            # Flat at both ends, steep in the middle: holds a near-dense budget
            # through the noisy phase instead of starting to cut immediately.
            k = lo + (hi - lo) * (1 - math.cos(math.pi * t)) / 2
        elif self.k_shape == "geom":
            # Constant ratio per unit t, which is what a log-scale plot of the
            # measured budget looks like over t/T in [0.5, 1].
            k = lo * (max(hi, 1.0) / max(lo, 1.0)) ** t
        elif self.k_shape == "step":
            k = hi if t >= 0.5 else lo
        else:
            raise ValueError(f"Unknown k schedule: {self.k_shape}")
        return max(int(round(k)), 1)

    def active_at(self, t_frac: Optional[float], layer_idx: Optional[int]) -> bool:
        """
        Whether the restriction applies at this point in sampling.

        Args:
            t_frac: Current timestep as a fraction of the total (1.0 at the
                noisiest step, approaching 0.0 at the final step). None when
                the caller has not registered a timestep, in which case the
                schedule is treated as satisfied.
            layer_idx: Index of the calling structure layer, or None.

        Returns:
            bool: True if the neighborhood mask should be applied.
        """
        if not self.enabled:
            return False
        if t_frac is not None and not (self.t_lo <= t_frac <= self.t_hi):
            return False
        if self.layers and layer_idx is not None and layer_idx not in self.layers:
            return False
        return True


# Module-level state. The denoiser's forward signature does not carry the
# diffusion timestep down to IPA, and threading it through every intermediate
# module would touch code unrelated to this experiment. The sampler registers
# the current timestep here instead; see `timestep` below.
CONFIG = SparsityConfig()
_T_FRAC: Optional[float] = None
_STEP_IDX: int = 0
# Native CA coordinates [B, N, 3] for mode "oracle": k-NN chosen from the true
# structure instead of the noisy one. Only the ground-truth denoising script
# sets it; sampling has no true structure to offer.
ORACLE: Optional[torch.Tensor] = None
# Noisy CA coordinates [B, N, 3] of the current forward pass, published by the
# denoiser: the trunk sees no frames, so the triangle gate has no other way
# to know where residues are.
TRUNK_COORDS: Optional[torch.Tensor] = None


def configure(**kwargs) -> SparsityConfig:
    """
    Update the global sparsity configuration in place.

    Args:
        **kwargs: Fields of SparsityConfig to overwrite.

    Returns:
        SparsityConfig: The updated global configuration.
    """
    for key, value in kwargs.items():
        if not hasattr(CONFIG, key):
            raise KeyError(f"Unknown sparsity option: {key}")
        setattr(CONFIG, key, value)
    return CONFIG


def configure_from_env() -> SparsityConfig:
    """
    Populate the global configuration from PB_SPARSE_* environment variables.

    Driving the sweep from the environment keeps every experimental condition
    in the job script, so a run is reproducible from its Slurm submission
    alone and no source edit is needed between conditions.

    Recognised variables:
        PB_SPARSE_ENABLED     1/0
        PB_SPARSE_K           int
        PB_SPARSE_MODE        knn | random | seq | radius
        PB_SPARSE_SYMMETRIZE  1/0
        PB_SPARSE_TLO         float
        PB_SPARSE_THI         float
        PB_SPARSE_LAYERS      comma-separated ints, empty for all
        PB_SPARSE_SEED        int

    Returns:
        SparsityConfig: The updated global configuration.
    """
    import os

    def _flag(name, default):
        raw = os.environ.get(name)
        return default if raw is None else raw.strip() not in ("0", "", "false")

    CONFIG.enabled = _flag("PB_SPARSE_ENABLED", CONFIG.enabled)
    CONFIG.symmetrize = _flag("PB_SPARSE_SYMMETRIZE", CONFIG.symmetrize)
    CONFIG.shared = _flag("PB_SPARSE_SHARED", CONFIG.shared)
    CONFIG.k = float(os.environ.get("PB_SPARSE_K", CONFIG.k))
    CONFIG.k_shape = os.environ.get("PB_SPARSE_KSHAPE", CONFIG.k_shape)
    CONFIG.k_hi = float(os.environ.get("PB_SPARSE_KHI", CONFIG.k_hi))
    CONFIG.k_lo = float(os.environ.get("PB_SPARSE_KLO", CONFIG.k_lo))
    CONFIG.k_frac = _flag("PB_SPARSE_KFRAC", CONFIG.k_frac)
    CONFIG.mode = os.environ.get("PB_SPARSE_MODE", CONFIG.mode)
    CONFIG.t_lo = float(os.environ.get("PB_SPARSE_TLO", CONFIG.t_lo))
    CONFIG.t_hi = float(os.environ.get("PB_SPARSE_THI", CONFIG.t_hi))
    CONFIG.seed = int(os.environ.get("PB_SPARSE_SEED", CONFIG.seed))
    CONFIG.eps = float(os.environ.get("PB_SPARSE_EPS", CONFIG.eps))
    CONFIG.term = os.environ.get("PB_SPARSE_TERM", CONFIG.term)
    CONFIG.alpha = float(os.environ.get("PB_SPARSE_ALPHA", CONFIG.alpha))
    CONFIG.target = os.environ.get("PB_SPARSE_TARGET", CONFIG.target)
    CONFIG.radius = float(os.environ.get("PB_SPARSE_RADIUS", CONFIG.radius))
    CONFIG.union = _flag("PB_SPARSE_UNION", CONFIG.union)
    if os.environ.get("PB_SPARSE_TRI_TLO"):
        CONFIG.tri_t_lo = float(os.environ["PB_SPARSE_TRI_TLO"])
    if os.environ.get("PB_SPARSE_TRI_THI"):
        CONFIG.tri_t_hi = float(os.environ["PB_SPARSE_TRI_THI"])
    raw_layers = os.environ.get("PB_SPARSE_LAYERS")
    if raw_layers is not None:
        CONFIG.layers = [int(x) for x in raw_layers.split(",") if x.strip()]
    return CONFIG


configure_from_env()


@contextlib.contextmanager
def timestep(t_frac: float, step_idx: int = 0):
    """
    Register the current diffusion timestep for the enclosed block.

    Args:
        t_frac: Timestep as a fraction of the total, 1.0 at the noisiest
            step and approaching 0.0 at the last.
        step_idx: Monotonic index of the sampling step, used to decorrelate
            the random-control neighborhoods across steps.
    """
    global _T_FRAC, _STEP_IDX
    prev_frac, prev_idx = _T_FRAC, _STEP_IDX
    _T_FRAC, _STEP_IDX = float(t_frac), int(step_idx)
    try:
        yield
    finally:
        _T_FRAC, _STEP_IDX = prev_frac, prev_idx


def current_timestep() -> Optional[float]:
    """
    Get the timestep fraction registered for the current sampling step.

    Returns:
        float | None: The fraction, or None if no timestep is registered.
    """
    return _T_FRAC


def neighborhood_mask(
    coords: torch.Tensor,
    valid: torch.Tensor,
    config: Optional[SparsityConfig] = None,
) -> torch.Tensor:
    """
    Build a binary neighborhood mask over residue pairs.

    Args:
        coords: [B, N, 3] residue coordinates from the current noisy
            structure (the translation component of the backbone frames).
        valid: [B, N] mask, nonzero for real residues and zero for padding.
        config: Configuration to use. Defaults to the global CONFIG.

    Returns:
        torch.Tensor: [B, N, N] mask, 1.0 where attention is permitted.
            Entry [b, i, j] permits query i to attend to key j. The diagonal
            is always 1.0 and padding columns are never selected.
    """
    cfg = config if config is not None else CONFIG
    B, N, _ = coords.shape
    device = coords.device
    valid_b = valid.bool()

    eye = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)

    if cfg.mode == "radius":
        # Everything within the cutoff, however many that is: the budget
        # follows the structure's density, not a fixed k. Symmetric by
        # construction.
        d = torch.cdist(coords.float(), coords.float())
        mask = ((d < cfg.radius) & valid_b.unsqueeze(-2) & valid_b.unsqueeze(-1)) | (eye & valid_b.unsqueeze(-1))
        return mask.to(coords.dtype)

    k_eff = min(cfg.k_at(_T_FRAC, int(valid_b.sum(-1).max())), N)
    if k_eff >= N:
        # Nothing to restrict; return an all-ones mask so that the caller
        # reproduces dense attention bit-for-bit.
        return torch.ones(B, N, N, device=device, dtype=coords.dtype)

    if cfg.mode in ("knn", "oracle"):
        score = torch.cdist(coords.float(), coords.float())
    elif cfg.mode == "seq":
        pos = torch.arange(N, device=device, dtype=torch.float32)
        score = (pos.unsqueeze(-1) - pos.unsqueeze(-2)).abs().unsqueeze(0)
        score = score.expand(B, N, N).contiguous()
    elif cfg.mode == "random":
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 1013 * _STEP_IDX)
        score = torch.rand(B, N, N, generator=gen).to(device)
    else:
        raise ValueError(f"Unknown neighborhood mode: {cfg.mode}")

    # Never select padding as a neighbor, and always rank self first so that
    # every row retains at least one finite entry.
    score = score.masked_fill(~valid_b.unsqueeze(-2), float("inf"))
    score = score.masked_fill(eye, -1.0)

    idx = score.topk(k_eff, dim=-1, largest=False).indices
    mask = torch.zeros(B, N, N, device=device, dtype=coords.dtype)
    mask.scatter_(-1, idx, 1.0)

    if cfg.symmetrize:
        mask = torch.maximum(mask, mask.transpose(-1, -2))

    # Padding rows attend to nothing and padding columns are attended by
    # nothing, including on the diagonal. The caller's validity mask already
    # zeroes these, but keeping them clean here means the neighborhood mask
    # alone satisfies the invariant and can be inspected on its own.
    valid_row = valid_b.unsqueeze(-1)
    mask = mask * valid_row
    mask = mask.masked_fill(eye & valid_row, 1.0)

    return mask


def apply_to_pair_mask(
    square_mask: torch.Tensor,
    coords: Optional[torch.Tensor],
    valid: torch.Tensor,
    layer_idx: Optional[int] = None,
) -> torch.Tensor:
    """
    Gate an existing pairwise validity mask by the neighborhood mask.

    This is the single hook used by IPA. When the restriction is inactive the
    input mask is returned unchanged, so the dense path is untouched.

    Args:
        square_mask: [B, N, N] pairwise validity mask, 1.0 where the pair is
            valid, as built by IPA from the 1D residue mask.
        coords: [B, N, 3] current noisy coordinates, or None when the caller
            has no frames available.
        valid: [B, N] residue mask.
        layer_idx: Index of the calling layer, for layer-selective schedules.

    Returns:
        torch.Tensor: [B, N, N] gated mask.
    """
    if coords is None or "ipa" not in CONFIG.target.split(",") or CONFIG.mode in ("thresh", "entmax") \
            or not CONFIG.active_at(_T_FRAC, layer_idx):
        return square_mask
    if CONFIG.mode == "oracle":
        coords = ORACLE
    return square_mask * neighborhood_mask(coords, valid)


def logit_threshold(a, geom, valid, layer_idx=None):
    """Per-head keep mask (B, H, N, N) for mode "thresh", or None when inactive."""
    if CONFIG.mode != "thresh" or "ipa" not in CONFIG.target.split(",") or not CONFIG.active_at(_T_FRAC, layer_idx):
        return None
    valid_b = valid.bool()
    N = a.shape[-1]
    # A key more than log(n/eps) below its row max has softmax weight under
    # eps/n, so everything dropped carries under eps of the row's mass. With
    # term="full" that bound holds for the attention actually used. With
    # term="geom" it holds for the distance term alone: the budget is still
    # adaptive, but chosen by geometry, which is what kNN approximates.
    if CONFIG.term == "full":
        s = a
    else:
        s = geom.masked_fill(~valid_b[:, None, None, :], float("-inf"))
    if CONFIG.shared:
        # One set for every head, like kNN; the bound then holds only for the
        # head-averaged logits, not for each head's attention.
        s = s.mean(1, keepdim=True)
    n = valid_b.sum(-1).float()[:, None, None, None]
    keep = s >= s.amax(-1, keepdim=True) - torch.log(n / CONFIG.eps)
    # Self is always kept, as in the kNN modes; adding keys cannot loosen the bound.
    eye = torch.eye(N, device=a.device, dtype=torch.bool)
    return keep | eye


_TRI_LOGGED = None


def triangle_gate(pair_mask):
    """Mediator gate [B, G+N, G+N] for a trunk triangle update, or None when inactive."""
    if "tri" not in CONFIG.target.split(",") or CONFIG.mode not in ("knn", "oracle", "random", "seq", "radius"):
        return None
    lo = CONFIG.t_lo if CONFIG.tri_t_lo is None else CONFIG.tri_t_lo
    hi = CONFIG.t_hi if CONFIG.tri_t_hi is None else CONFIG.tri_t_hi
    if not CONFIG.enabled or (_T_FRAC is not None and not (lo <= _T_FRAC <= hi)):
        return None
    coords = ORACLE if CONFIG.mode == "oracle" else TRUNK_COORDS
    B, N = coords.shape[:2]
    G = pair_mask.shape[-1] - N  # global tokens are prepended and stay fully connected
    valid = pair_mask.diagonal(dim1=-2, dim2=-1)[:, G:]
    m = torch.ones_like(pair_mask)
    nb = neighborhood_mask(coords, valid)
    m[:, G:, G:] = nb
    record_triangle(nb, valid)
    # The caller masks a_ik by m_ik and b_jk by m_jk (m is symmetric). Both
    # masked: k must neighbour i and j. With CONFIG.union: i or j.
    return m


@torch.no_grad()
def record_triangle(nb, valid):
    """Mean residue mediators per residue pair i != j, once per sampling step."""
    global _TRI_LOGGED
    if not RECORD or _TRI_LOGGED == (_STEP_IDX, _T_FRAC):
        return
    if RECORD_STRIDE > 1 and (_STEP_IDX % RECORD_STRIDE) != 0:
        return
    _TRI_LOGGED = (_STEP_IDX, _T_FRAC)
    v = valid.float()
    both = torch.matmul(nb, nb.transpose(-1, -2))
    deg = nb.sum(-1)
    either = deg.unsqueeze(-1) + deg.unsqueeze(-2) - both
    pair = v.unsqueeze(-1) * v.unsqueeze(-2) * (1 - torch.eye(nb.shape[-1], device=nb.device))
    mean = lambda x: float((x * pair).sum() / pair.sum().clamp_min(1))
    _RECORDS.append({
        "target": "tri", "step": _STEP_IDX, "t": _T_FRAC, "n": int(v.sum(-1).max()),
        "mode": CONFIG.mode, "union": CONFIG.union,
        "med_and": mean(both), "med_or": mean(either),
        "med_kept": mean(either if CONFIG.union else both),
    })


ENTMAX_ITERS = 50


def entmax(a, layer_idx=None):
    """alpha-entmax over keys (B, H, N, N) for mode "entmax", or None when inactive."""
    if CONFIG.mode != "entmax" or "ipa" not in CONFIG.target.split(",") or not CONFIG.active_at(_T_FRAC, layer_idx):
        return None
    al = CONFIG.alpha
    # Softmax-trained logits used as-is: the direct conversion of AdaSplash-2
    # Table 6, with no temperature fitted. Bisection on the threshold within
    # the Peters et al. bracket; N <= 500 so there is no need for their kernels.
    z = (al - 1) * a.float()
    mx = z.amax(-1, keepdim=True)
    lo, hi = mx - 1, mx - a.shape[-1] ** (1 - al)
    for _ in range(ENTMAX_ITERS):
        tau = (lo + hi) / 2
        f = (z - tau).clamp_min(0).pow(1 / (al - 1)).sum(-1, keepdim=True) - 1
        lo, hi = torch.where(f > 0, tau, lo), torch.where(f > 0, hi, tau)
    # lo never exceeds the true threshold, so the support can only be a
    # superset of the exact one (their Prop. 1); renormalise the residual.
    p = (z - lo).clamp_min(0).pow(1 / (al - 1))
    return (p / p.sum(-1, keepdim=True)).to(a.dtype)


# ---------------------------------------------------------------------------
# Phase 1 instrumentation: how local is dense attention already?
# ---------------------------------------------------------------------------

RECORD: bool = False
RECORD_KS: List[int] = [8, 16, 32, 64, 128]
RECORD_STRIDE: int = 1
_RECORDS: List[dict] = []

# ---------------------------------------------------------------------------
# Step-by-step trace: what the neighborhood rule keeps and what it discards
# ---------------------------------------------------------------------------
#
# The locality recorder above answers "how much mass is within k neighbors".
# The trace answers the next question: *which* mass the rule discards, where
# that mass sits, and which residues lose it. Run it on a dense trajectory and
# the discarded mass is counterfactual - what k-NN would have removed had it
# been on; run it on a restricted trajectory and it is the residual, which
# should be ~0 and serves as a check.

TRACE: bool = False
TRACE_STRIDE: int = 1
_TRACES: List[dict] = []
# Angstrom and |i - j| bin edges. The last bin of each is open.
DIST_EDGES = [0.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0, 40.0]
SEP_EDGES = [1, 2, 4, 8, 16, 32, 64, 128]


@contextlib.contextmanager
def tracing(stride: int = 1):
    """Collect per-step, per-layer neighborhood traces for the enclosed block."""
    global TRACE, TRACE_STRIDE, _TRACES
    prev, prev_stride = TRACE, TRACE_STRIDE
    TRACE, TRACE_STRIDE = True, int(stride)
    _TRACES = []
    try:
        yield _TRACES
    finally:
        TRACE, TRACE_STRIDE = prev, prev_stride


def take_traces() -> List[dict]:
    """Drain the collected traces."""
    global _TRACES
    out, _TRACES = _TRACES, []
    return out


@torch.no_grad()
def _trace_step(a, coords, valid, layer_idx):
    """Mass the neighborhood rule keeps and drops at this step and layer."""
    if TRACE_STRIDE > 1 and (_STEP_IDX % TRACE_STRIDE) != 0:
        return
    valid_b = valid.bool()
    B, H, N, _ = a.shape
    rows = valid_b.unsqueeze(1).expand(B, H, N)

    # The neighborhood the current config would select here, whether or not it
    # is actually being applied on this trajectory.
    nb = neighborhood_mask(coords, valid).unsqueeze(1)
    kept = (a * nb).sum(-1)
    drop = a * (1 - nb)

    d = torch.cdist(coords.float(), coords.float()).unsqueeze(1)
    pos = torch.arange(N, device=a.device)
    sep = (pos.unsqueeze(-1) - pos.unsqueeze(-2)).abs().float()[None, None]

    def hist(x, edges):
        # Mass per bin, normalised by total dropped mass, so the bins say
        # where the loss came from rather than how big it was.
        tot = drop.sum().clamp_min(1e-12)
        out = []
        for i, lo in enumerate(edges):
            hi = edges[i + 1] if i + 1 < len(edges) else float("inf")
            out.append(round(float((drop * ((x >= lo) & (x < hi))).sum() / tot), 5))
        return out

    # Radius of the neighborhood: how far the rule actually reaches at this
    # step. It grows over sampling rather than shrinking - at t/T = 1 the
    # coordinates sit at prior scale, so 32 neighbours fit inside ~1 A and the
    # graph is spatially meaningless; it reaches physical scale (~18 A at
    # L = 100) only once the structure has expanded.
    reach = (d.squeeze(1) * nb.squeeze(1)).amax(-1)

    per_res = (1 - kept).mean(1)  # [B, N], dropped mass per residue
    _TRACES.append({
        "layer": layer_idx,
        "step": _STEP_IDX,
        "t_frac": _T_FRAC,
        "n_res": int(valid_b.sum(-1).max()),
        "k": CONFIG.k_at(_T_FRAC, int(valid_b.sum(-1).max())),
        "keys": round(float((nb.squeeze(1).sum(-1) * valid_b).sum() / valid_b.sum().clamp_min(1)), 2),
        "kept": round(float((kept * rows).sum() / rows.sum().clamp_min(1)), 5),
        "kept_by_head": [round(float(x), 4) for x in
                         ((kept * rows).sum(-1) / rows.sum(-1).clamp_min(1)).mean(0)],
        "reach": round(float((reach * valid_b).sum() / valid_b.sum().clamp_min(1)), 2),
        "drop_by_dist": hist(d, DIST_EDGES),
        "drop_by_sep": hist(sep, SEP_EDGES),
        "drop_per_res": [round(float(x), 4) for x in per_res[0][: int(valid_b[0].sum())]],
    })


def recording_enabled_from_env() -> Optional[str]:
    """
    Check whether Phase 1 recording was requested via the environment.

    Recording is expensive (a distance sort per IPA call), so it is opt-in and
    strided rather than always on.

    Recognised variables:
        PB_RECORD_OUT     path to a JSONL file; enables recording when set
        PB_RECORD_STRIDE  record every Nth sampling step (default 10)
        PB_RECORD_KS      comma-separated neighborhood sizes to measure

    Returns:
        str | None: Output path if recording is enabled, else None.
    """
    import os

    global RECORD, RECORD_STRIDE, RECORD_KS
    out = os.environ.get("PB_RECORD_OUT")
    if not out:
        return None
    RECORD = True
    RECORD_STRIDE = int(os.environ.get("PB_RECORD_STRIDE", 10))
    raw_ks = os.environ.get("PB_RECORD_KS")
    if raw_ks:
        RECORD_KS = [int(x) for x in raw_ks.split(",") if x.strip()]
    return out


@contextlib.contextmanager
def recording(ks: Optional[List[int]] = None):
    """
    Record attention locality statistics for the enclosed block.

    Args:
        ks: Neighborhood sizes at which to measure captured attention mass.
    """
    global RECORD, RECORD_KS, _RECORDS
    prev_record, prev_ks = RECORD, RECORD_KS
    RECORD = True
    if ks is not None:
        RECORD_KS = list(ks)
    _RECORDS = []
    try:
        yield _RECORDS
    finally:
        RECORD, RECORD_KS = prev_record, prev_ks


def take_records() -> List[dict]:
    """
    Drain the recorded statistics.

    Returns:
        list[dict]: One entry per recorded IPA call.
    """
    global _RECORDS
    out, _RECORDS = _RECORDS, []
    return out


@torch.no_grad()
def record_attention(
    a: torch.Tensor,
    coords: Optional[torch.Tensor],
    valid: torch.Tensor,
    layer_idx: Optional[int] = None,
    parts: Optional[dict] = None,
    mask_logits: Optional[torch.Tensor] = None,
) -> None:
    """
    Measure how much attention mass falls inside spatial neighborhoods.

    This answers the question the masking experiment presupposes: before
    restricting anything, how much of the dense model's attention is already
    spatially local, and how does that change over the trajectory?

    Args:
        a: [B, H, N, N] post-softmax attention weights.
        coords: [B, N, 3] current noisy coordinates, or None.
        valid: [B, N] residue mask.
        layer_idx: Index of the calling layer.
    """
    if TRACE and coords is not None:
        _trace_step(a, coords, valid, layer_idx)
    if not RECORD or coords is None:
        return
    if RECORD_STRIDE > 1 and (_STEP_IDX % RECORD_STRIDE) != 0:
        return

    valid_b = valid.bool()
    B, H, N, _ = a.shape
    n_real = valid_b.sum(-1)

    # Entropy of each query's attention, averaged over real residues/heads.
    p = a.clamp_min(1e-12)
    ent = -(p * p.log()).sum(-1)
    row_valid = valid_b.unsqueeze(1).expand(B, H, N)
    mean_ent = (ent * row_valid).sum() / row_valid.sum().clamp_min(1)

    dist = torch.cdist(coords.float(), coords.float())
    dist = dist.masked_fill(~valid_b.unsqueeze(-2), float("inf"))
    eye = torch.eye(N, device=a.device, dtype=torch.bool).unsqueeze(0)
    dist = dist.masked_fill(eye, -1.0)
    order = dist.argsort(dim=-1)

    # Per head as well as pooled: a mean over 12 heads cannot distinguish
    # "all heads moderately local" from "half global, half pinned", and those
    # imply different things about what a sparse architecture could drop.
    ent_h = (ent * row_valid).sum(-1) / row_valid.sum(-1).clamp_min(1)  # (B, H)
    eff_h = ent_h.exp().mean(0)                                         # (H,)

    entry = {
        "layer": layer_idx,
        "t_frac": _T_FRAC,
        "step": _STEP_IDX,
        "n_res": float(n_real.float().mean()),
        "attn_entropy": float(mean_ent),
        "eff_neighbors": float(mean_ent.exp()),
        "eff_by_head": [round(float(x), 3) for x in eff_h],
        # exp(mean entropy) is a geometric mean over (residue, head), so a few
        # global heads get buried by many local ones. The arithmetic version
        # keeps them visible; reporting both shows the heterogeneity.
        "eff_arith": float(((ent.exp() * row_valid).sum() / row_valid.sum().clamp_min(1))),
    }

    # Keys with nonzero weight. Masked logits sit ~1e5 down and underflow to
    # exactly 0, so under a hard restriction this is the kept set; it is how
    # the adaptive modes report the budget they chose.
    sup = (a > 0).sum(-1).float()                                       # (B, H, N)
    sup_h = (sup * row_valid).sum(-1) / row_valid.sum(-1).clamp_min(1)  # (B, H)
    entry["support"] = float((sup * row_valid).sum() / row_valid.sum().clamp_min(1))
    entry["support_by_head"] = [round(float(x), 2) for x in sup_h.mean(0)]

    # Each logit term softmaxed alone, on the real noised trajectory. The
    # distance term is architectural and present at init; the content term is
    # learned. Without separating them, a broad early curve cannot distinguish
    # the trained model overriding the geometric prior from that prior simply
    # being dormant at prior-scale coordinates.
    if parts is not None:
        # All 8 subsets of {content, pair, geom}, mask always on. Standalone
        # and leave-one-out bound the interaction from opposite sides: a term
        # alone versus a term absent. They agree only if it acts additively,
        # and the full attention here is sharper than every standalone term,
        # which already says they do not. Shapley over these 8 is then an
        # order-independent attribution. The empty subset is uniform over
        # valid residues and doubles as a sanity check.
        names = ["content", "pair", "geom"]
        terms = [parts[n] for n in names]
        for bits in range(8):
            sel = [i for i in range(3) if (bits >> i) & 1]
            logits = torch.zeros_like(a) if mask_logits is None else mask_logits.expand_as(a).clone()
            for i in sel:
                logits = logits + terms[i]
            q = torch.softmax(logits, dim=-1).clamp_min(1e-12)
            e = -(q * q.log()).sum(-1)
            mm = (e * row_valid).sum() / row_valid.sum().clamp_min(1)
            key = "".join(names[i][0] for i in sel) or "none"
            entry[f"eff_s_{key}"] = round(float(mm.exp()), 3)

    # Fraction of attention mass captured by the k nearest neighbors.
    for k in RECORD_KS:
        if k >= N:
            entry[f"mass_k{k}"] = 1.0
            continue
        idx = order[..., :k]
        gathered = a.gather(-1, idx.unsqueeze(1).expand(B, H, N, k))
        mass = gathered.sum(-1)
        entry[f"mass_k{k}"] = float(
            (mass * row_valid).sum() / row_valid.sum().clamp_min(1)
        )

    _RECORDS.append(entry)
