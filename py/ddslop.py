"""DDS texture writer with DXT1/DXT5 (BC1/BC3) compression and mipmap support.

Dependencies: numpy, Pillow
"""

import struct
import numpy as np
from PIL import Image

# ── DDS constants ──────────────────────────────────────────────────────────────

_DDSD_CAPS        = 0x1
_DDSD_HEIGHT      = 0x2
_DDSD_WIDTH       = 0x4
_DDSD_PIXELFORMAT = 0x1000
_DDSD_MIPMAPCOUNT = 0x20000
_DDSD_LINEARSIZE  = 0x80000

_DDPF_FOURCC = 0x4

_DDSCAPS_COMPLEX = 0x8
_DDSCAPS_TEXTURE = 0x1000
_DDSCAPS_MIPMAP  = 0x400000

_FOURCC_DXT1 = struct.unpack("<I", b"DXT1")[0]
_FOURCC_DXT5 = struct.unpack("<I", b"DXT5")[0]

# ── Lookup tables (allocated once) ────────────────────────────────────────────

# DXT1: palette positions along endpoint line at t = 0, 1/3, 2/3, 1
#   sorted by t: idx1(0) idx3(1/3) idx2(2/3) idx0(1)
_BC1_INDEX_MAP = np.array([1, 3, 2, 0], dtype=np.uint8)
_BC1_PACK_WEIGHTS = (np.uint32(1) << (2 * np.arange(16, dtype=np.uint32)))

# DXT5 alpha: 8 positions along endpoint line at t = 0, 1/7, …, 6/7, 1
#   sorted by t: idx1(0) idx7(1/7) idx6(2/7) … idx2(6/7) idx0(1)
_BC3A_INDEX_MAP = np.array([1, 7, 6, 5, 4, 3, 2, 0], dtype=np.uint8)
_BC3A_BYTE_SHIFTS = np.array([0, 8, 16, 24, 32, 40], dtype=np.uint64)
_BC3A_PACK_WEIGHTS = (np.uint64(1) << (3 * np.arange(16, dtype=np.uint64)))

# ── Global quality knobs ───────────────────────────────────────────────────────

# When True the 2-subset and 3-subset partition search functions replace the
# pure bounding-box range heuristic with a single PCA power iteration per
# subset.  The power-iteration score (sum of squared projections onto the
# approximate principal axis) is a better proxy for within-subset variance than
# the axis-aligned bbox diagonal, at a modest extra cost.
_PARTITION_PCA: bool = True

# ── Helpers ────────────────────────────────────────────────────────────────────

def _pack_565(rgb):
    """(n, 3) float32 [0-255] → (n,) uint16 RGB565."""
    r = np.clip(rgb[..., 0], 0, 255).astype(np.uint16)
    g = np.clip(rgb[..., 1], 0, 255).astype(np.uint16)
    b = np.clip(rgb[..., 2], 0, 255).astype(np.uint16)
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _unpack_565(c):
    """(n,) uint16 RGB565 → (n, 3) float32 [0-255]."""
    r = ((c >> 11) & 0x1F).astype(np.float32) * (255.0 / 31.0)
    g = ((c >> 5) & 0x3F).astype(np.float32) * (255.0 / 63.0)
    b = (c & 0x1F).astype(np.float32) * (255.0 / 31.0)
    return np.stack([r, g, b], axis=-1)


def _pad4(arr):
    """Pad H, W to multiples of 4 via edge replication."""
    h, w = arr.shape[:2]
    ph, pw = (-h) % 4, (-w) % 4
    if ph == 0 and pw == 0:
        return arr
    pad_width = [(0, ph), (0, pw)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, pad_width, mode="edge")


def _to_blocks(arr):
    """(H, W, C) uint8 → (n_blocks, 16, C); H, W must be multiples of 4."""
    h, w, c = arr.shape
    return (arr
            .reshape(h // 4, 4, w // 4, 4, c)
            .transpose(0, 2, 1, 3, 4)
            .reshape(-1, 16, c))

# ── Endpoint selection ────────────────────────────────────────────────────────

def _endpoints_bbox(pixels):
    """Bounding-box endpoint selection. pixels: (n,16,3) float32 → c0,c1 uint16."""
    return _pack_565(pixels.max(axis=1)), _pack_565(pixels.min(axis=1))

def _pca_endpoints(pixels, iterations=3, gd_iters=3, indexes=4):
    """PCA endpoint selection via power iteration.
    pixels: (n, 16, C) float32 → e0, e1: (n, C) float32."""
    C = pixels.shape[2]
    mean = pixels.mean(axis=1, keepdims=True)                    # (n, 1, C)
    centered = pixels - mean                                     # (n, 16, C)
    cov = np.einsum("npc,npd->ncd", centered, centered)          # (n, C, C)

    # Initial guess: span between first and last pixel (more robust than
    # bbox diagonal, which can be orthogonal to the principal axis)
    v = pixels[:, 0, :] - pixels[:, -1, :]                       # (n, C)
    # Fallback to bbox diagonal, then to unit axis
    bbox_d = pixels.max(axis=1) - pixels.min(axis=1)
    norm_v = np.linalg.norm(v, axis=1, keepdims=True)
    norm_b = np.linalg.norm(bbox_d, axis=1, keepdims=True)
    fallback = np.zeros(C, dtype=np.float32); fallback[0] = 1.0
    v = np.where(norm_v > 1e-10, v / np.maximum(norm_v, 1e-10),
        np.where(norm_b > 1e-10, bbox_d / np.maximum(norm_b, 1e-10), fallback))

    for _ in range(iterations):
        v = np.einsum("nij,nj->ni", cov, v)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.maximum(norm, 1e-10)

    # Project pixels onto the principal axis, reconstruct endpoints on-axis
    proj = (centered * v[:, None, :]).sum(axis=2)                # (n, 16)
    t_max = proj.max(axis=1, keepdims=True)                      # (n, 1)
    t_min = proj.min(axis=1, keepdims=True)                      # (n, 1)
    mean_color = mean.squeeze(1)                                 # (n, C)
    
    rmse = np.sqrt(np.mean((centered/255.0*2.0)**2, axis=1, keepdims=True))  # (n, 1)
    mae = np.mean(np.abs(centered/255.0*2.0)/(np.maximum(rmse, 0.000001)), axis=1, keepdims=True)       # (n, 1)
    ratio = rmse / (mae + 1e-8)
    #print(rmse)
    #print(".")
    #print(mae)
    #print(".")
    #print(ratio)
    #print(".")
    ratio = ratio.mean(axis=2)
    ratio = np.maximum(1.0 - ratio, 0.0)
    ratio = np.abs(ratio)
    ratio = np.minimum(ratio * 2.0, 1.0)
    #print(ratio)
    #print("-----")
    # DO NOT DELETE. FOR DEBUGGING.
    
    # -------------------------------------------------------------------------
    # 1-Dimensional Gradient Descent
    # Optimizes t0 and t1 directly on the principal axis using the compressor's
    # exact projection and quantization logic.
    # -------------------------------------------------------------------------
    t0 = t_max
    t1 = t_min
    
    cx = indexes - 1.0
    
    # Outlier inset: move endpoints towards the mean (1D squish)
    t0 -= t0 / cx / 8.0 * ratio
    t1 -= t1 / cx / 8.0 * ratio
    
    lr = 2.0  # Learning rate

    for _ in range(gd_iters):
        # 1. Project each pixel onto the t1→t0 line; t_val∈[0,1]
        range_t = t0 - t1
        range_t = np.where(np.abs(range_t) > 1e-10, range_t, 1e-10)
        
        t_val = np.clip((proj - t1) / range_t, 0, 1)                 # (n, 16)
        
        # 2. Quantise to 4 palette slots using the compressor's exact rounding
        # t_idx goes from 0 (t1) to 3 (t0)
        t_idx = np.clip((t_val * cx + 0.5).astype(np.int32), 0, int(cx))  # (n, 16)
        
        # Convert integer index to continuous interpolation weights
        w_t0 = (t_idx / cx).astype(np.float32)                      # (n, 16)
        w_t1 = 1.0 - w_t0                                            # (n, 16)
        
        # 3. Reconstruct chosen 1D projections
        chosen_t = t0 * w_t0 + t1 * w_t1                             # (n, 16)
        
        # 4. Compute 1D residuals
        r = chosen_t - proj                                          # (n, 16)
        
        # 5. Gradients: Pull t0 and t1 to minimize squared error
        grad_t0 = (r * w_t0).mean(axis=1, keepdims=True)             # (n, 1)
        grad_t1 = (r * w_t1).mean(axis=1, keepdims=True)             # (n, 1)
        
        t0 -= lr * grad_t0
        t1 -= lr * grad_t1

    # Reconstruct final 3D endpoints from the 1D scalars
    e0 = mean_color + t0 * v
    e1 = mean_color + t1 * v

    return e0, e1

def _endpoints_pca(pixels, iterations=4):
    """PCA endpoint selection for BC1. pixels: (n,16,3) float32 → c0,c1 uint16."""
    e0, e1 = _pca_endpoints(pixels, iterations)
    return _pack_565(e0), _pack_565(e1)

# ── BC1 (DXT1) color compression ──────────────────────────────────────────────

def _compress_bc1(blocks, pca=0):
    """(n, 16, 3) uint8 → (n, 8) uint8 BC1 encoded."""
    n = blocks.shape[0]
    pixels = blocks.astype(np.float32)

    # Endpoint selection
    if pca:
        c0, c1 = _endpoints_pca(pixels, iterations=pca)
    else:
        c0, c1 = _endpoints_bbox(pixels)

    # Ensure c0 > c1 (selects 4-color mode)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)

    # Force c0 > c1 even for solid-color blocks
    eq = c0 == c1
    c0 = np.where(eq & (c0 < 0xFFFF), c0 + 1, c0).astype(np.uint16)
    still_eq = c0 == c1
    c1 = np.where(still_eq,
                   np.maximum(c1.astype(np.int32) - 1, 0),
                   c1.astype(np.int32)).astype(np.uint16)

    # Reconstruct endpoints from 565 for accurate palette matching
    e0 = _unpack_565(c0)  # (n, 3)
    e1 = _unpack_565(c1)  # (n, 3)

    # Project each pixel onto the e1→e0 line; t∈[0,1]
    direction = e0 - e1                                         # (n, 3)
    len_sq = np.maximum(
        (direction * direction).sum(axis=1, keepdims=True),     # (n, 1)
        1e-10)
    t = np.clip(
        ((pixels - e1[:, None, :]) * direction[:, None, :])
            .sum(axis=2) / len_sq,                              # (n, 16)
        0, 1)

    # Quantise to 4 palette slots → DXT index
    indices = _BC1_INDEX_MAP[
        np.clip((t * 3 + 0.5).astype(np.int32), 0, 3)]        # (n, 16)

    # Pack 16×2-bit indices into one uint32 per block
    packed = (indices.astype(np.uint32) *
              _BC1_PACK_WEIGHTS[None, :]).sum(axis=1).astype(np.uint32)

    out = np.empty((n, 8), dtype=np.uint8)
    out[:, 0] = c0 & 0xFF
    out[:, 1] = c0 >> 8
    out[:, 2] = c1 & 0xFF
    out[:, 3] = c1 >> 8
    out[:, 4] = packed & 0xFF
    out[:, 5] = (packed >> 8) & 0xFF
    out[:, 6] = (packed >> 16) & 0xFF
    out[:, 7] = (packed >> 24) & 0xFF
    return out

# ── BC3 (DXT5) alpha compression ──────────────────────────────────────────────

def _compress_bc3_alpha(blocks):
    """(n, 16) uint8 → (n, 8) uint8 BC3 alpha encoded."""
    n = blocks.shape[0]

    a0 = blocks.max(axis=1).astype(np.int16)  # (n,)
    a1 = blocks.min(axis=1).astype(np.int16)

    # Ensure a0 > a1 (selects 8-interpolant mode)
    eq = a0 == a1
    a0 = np.where(eq, np.minimum(a0 + 1, 255), a0)
    still_eq = a0 == a1
    a1 = np.where(still_eq, np.maximum(a1 - 1, 0), a1)

    range_a = np.maximum((a0 - a1).astype(np.float32), 1.0)
    t = np.clip(
        (blocks.astype(np.float32) - a1[:, None].astype(np.float32))
            / range_a[:, None],
        0, 1)

    indices = _BC3A_INDEX_MAP[
        np.clip((t * 7 + 0.5).astype(np.int32), 0, 7)]

    # Pack 16×3-bit indices into 48 bits
    packed = (indices.astype(np.uint64) *
              _BC3A_PACK_WEIGHTS[None, :]).sum(axis=1)

    out = np.empty((n, 8), dtype=np.uint8)
    out[:, 0] = a0.astype(np.uint8)
    out[:, 1] = a1.astype(np.uint8)
    out[:, 2:8] = ((packed[:, None] >> _BC3A_BYTE_SHIFTS[None, :])
                    & np.uint64(0xFF)).astype(np.uint8)
    return out

# ── BC7 (BPTC) compression ────────────────────────────────────────────────────
#
# Implements modes 0, 1, 2, 3, 4, 5, 6 — covering 3-subset RGB (P-bit and plain),
# opaque 2-subset, precise 2-subset, separate-alpha (5-bit+6-bit and 7-bit+8-bit),
# and single-partition RGBA.

# 2-subset partition table (64 partitions × 16 pixels)
_BC7_P2 = np.array([int(c) for c in (
    "0011001100110011" "0001000100010001" "0111011101110111" "0001001100110111"
    "0000000100010011" "0011011101111111" "0001001101111111" "0000000100110111"
    "0000000000010011" "0011011111111111" "0000000101111111" "0000000000010111"
    "0001011111111111" "0000000011111111" "0000111111111111" "0000000000001111"
    "0000100011101111" "0111000100000000" "0000000010001110" "0111001100010000"
    "0011000100000000" "0000100011001110" "0000000010001100" "0111001100110001"
    "0011000100010000" "0000100010001100" "0110011001100110" "0011011001101100"
    "0001011111101000" "0000111111110000" "0111000110001110" "0011100110011100"
    "0101010101010101" "0000111100001111" "0101101001011010" "0011001111001100"
    "0011110000111100" "0101010110101010" "0110100101101001" "0101101010100101"
    "0111001111001110" "0001001111001000" "0011001001001100" "0011101111011100"
    "0110100110010110" "0011110011000011" "0110011010011001" "0000011001100000"
    "0100111001000000" "0010011100100000" "0000001001110010" "0000010011100100"
    "0110110010010011" "0011011011001001" "0110001110011100" "0011100111000110"
    "0110110011001001" "0110001100111001" "0111111010000001" "0001100011100111"
    "0000111100110011" "0011001111110000" "0010001011101110" "0100010001110111"
)], dtype=np.uint8).reshape(64, 16)

# 3-subset partition table (64 × 16) — needed for anchor table validation
_BC7_P3 = np.array([int(c) for c in (
    "0011001102212222" "0001001122112221" "0000200122112211" "0222002200110111"
    "0000000011221122" "0011001100220022" "0022002211111111" "0011001122112211"
    "0000000011112222" "0000111111112222" "0000111122222222" "0012001200120012"
    "0112011201120112" "0122012201220122" "0011011211221222" "0011200122002220"
    "0001001101121122" "0111001120012200" "0000112211221122" "0022002200221111"
    "0111011102220222" "0001000122212221" "0000001101220122" "0000110022102210"
    "0122012200110000" "0012001211222222" "0110122112210110" "0000011012211221"
    "0022110211020022" "0110011020022222" "0011012201220011" "0000200022112221"
    "0000000211221222" "0222002200120011" "0011001200220222" "0120012001200120"
    "0000111122220000" "0120120120120120" "0120201212010120" "0011220011220011"
    "0011112222000011" "0101010122222222" "0000000021212121" "0022112200221122"
    "0022001100220011" "0220122102201221" "0101222222220101" "0000212121212121"
    "0101010101012222" "0222011102220111" "0002111200021112" "0000211221122112"
    "0222011101110222" "0002111211120002" "0110011001102222" "0000000021122112"
    "0110011022222222" "0022001100110022" "0022112211220022" "0000000000002112"
    "0002000100020001" "0222122202221222" "0101222222222222" "0111201122012220"
)], dtype=np.uint8).reshape(64, 16)

# Anchor index for subset 1 in 2-subset modes (indexed by partition ID)
_BC7_A2 = np.array([
    15,15,15,15,15,15,15,15, 15,15,15,15,15,15,15,15,
    15, 2, 8, 2, 2, 8, 8,15,  2, 8, 2, 2, 8, 8, 2, 2,
    15,15, 6, 8, 2, 8,15,15,  2, 8, 2, 2, 2,15,15, 6,
     6, 2, 6, 8,15,15, 2, 2, 15,15,15,15,15, 2, 2,15,
], dtype=np.uint8)

# Anchor indices for 3-subset modes (subset 1 and subset 2)
_BC7_A3_2 = np.array([
     3, 3,15,15, 8, 3,15,15,  8, 8, 6, 6, 6, 5, 3, 3,
     3, 3, 8,15, 3, 3, 6,10,  5, 8, 8, 6, 8, 5,15,15,
     8,15, 3, 5, 6,10, 8,15, 15, 3,15, 5,15,15,15,15,
     3,15, 5, 5, 5, 8, 5,10,  5,10, 8,13,15,12, 3, 3,
], dtype=np.uint8)

_BC7_A3_3 = np.array([
    15, 8, 8, 3,15,15, 3, 8, 15,15,15,15,15,15,15, 8,
    15, 8,15, 3,15, 8,15, 8,  3,15, 6,10,15,15,10, 8,
    15, 3,15,10,10, 8, 9,10,  6,15, 8,15, 3, 6, 6, 8,
    15, 3,15,15,15,15,15,15, 15,15,15,15, 3,15,15, 8,
], dtype=np.uint8)

# Interpolation weights (integer, out of 64)
_BC7_W2 = np.array([0, 21, 43, 64], dtype=np.int32)
_BC7_W3 = np.array([0, 9, 18, 27, 37, 46, 55, 64], dtype=np.int32)
_BC7_W4 = np.array([0, 4, 9, 13, 17, 21, 26, 30,
                     34, 38, 43, 47, 51, 55, 60, 64], dtype=np.int32)

# Searchsorted boundaries (normalised to [0,1]) for index assignment
_BC7_B2 = (_BC7_W2[:-1] + _BC7_W2[1:]).astype(np.float64) / 128.0
_BC7_B3 = (_BC7_W3[:-1] + _BC7_W3[1:]).astype(np.float64) / 128.0
_BC7_B4 = (_BC7_W4[:-1] + _BC7_W4[1:]).astype(np.float64) / 128.0

# ── BC7 quantisation tables ──────────────────────────────────────────────────
# Dequant: N-bit value → 8-bit value   (bit replication)
# Quant:   8-bit value → N-bit value   (nearest match)

_BC7_DEQUANT = {}
_BC7_QUANT = {}
for _p in (5, 6, 7, 8):
    _v = np.arange(1 << _p, dtype=np.uint16)
    _d = ((_v << (8 - _p)) | (_v >> (2 * _p - 8))) & 0xFF
    _BC7_DEQUANT[_p] = _d.astype(np.uint8)
    _t = np.arange(256, dtype=np.int16)[:, None]
    _BC7_QUANT[_p] = np.abs(_t - _d.astype(np.int16)[None, :]).argmin(axis=1).astype(np.uint8)

# P-bit–aware tables: key = (color_bits, p_bit)
_BC7_DEQUANT_PB = {}
_BC7_QUANT_PB = {}
for _cb in (4, 5, 6, 7):
    _eff = _cb + 1
    for _pb in (0, 1):
        _cv = np.arange(1 << _cb, dtype=np.uint16)
        _ev = (_cv << 1) | _pb
        _d = ((_ev << (8 - _eff)) | (_ev >> (2 * _eff - 8))) & 0xFF
        _BC7_DEQUANT_PB[(_cb, _pb)] = _d.astype(np.uint8)
        _t = np.arange(256, dtype=np.int16)[:, None]
        _BC7_QUANT_PB[(_cb, _pb)] = np.abs(_t - _d.astype(np.int16)[None, :]).argmin(axis=1).astype(np.uint8)

# ── BC7 bit-packing helpers ───────────────────────────────────────────────────

def _bw(lo, hi, off, nb, vals):
    """Write *nb* bits of *vals* into (lo, hi) uint64 arrays at bit *off*."""
    if nb == 0:
        return
    v = vals.astype(np.uint64) & ((np.uint64(1) << np.uint64(nb)) - np.uint64(1))
    o = int(off)
    if o + nb <= 64:
        lo |= v << np.uint64(o)
    elif o >= 64:
        hi |= v << np.uint64(o - 64)
    else:
        lo |= v << np.uint64(o)
        hi |= v >> np.uint64(64 - o)


def _bc7_pack(lo, hi):
    """(n,) uint64 pair → (n, 16) uint8 block bytes."""
    n = lo.shape[0]
    out = np.empty((n, 16), dtype=np.uint8)
    for i in range(8):
        s = np.uint64(i * 8)
        out[:, i]     = (lo >> s) & np.uint64(0xFF)
        out[:, 8 + i] = (hi >> s) & np.uint64(0xFF)
    return out


def _bc7_project_idx(pixels, e0, e1, boundaries):
    """Project pixels onto e0→e1 line and quantise to index via searchsorted.
    pixels: (n,16,C) float32, e0/e1: (n,C) float32.
    Returns (n,16) uint8 indices."""
    d = e1 - e0                                                         # (n, C)
    lsq = np.maximum((d * d).sum(axis=1, keepdims=True), np.float32(1e-10))
    t = np.clip(
        ((pixels - e0[:, None, :]) * d[:, None, :]).sum(axis=2) / lsq,  # (n,16)
        0, 1)
    return np.searchsorted(boundaries, t.ravel()).reshape(pixels.shape[:2]).astype(np.uint8)


def _bc7_error(pixels, e0, e1, indices, weights):
    """Compute per-block SSE using integer-precise interpolation.
    pixels (n,16,C) float32, e0/e1 (n,C) float32, indices (n,16) uint8."""
    w = weights[indices.astype(np.int32)].astype(np.float32)             # (n,16)
    recon = (np.float32(64) - w[:,:,None]) * e0[:,None,:] + \
             w[:,:,None] * e1[:,None,:]
    recon = np.floor((recon + 32) / 64)
    return ((pixels - recon) ** 2).sum(axis=(1, 2))

# ── BC7 Mode 6  (1 subset, RGBA 7+1=8bit, 4-bit idx) ────────────────────────

def _bc7_mode6(pixels, pca=0):
    """pixels: (n,16,4) float32.  Returns (blocks, error)."""
    n = pixels.shape[0]

    # Endpoint selection — always use ≥1 PCA iteration for BC7 (bbox alone
    # produces off-axis endpoints that are catastrophic at 4-bit indexing)
    e0r, e1r = _pca_endpoints(pixels, iterations=max(pca, 1), indexes=16)

    # Quantise: 7 colour bits + 1 unique P-bit = full 8-bit
    # Try both P-bit values, pick lowest total error per endpoint
    def _q6(e):
        e = np.clip(np.round(e), 0, 255).astype(np.int32)       # (n, 4)
        best_c = np.empty((n, 4), np.uint8)
        best_p = np.empty(n, np.uint8)
        best_e = np.full(n, np.float32(1e30))
        for pb in range(2):
            c7 = np.clip(((e - pb + 1) >> 1), 0, 127).astype(np.uint8)
            rec = (c7.astype(np.int32) << 1) | pb
            err = ((e - rec).astype(np.float32) ** 2).sum(axis=1)
            better = err < best_e
            best_c[better] = c7[better]
            best_p[better] = pb
            best_e[better] = err[better]
        return best_c, best_p
    c7_0, p0 = _q6(e0r);  c7_1, p1 = _q6(e1r)

    # Reconstruct 8-bit endpoints for palette
    r0 = (c7_0.astype(np.int32) << 1 | p0[:,None]).astype(np.float32)
    r1 = (c7_1.astype(np.int32) << 1 | p1[:,None]).astype(np.float32)

    # Index assignment
    indices = _bc7_project_idx(pixels, r0, r1, _BC7_B4)

    # Anchor: pixel 0, 4-bit → MSB threshold 8
    swap = indices[:, 0] >= 8
    indices[swap] = np.uint8(15) - indices[swap]
    c7_0, c7_1 = (np.where(swap[:,None], c7_1, c7_0),
                   np.where(swap[:,None], c7_0, c7_1))
    p0, p1 = np.where(swap, p1, p0), np.where(swap, p0, p1)
    r0 = (c7_0.astype(np.int32) << 1 | p0[:,None]).astype(np.float32)
    r1 = (c7_1.astype(np.int32) << 1 | p1[:,None]).astype(np.float32)

    error = _bc7_error(pixels, r0, r1, indices, _BC7_W4)

    # Pack 128 bits
    lo = np.zeros(n, np.uint64); hi = np.zeros(n, np.uint64)
    _bw(lo, hi, 0, 7, np.full(n, 1 << 6, np.uint8))              # mode 6
    off = 7
    for ch in range(4):                                            # R0 R1 G0 G1 B0 B1 A0 A1
        _bw(lo, hi, off, 7, c7_0[:, ch]); off += 7
        _bw(lo, hi, off, 7, c7_1[:, ch]); off += 7
    _bw(lo, hi, off, 1, p0); off += 1
    _bw(lo, hi, off, 1, p1); off += 1
    _bw(lo, hi, off, 3, indices[:, 0]); off += 3                   # anchor (3 bits)
    for i in range(1, 16):
        _bw(lo, hi, off, 4, indices[:, i]); off += 4
    assert off == 128
    return _bc7_pack(lo, hi), error

# ── BC7 Mode 5  (1 subset, RGB 7bit + A 8bit, separate 2-bit idx) ────────────

def _bc7_mode5(pixels, pca=0):
    """pixels: (n,16,4) float32.  Returns (blocks, error)."""
    n = pixels.shape[0]
    rgb = pixels[:,:,:3]; alpha = pixels[:,:,3]

    # Colour endpoints (7-bit, no P-bit)
    e0c, e1c = _pca_endpoints(rgb, iterations=max(pca, 1), indexes=4)
    c7_0 = _BC7_QUANT[7][np.clip(np.round(e0c), 0, 255).astype(np.int32)]
    c7_1 = _BC7_QUANT[7][np.clip(np.round(e1c), 0, 255).astype(np.int32)]
    dq0 = _BC7_DEQUANT[7][c7_0].astype(np.float32)
    dq1 = _BC7_DEQUANT[7][c7_1].astype(np.float32)

    # Alpha endpoints (8-bit, exact)
    #a0 = np.clip(np.round(alpha.max(axis=1)), 0, 255).astype(np.uint8)
    #a1 = np.clip(np.round(alpha.min(axis=1)), 0, 255).astype(np.uint8)
    a0 = np.clip(np.round(alpha.min(axis=1)), 0, 255).astype(np.uint8)
    a1 = np.clip(np.round(alpha.max(axis=1)), 0, 255).astype(np.uint8)

    # Colour indices (2-bit)
    ci = _bc7_project_idx(rgb, dq0, dq1, _BC7_B2)
    swap_c = ci[:, 0] >= 2
    ci[swap_c] = np.uint8(3) - ci[swap_c]
    c7_0, c7_1 = (np.where(swap_c[:,None], c7_1, c7_0),
                   np.where(swap_c[:,None], c7_0, c7_1))
    dq0 = _BC7_DEQUANT[7][c7_0].astype(np.float32)
    dq1 = _BC7_DEQUANT[7][c7_1].astype(np.float32)

    # Alpha indices (2-bit)
    a_range = np.maximum(a1.astype(np.float32) - a0.astype(np.float32), np.float32(1e-10))
    t_a = np.clip((alpha - a0[:,None].astype(np.float32)) / a_range[:,None], 0, 1)
    ai = np.searchsorted(_BC7_B2, t_a.ravel()).reshape(n, 16).astype(np.uint8)
    swap_a = ai[:, 0] >= 2
    ai[swap_a] = np.uint8(3) - ai[swap_a]
    a0, a1 = np.where(swap_a, a1, a0), np.where(swap_a, a0, a1)

    # Error
    cerr = _bc7_error(rgb, dq0, dq1, ci, _BC7_W2)
    w_a = _BC7_W2[ai.astype(np.int32)].astype(np.float32)
    recon_a = np.floor((np.float32(64) - w_a) * a0[:,None].astype(np.float32) +
                        w_a * a1[:,None].astype(np.float32) + 32) / 64
    aerr = ((alpha - recon_a) ** 2).sum(axis=1)
    error = cerr + aerr

    # Pack
    lo = np.zeros(n, np.uint64); hi = np.zeros(n, np.uint64)
    _bw(lo, hi, 0, 6, np.full(n, 1 << 5, np.uint8))              # mode 5
    _bw(lo, hi, 6, 2, np.zeros(n, np.uint8))                      # rotation=0
    off = 8
    for ch in range(3):
        _bw(lo, hi, off, 7, c7_0[:, ch]); off += 7
        _bw(lo, hi, off, 7, c7_1[:, ch]); off += 7
    _bw(lo, hi, off, 8, a0); off += 8
    _bw(lo, hi, off, 8, a1); off += 8
    _bw(lo, hi, off, 1, ci[:, 0]); off += 1                       # colour anchor
    for i in range(1, 16):
        _bw(lo, hi, off, 2, ci[:, i]); off += 2
    _bw(lo, hi, off, 1, ai[:, 0]); off += 1                       # alpha anchor
    for i in range(1, 16):
        _bw(lo, hi, off, 2, ai[:, i]); off += 2
    assert off == 128
    return _bc7_pack(lo, hi), error

# ── BC7 Mode 4  (1 subset, RGB 5bit + A 6bit, 2-bit color / 3-bit alpha idx) ──

def _bc7_mode4_0(pixels, pca=0):
    """pixels: (n,16,4) float32.  Returns (blocks, error).

    Mode 4 encodes colour at 5-bit precision with 2-bit indices (4 levels) and
    alpha at 6-bit precision with 3-bit indices (8 levels), using separate index
    streams.  The higher alpha index resolution makes it competitive with mode 5
    on blocks that have gradual alpha transitions and relatively flat colour.
    Rotation and index-selection are fixed to 0 (no channel swap, primary stream
    carries colour).
    """
    n = pixels.shape[0]
    rgb   = pixels[:, :, :3]
    alpha = pixels[:, :, 3]

    # ── Colour endpoints: 5-bit, no P-bit, 2-bit indices (4 palette levels) ──
    e0c, e1c = _pca_endpoints(rgb, iterations=max(pca, 1), indexes=4)
    c5_0 = _BC7_QUANT[5][np.clip(np.round(e0c), 0, 255).astype(np.int32)]  # (n,3) uint8
    c5_1 = _BC7_QUANT[5][np.clip(np.round(e1c), 0, 255).astype(np.int32)]
    dq0  = _BC7_DEQUANT[5][c5_0].astype(np.float32)
    dq1  = _BC7_DEQUANT[5][c5_1].astype(np.float32)

    # ── Alpha endpoints: 6-bit, no P-bit, 3-bit indices (8 palette levels) ──
    a0_q = _BC7_QUANT[6][np.clip(np.round(alpha.min(axis=1)), 0, 255).astype(np.int32)]  # (n,) uint8
    a1_q = _BC7_QUANT[6][np.clip(np.round(alpha.max(axis=1)), 0, 255).astype(np.int32)]
    dqa0 = _BC7_DEQUANT[6][a0_q].astype(np.float32)  # (n,) float32 (8-bit after dequant)
    dqa1 = _BC7_DEQUANT[6][a1_q].astype(np.float32)

    # ── Colour indices (2-bit); anchor = pixel 0, MSB suppressed → threshold 2 ──
    ci = _bc7_project_idx(rgb, dq0, dq1, _BC7_B2)
    swap_c = ci[:, 0] >= 2
    ci[swap_c] = np.uint8(3) - ci[swap_c]
    c5_0, c5_1 = (np.where(swap_c[:, None], c5_1, c5_0),
                  np.where(swap_c[:, None], c5_0, c5_1))
    dq0 = _BC7_DEQUANT[5][c5_0].astype(np.float32)
    dq1 = _BC7_DEQUANT[5][c5_1].astype(np.float32)

    # ── Alpha indices (3-bit); anchor = pixel 0, MSB suppressed → threshold 4 ──
    # Reshape alpha to (n,16,1) so _bc7_project_idx handles it uniformly.
    ai = _bc7_project_idx(alpha[:, :, None], dqa0[:, None], dqa1[:, None], _BC7_B3)
    swap_a = ai[:, 0] >= 4
    ai[swap_a] = np.uint8(7) - ai[swap_a]
    a0_q, a1_q = (np.where(swap_a, a1_q, a0_q),
                  np.where(swap_a, a0_q, a1_q))
    dqa0 = _BC7_DEQUANT[6][a0_q].astype(np.float32)
    dqa1 = _BC7_DEQUANT[6][a1_q].astype(np.float32)

    # ── Error ──────────────────────────────────────────────────────────────────
    cerr  = _bc7_error(rgb, dq0, dq1, ci, _BC7_W2)
    w_a   = _BC7_W3[ai.astype(np.int32)].astype(np.float32)          # (n, 16)
    recon_a = np.floor((np.float32(64) - w_a) * dqa0[:, None] +
                        w_a * dqa1[:, None] + 32) / 64
    aerr  = ((alpha - recon_a) ** 2).sum(axis=1)
    error = cerr + aerr

    # ── Pack 128 bits ──────────────────────────────────────────────────────────
    # Layout:
    #   [0:4]   mode indicator – bit 4 set (5 bits, value = 0b10000)
    #   [5:6]   rotation = 0 (2 bits)
    #   [7]     indexSelection = 0 (1 bit)
    #   [8:37]  R0,R1,G0,G1,B0,B1 (5 bits each = 30 bits)
    #   [38:49] A0, A1 (6 bits each = 12 bits)
    #   [50:80] colour indices – anchor 1 bit + 15x2 bits = 31 bits
    #   [81:127] alpha indices – anchor 2 bits + 15x3 bits = 47 bits
    #   Total: 5+2+1+30+12+31+47 = 128
    lo = np.zeros(n, np.uint64); hi = np.zeros(n, np.uint64)
    _bw(lo, hi, 0, 5, np.full(n, 1 << 4, np.uint8))               # mode 4
    _bw(lo, hi, 5, 2, np.zeros(n, np.uint8))                       # rotation = 0
    _bw(lo, hi, 7, 1, np.zeros(n, np.uint8))                       # indexSelection = 0
    off = 8
    for ch in range(3):
        _bw(lo, hi, off, 5, c5_0[:, ch]); off += 5
        _bw(lo, hi, off, 5, c5_1[:, ch]); off += 5
    _bw(lo, hi, off, 6, a0_q); off += 6
    _bw(lo, hi, off, 6, a1_q); off += 6
    _bw(lo, hi, off, 1, ci[:, 0]); off += 1                        # colour anchor
    for i in range(1, 16):
        _bw(lo, hi, off, 2, ci[:, i]); off += 2
    _bw(lo, hi, off, 2, ai[:, 0]); off += 2                        # alpha anchor
    for i in range(1, 16):
        _bw(lo, hi, off, 3, ai[:, i]); off += 3
    assert off == 128
    return _bc7_pack(lo, hi), error


# ── BC7 Mode 4  (1 subset, RGB 5bit + A 6bit, 3-bit color / 2-bit alpha idx) ──
 
def _bc7_mode4_1(pixels, pca=0):
    """pixels: (n,16,4) float32.  Returns (blocks, error).
 
    indexSelection=1: the decoder swaps the two index streams, so the 2-bit
    primary stream in the bitstream carries alpha and the 3-bit secondary stream
    carries colour.  This gives colour 8 palette levels (better gradients) at
    the cost of only 4 alpha levels.  Rotation is fixed to 0 (no channel swap).
 
    Bitstream layout (identical positions regardless of indexSelection):
      [0:4]   mode indicator – bit 4 set (5 bits, value = 0b10000)
      [5:6]   rotation = 0 (2 bits)
      [7]     indexSelection = 1 (1 bit)
      [8:37]  R0,R1,G0,G1,B0,B1 (5 bits each = 30 bits)
      [38:49] A0, A1 (6 bits each = 12 bits)
      [50:80] primary stream – alpha indices, 2-bit each, anchor 1 bit (31 bits)
      [81:127] secondary stream – colour indices, 3-bit each, anchor 2 bits (47 bits)
      Total: 5+2+1+30+12+31+47 = 128
    """
    n = pixels.shape[0]
    rgb   = pixels[:, :, :3]
    alpha = pixels[:, :, 3]
 
    # ── Colour endpoints: 5-bit, no P-bit, 3-bit indices (8 palette levels) ──
    e0c, e1c = _pca_endpoints(rgb, iterations=max(pca, 1), indexes=8)
    c5_0 = _BC7_QUANT[5][np.clip(np.round(e0c), 0, 255).astype(np.int32)]  # (n,3) uint8
    c5_1 = _BC7_QUANT[5][np.clip(np.round(e1c), 0, 255).astype(np.int32)]
    dq0  = _BC7_DEQUANT[5][c5_0].astype(np.float32)
    dq1  = _BC7_DEQUANT[5][c5_1].astype(np.float32)
 
    # ── Alpha endpoints: 6-bit, no P-bit, 2-bit indices (4 palette levels) ──
    a0_q = _BC7_QUANT[6][np.clip(np.round(alpha.min(axis=1)), 0, 255).astype(np.int32)]  # (n,) uint8
    a1_q = _BC7_QUANT[6][np.clip(np.round(alpha.max(axis=1)), 0, 255).astype(np.int32)]
    dqa0 = _BC7_DEQUANT[6][a0_q].astype(np.float32)  # (n,) float32 (8-bit after dequant)
    dqa1 = _BC7_DEQUANT[6][a1_q].astype(np.float32)
 
    # ── Alpha indices (2-bit) → primary stream; anchor pixel 0, threshold 2 ──
    ai = _bc7_project_idx(alpha[:, :, None], dqa0[:, None], dqa1[:, None], _BC7_B2)
    swap_a = ai[:, 0] >= 2
    ai[swap_a] = np.uint8(3) - ai[swap_a]
    a0_q, a1_q = (np.where(swap_a, a1_q, a0_q),
                  np.where(swap_a, a0_q, a1_q))
    dqa0 = _BC7_DEQUANT[6][a0_q].astype(np.float32)
    dqa1 = _BC7_DEQUANT[6][a1_q].astype(np.float32)
 
    # ── Colour indices (3-bit) → secondary stream; anchor pixel 0, threshold 4 ──
    ci = _bc7_project_idx(rgb, dq0, dq1, _BC7_B3)
    swap_c = ci[:, 0] >= 4
    ci[swap_c] = np.uint8(7) - ci[swap_c]
    c5_0, c5_1 = (np.where(swap_c[:, None], c5_1, c5_0),
                  np.where(swap_c[:, None], c5_0, c5_1))
    dq0 = _BC7_DEQUANT[5][c5_0].astype(np.float32)
    dq1 = _BC7_DEQUANT[5][c5_1].astype(np.float32)
 
    # ── Error ──────────────────────────────────────────────────────────────────
    cerr  = _bc7_error(rgb, dq0, dq1, ci, _BC7_W3)
    w_a   = _BC7_W2[ai.astype(np.int32)].astype(np.float32)          # (n, 16)
    recon_a = np.floor((np.float32(64) - w_a) * dqa0[:, None] +
                        w_a * dqa1[:, None] + 32) / 64
    aerr  = ((alpha - recon_a) ** 2).sum(axis=1)
    error = cerr + aerr
 
    # ── Pack 128 bits ──────────────────────────────────────────────────────────
    lo = np.zeros(n, np.uint64); hi = np.zeros(n, np.uint64)
    _bw(lo, hi, 0, 5, np.full(n, 1 << 4, np.uint8))               # mode 4
    _bw(lo, hi, 5, 2, np.zeros(n, np.uint8))                       # rotation = 0
    _bw(lo, hi, 7, 1, np.ones(n,  np.uint8))                       # indexSelection = 1
    off = 8
    for ch in range(3):
        _bw(lo, hi, off, 5, c5_0[:, ch]); off += 5
        _bw(lo, hi, off, 5, c5_1[:, ch]); off += 5
    _bw(lo, hi, off, 6, a0_q); off += 6
    _bw(lo, hi, off, 6, a1_q); off += 6
    _bw(lo, hi, off, 1, ai[:, 0]); off += 1                        # alpha anchor (primary)
    for i in range(1, 16):
        _bw(lo, hi, off, 2, ai[:, i]); off += 2
    _bw(lo, hi, off, 2, ci[:, 0]); off += 2                        # colour anchor (secondary)
    for i in range(1, 16):
        _bw(lo, hi, off, 3, ci[:, i]); off += 3
    assert off == 128
    return _bc7_pack(lo, hi), error
 

# ── BC7 2-subset partition search (shared by Mode 1 & 3) ─────────────────────

def _bc7_best_partition(pixels_rgb, n, lite=False, nano=False):
    """Partition search → (n,) best partition ID.

    Scoring uses either the bbox-range heuristic or (when the global
    ``_PARTITION_PCA`` flag is set) a single PCA power iteration per subset.
    Both aim to minimise within-subset variance; the PCA path is more accurate
    because it scores variance along the dominant colour axis rather than the
    axis-aligned bounding-box diagonal."""
    best_err = np.full(n, np.float32(1e30))
    best_pid = np.zeros(n, np.uint8)
    for pid in range(64) if not lite and not nano else [0, 13, 7, 19, 26, 29, 15, 14, 1, 2] if not nano else [0, 13]:
        part = _BC7_P2[pid]
        m0 = part == 0; m1 = ~m0
        e = np.float32(0)
        for m in (m0, m1):
            sub = pixels_rgb[:, m, :]                                   # (n, cnt, 3)
            if _PARTITION_PCA:
                mean = sub.mean(axis=1, keepdims=True)                  # (n, 1, 3)
                cent = sub - mean                                        # (n, cnt, 3)
                # Initial direction: bbox diagonal (robust to degenerate spans)
                bbox_d = sub.max(axis=1) - sub.min(axis=1)              # (n, 3)
                norm   = np.linalg.norm(bbox_d, axis=1, keepdims=True)
                v      = np.where(norm > np.float32(1e-10),
                                  bbox_d / np.maximum(norm, np.float32(1e-10)),
                                  np.array([1., 0., 0.], np.float32))
                # One power iteration: v ← cov @ v, then renormalise
                cov  = np.einsum("npc,npd->ncd", cent, cent)            # (n, 3, 3)
                v    = np.einsum("nij,nj->ni",   cov,  v)               # (n, 3)
                norm = np.linalg.norm(v, axis=1, keepdims=True)
                v    = v / np.maximum(norm, np.float32(1e-10))
                # Score: off-axis residual = total variance − variance along
                # principal axis.  BC7 encodes each subset as a 1-D line, so
                # the residual is what cannot be represented; minimise it to
                # pick the partition where subsets lie most along a single line.
                proj     = (cent * v[:, None, :]).sum(axis=2)           # (n, cnt)
                total_sq = (cent ** 2).sum(axis=(1, 2))                 # (n,)
                e        = e + total_sq - (proj ** 2).sum(axis=1)
            else:
                mx = sub.max(axis=1)                                    # (n, 3)
                mn = sub.min(axis=1)
                e  = e + ((mx - mn) ** 2).sum(axis=1)
        better = e < best_err
        best_pid[better] = pid
        best_err[better] = e[better]
    return best_pid

# ── BC7 Mode 1  (2 subsets, RGB 6+1=7bit, 3-bit idx, shared P-bit) ──────────

def _bc7_encode_2subset(pixels, mode, pca=0, best_pid=None, lite=False, nano=False):
    """Encode blocks using a 2-subset mode.
    mode 1: color_bits=6, shared P-bit, 3-bit indices
    mode 3: color_bits=7, unique P-bits, 2-bit indices
    pixels: (n,16,4) float32.  Returns (blocks, error)."""
    n = pixels.shape[0]
    rgb = pixels[:,:,:3]

    if best_pid is None:
        best_pid = _bc7_best_partition(rgb, n)

    out_blocks = np.zeros((n, 16), np.uint8)
    out_error  = np.full(n, np.float32(1e30))

    if mode == 1:
        cb, shared_p, idx_bits = 6, True, 3
        weights, bounds = _BC7_W3, _BC7_B3
        mode_val, mode_bits = 1 << 1, 2
        max_idx = 7
    else:  # mode 3
        cb, shared_p, idx_bits = 7, False, 2
        weights, bounds = _BC7_W2, _BC7_B2
        mode_val, mode_bits = 1 << 3, 4
        max_idx = 3

    anchor_thresh = 1 << (idx_bits - 1)
    max_c = (1 << cb) - 1
    big = np.float32(1e10)

    for pid in range(64):
        sel = best_pid == pid
        if not sel.any():
            continue
        pix = rgb[sel]                                            # (k,16,3)
        pix_full = pixels[sel]
        k = pix.shape[0]
        part = _BC7_P2[pid]; m0 = part == 0; m1 = ~m0
        anchor1 = int(_BC7_A2[pid])

        # Subset endpoints via _pca_endpoints (multi-iteration PCA + 1-D GD)
        ep = []                                                   # [(e0,e1), (e0,e1)]
        for m in (m0, m1):
            sub = pix[:, m, :]                                    # (k, m_count, 3)
            ehi, elo = _pca_endpoints(sub, iterations=max(pca, 1), indexes=max_idx + 1)
            ep.append((ehi, elo))

        # Quantise & dequantise endpoints, choose P-bits
        c_list = []   # 4 arrays of (k,3) uint8  (c_s0_0, c_s0_1, c_s1_0, c_s1_1)
        pb_list = []  # P-bit per subset (or per endpoint)
        dq_list = []  # dequantised 8-bit  (4 × (k,3) float32)

        for si, (ehi, elo) in enumerate(ep):
            if shared_p:
                # Try both P-bit values; pick better for both endpoints combined
                best_sp = np.zeros(k, np.uint8)
                best_sp_err = np.full(k, np.float32(1e30))
                best_c0 = np.zeros((k,3), np.uint8)
                best_c1 = np.zeros((k,3), np.uint8)
                for pb in range(2):
                    qt = _BC7_QUANT_PB[(cb, pb)]
                    c0 = qt[np.clip(np.round(ehi), 0, 255).astype(np.int32)]
                    c1 = qt[np.clip(np.round(elo), 0, 255).astype(np.int32)]
                    dqt = _BC7_DEQUANT_PB[(cb, pb)]
                    r0 = dqt[c0].astype(np.float32)
                    r1 = dqt[c1].astype(np.float32)
                    err = ((ehi - r0)**2).sum(axis=1) + ((elo - r1)**2).sum(axis=1)
                    bt = err < best_sp_err
                    best_sp[bt] = pb; best_sp_err[bt] = err[bt]
                    best_c0[bt] = c0[bt]; best_c1[bt] = c1[bt]
                pb_list.append(best_sp)
                c_list.extend([best_c0, best_c1])
                dqt0 = np.empty((k,3), np.float32)
                dqt1 = np.empty((k,3), np.float32)
                for pb in range(2):
                    mask_pb = best_sp == pb
                    if mask_pb.any():
                        dqt = _BC7_DEQUANT_PB[(cb, pb)]
                        dqt0[mask_pb] = dqt[best_c0[mask_pb]].astype(np.float32)
                        dqt1[mask_pb] = dqt[best_c1[mask_pb]].astype(np.float32)
                dq_list.extend([dqt0, dqt1])
            else:
                # Unique P-bits per endpoint
                def _quant_ep(ev):
                    ev_i = np.clip(np.round(ev), 0, 255).astype(np.int32)
                    best_c = np.empty((k,3), np.uint8)
                    best_pb = np.empty(k, np.uint8)
                    best_err = np.full(k, np.float32(1e30))
                    for pb in range(2):
                        c = _BC7_QUANT_PB[(cb, pb)][ev_i]
                        dq = _BC7_DEQUANT_PB[(cb, pb)][c].astype(np.float32)
                        err = ((ev - dq)**2).sum(axis=1)
                        bt = err < best_err
                        best_c[bt] = c[bt]; best_pb[bt] = pb
                        best_err[bt] = err[bt]
                    return best_c, best_pb
                c0, pb0 = _quant_ep(ep[si][0])
                c1, pb1 = _quant_ep(ep[si][1])
                c_list.extend([c0, c1])
                pb_list.extend([pb0, pb1])
                dqt0 = np.empty((k,3), np.float32)
                dqt1 = np.empty((k,3), np.float32)
                for pb in range(2):
                    m0p = pb0 == pb
                    if m0p.any():
                        dqt0[m0p] = _BC7_DEQUANT_PB[(cb, pb)][c0[m0p]].astype(np.float32)
                    m1p = pb1 == pb
                    if m1p.any():
                        dqt1[m1p] = _BC7_DEQUANT_PB[(cb, pb)][c1[m1p]].astype(np.float32)
                dq_list.extend([dqt0, dqt1])

        # dq_list = [dq_s0_0, dq_s0_1, dq_s1_0, dq_s1_1]  each (k,3)
        # Build per-pixel base / endpoint from subset membership
        base  = np.where(m0[None,:,None], dq_list[0][:,None,:], dq_list[2][:,None,:])
        end   = np.where(m0[None,:,None], dq_list[1][:,None,:], dq_list[3][:,None,:])
        d = end - base                                            # (k,16,3)
        lsq = np.maximum((d * d).sum(axis=2), np.float32(1e-10)) # (k,16)
        t = np.clip(((pix - base) * d).sum(axis=2) / lsq, 0, 1)
        indices = np.searchsorted(bounds, t.ravel()).reshape(k, 16).astype(np.uint8)

        # Anchor handling — subset 0 anchor is pixel 0, subset 1 anchor is anchor1
        swap0 = indices[:, 0] >= anchor_thresh
        swap1 = indices[:, anchor1] >= anchor_thresh
        inv = np.uint8(max_idx)
        for i in range(16):
            is_s0 = m0[i]
            sw = swap0 if is_s0 else swap1
            indices[:, i] = np.where(sw, inv - indices[:, i], indices[:, i])

        # Swap endpoints where needed
        c0_s0, c1_s0, c0_s1, c1_s1 = c_list
        dq0_s0, dq1_s0, dq0_s1, dq1_s1 = dq_list
        c0_s0, c1_s0 = (np.where(swap0[:,None], c1_s0, c0_s0),
                         np.where(swap0[:,None], c0_s0, c1_s0))
        dq0_s0, dq1_s0 = (np.where(swap0[:,None], dq1_s0, dq0_s0),
                           np.where(swap0[:,None], dq0_s0, dq1_s0))
        c0_s1, c1_s1 = (np.where(swap1[:,None], c1_s1, c0_s1),
                         np.where(swap1[:,None], c0_s1, c1_s1))
        dq0_s1, dq1_s1 = (np.where(swap1[:,None], dq1_s1, dq0_s1),
                           np.where(swap1[:,None], dq0_s1, dq1_s1))
        # P-bits don't change for shared (symmetric); for unique, swap
        if not shared_p:
            pb_list[0], pb_list[1] = (np.where(swap0, pb_list[1], pb_list[0]),
                                      np.where(swap0, pb_list[0], pb_list[1]))
            pb_list[2], pb_list[3] = (np.where(swap1, pb_list[3], pb_list[2]),
                                      np.where(swap1, pb_list[2], pb_list[3]))

        # Error (RGB only; alpha forced to 255 → add alpha mismatch)
        base_f = np.where(m0[None,:,None], dq0_s0[:,None,:], dq0_s1[:,None,:])
        end_f  = np.where(m0[None,:,None], dq1_s0[:,None,:], dq1_s1[:,None,:])
        w = weights[indices.astype(np.int32)].astype(np.float32)
        recon = np.floor((np.float32(64) - w[:,:,None]) * base_f +
                          w[:,:,None] * end_f + 32) / 64
        rgb_err = ((pix - recon) ** 2).sum(axis=(1, 2))
        a_err = ((pix_full[:,:,3] - 255) ** 2).sum(axis=1)
        error = rgb_err + a_err

        # Pack
        lo = np.zeros(k, np.uint64); hi = np.zeros(k, np.uint64)
        _bw(lo, hi, 0, mode_bits, np.full(k, mode_val, np.uint8))
        off = mode_bits
        _bw(lo, hi, off, 6, np.full(k, pid, np.uint8)); off += 6
        # Endpoints: R0 R1 R2 R3 / G / B
        eps = [c0_s0, c1_s0, c0_s1, c1_s1]
        for ch in range(3):
            for ep in eps:
                _bw(lo, hi, off, cb, ep[:, ch]); off += cb
        # P-bits
        if shared_p:
            _bw(lo, hi, off, 1, pb_list[0]); off += 1  # subset 0
            _bw(lo, hi, off, 1, pb_list[1]); off += 1  # subset 1
        else:
            for pb_arr in pb_list:                       # 4 unique P-bits
                _bw(lo, hi, off, 1, pb_arr); off += 1
        # Indices
        for i in range(16):
            is_anchor = (i == 0) or (i == anchor1)
            nb = idx_bits - 1 if is_anchor else idx_bits
            _bw(lo, hi, off, nb, indices[:, i]); off += nb
        assert off == 128, f"mode {mode} bits = {off}"

        out_blocks[sel] = _bc7_pack(lo, hi)
        out_error[sel]  = error

    return out_blocks, out_error

# ── BC7 3-subset partition search ────────────────────────────────────────────

def _bc7_best_partition_3subset(pixels_rgb, n, lite=False):
    """Partition search → (best_pid_small, best_pid) for 3-subset modes.

    Scoring uses either the bbox-range heuristic or (when the global
    ``_PARTITION_PCA`` flag is set) a single PCA power iteration per subset.
    See ``_bc7_best_partition`` for rationale."""
    big = np.float32(1e10)  # kept for any future use; not used in loops below
    best_err = np.full(n, np.float32(1e30))
    best_pid_small = np.zeros(n, np.uint8)
    best_pid = np.zeros(n, np.uint8)

    pid_range = range(64) if not lite else [1, 3, 4, 9, 12, 14, 15]
    for pid in pid_range:
        part = _BC7_P3[pid]
        e = np.float32(0)
        for s in range(3):
            ms = part == s
            sub = pixels_rgb[:, ms, :]                                  # (n, cnt, 3)
            if _PARTITION_PCA:
                mean = sub.mean(axis=1, keepdims=True)                  # (n, 1, 3)
                cent = sub - mean                                        # (n, cnt, 3)
                bbox_d = sub.max(axis=1) - sub.min(axis=1)              # (n, 3)
                norm   = np.linalg.norm(bbox_d, axis=1, keepdims=True)
                v      = np.where(norm > np.float32(1e-10),
                                  bbox_d / np.maximum(norm, np.float32(1e-10)),
                                  np.array([1., 0., 0.], np.float32))
                cov  = np.einsum("npc,npd->ncd", cent, cent)            # (n, 3, 3)
                v    = np.einsum("nij,nj->ni",   cov,  v)               # (n, 3)
                norm = np.linalg.norm(v, axis=1, keepdims=True)
                v    = v / np.maximum(norm, np.float32(1e-10))
                proj     = (cent * v[:, None, :]).sum(axis=2)           # (n, cnt)
                total_sq = (cent ** 2).sum(axis=(1, 2))                 # (n,)
                e        = e + total_sq - (proj ** 2).sum(axis=1)       # off-axis residual
            else:
                mx = sub.max(axis=1)                                    # (n, 3)
                mn = sub.min(axis=1)
                e  = e + ((mx - mn) ** 2).sum(axis=1)
        better = e < best_err
        best_pid[better] = pid
        best_err[better] = e[better]
        if pid < 16:
            best_pid_small[better] = pid
    return best_pid_small, best_pid


# ── BC7 Modes 0 & 2  (3 subsets, RGB only) ───────────────────────────────────

def _bc7_encode_3subset(pixels, mode, pca=0, best_pid=None, lite=False):
    """Encode blocks using a 3-subset RGB mode.
    mode 0: color_bits=4, unique P-bits per endpoint, 3-bit indices
    mode 2: color_bits=5, no P-bits, 2-bit indices
    pixels: (n,16,4) float32.  Returns (blocks, error)."""
    n = pixels.shape[0]
    rgb = pixels[:, :, :3]

    if best_pid is None:
        best_pid, _ = _bc7_best_partition_3subset(rgb, n, lite=lite)

    out_blocks = np.zeros((n, 16), np.uint8)
    out_error  = np.full(n, np.float32(1e30))

    if mode == 0:
        cb, has_p, idx_bits = 4, True, 3
        weights, bounds     = _BC7_W3, _BC7_B3
        mode_val, mode_bits, part_bits = 1 << 0, 1, 4
        max_idx = 7
    else:  # mode 2
        cb, has_p, idx_bits = 5, False, 2
        weights, bounds     = _BC7_W2, _BC7_B2
        mode_val, mode_bits, part_bits = 1 << 2, 3, 6
        max_idx = 3

    anchor_thresh = 1 << (idx_bits - 1)

    for pid in range(64):
        sel = best_pid == pid
        if not sel.any():
            continue
        pix      = rgb[sel]        # (k, 16, 3)
        pix_full = pixels[sel]
        k        = pix.shape[0]
        part     = _BC7_P3[pid]
        anchor   = [0, int(_BC7_A3_2[pid]), int(_BC7_A3_3[pid])]

        # --- Subset endpoint estimation via _pca_endpoints (multi-iteration PCA + 1-D GD) ---
        ep = []  # 3 × (ehi, elo)  each (k, 3)
        for s in range(3):
            ms  = part == s
            sub = pix[:, ms, :]                                         # (k, count_s, 3)
            ehi, elo = _pca_endpoints(sub, iterations=max(pca, 1), indexes=max_idx + 1)
            ep.append((ehi, elo))

        # --- Quantise endpoints ---
        # c_list / dq_list indexed as [s0_ep0, s0_ep1, s1_ep0, s1_ep1, s2_ep0, s2_ep1]
        c_list  = []  # 6 × (k, 3) uint8
        pb_list = []  # 6 × (k,) uint8   (mode 0 only)
        dq_list = []  # 6 × (k, 3) float32

        for s, (ehi, elo) in enumerate(ep):
            if has_p:
                def _quant_ep(ev, k=k, cb=cb):
                    ev_i   = np.clip(np.round(ev), 0, 255).astype(np.int32)
                    best_c  = np.empty((k, 3), np.uint8)
                    best_pb = np.zeros(k, np.uint8)
                    best_e  = np.full(k, np.float32(1e30))
                    for pb in range(2):
                        c   = _BC7_QUANT_PB[(cb, pb)][ev_i]
                        dq  = _BC7_DEQUANT_PB[(cb, pb)][c].astype(np.float32)
                        err = ((ev - dq) ** 2).sum(axis=1)
                        bt  = err < best_e
                        best_c[bt] = c[bt]; best_pb[bt] = pb; best_e[bt] = err[bt]
                    return best_c, best_pb
                c0, pb0 = _quant_ep(ehi)
                c1, pb1 = _quant_ep(elo)
                c_list.extend([c0, c1])
                pb_list.extend([pb0, pb1])
                dqt0 = np.empty((k, 3), np.float32)
                dqt1 = np.empty((k, 3), np.float32)
                for pb in range(2):
                    m0p = pb0 == pb
                    if m0p.any():
                        dqt0[m0p] = _BC7_DEQUANT_PB[(cb, pb)][c0[m0p]].astype(np.float32)
                    m1p = pb1 == pb
                    if m1p.any():
                        dqt1[m1p] = _BC7_DEQUANT_PB[(cb, pb)][c1[m1p]].astype(np.float32)
                dq_list.extend([dqt0, dqt1])
            else:
                c0  = _BC7_QUANT[cb][np.clip(np.round(ehi), 0, 255).astype(np.int32)]
                c1  = _BC7_QUANT[cb][np.clip(np.round(elo), 0, 255).astype(np.int32)]
                dq0 = _BC7_DEQUANT[cb][c0].astype(np.float32)
                dq1 = _BC7_DEQUANT[cb][c1].astype(np.float32)
                c_list.extend([c0, c1])
                dq_list.extend([dq0, dq1])

        # --- Per-pixel base/end arrays from subset membership ---
        dq_base = np.empty((k, 16, 3), np.float32)
        dq_end  = np.empty((k, 16, 3), np.float32)
        for s in range(3):
            ms = part == s
            dq_base[:, ms, :] = dq_list[2*s    ][:, None, :]
            dq_end [:, ms, :] = dq_list[2*s + 1][:, None, :]

        d   = dq_end - dq_base
        lsq = np.maximum((d * d).sum(axis=2), np.float32(1e-10))
        t   = np.clip(((pix - dq_base) * d).sum(axis=2) / lsq, 0, 1)
        indices = np.searchsorted(bounds, t.ravel()).reshape(k, 16).astype(np.uint8)

        # --- Anchor handling: force MSB=0 at each subset's anchor pixel ---
        swaps = []
        for s in range(3):
            anc = anchor[s]
            sw  = indices[:, anc] >= anchor_thresh
            swaps.append(sw)
            ms  = part == s
            for i in range(16):
                if ms[i]:
                    indices[:, i] = np.where(sw,
                                             np.uint8(max_idx) - indices[:, i],
                                             indices[:, i])

        # --- Swap endpoints to match corrected index orientation ---
        for s in range(3):
            sw = swaps[s]
            c_list[2*s], c_list[2*s+1] = (
                np.where(sw[:, None], c_list[2*s+1], c_list[2*s  ]),
                np.where(sw[:, None], c_list[2*s  ], c_list[2*s+1]))
            dq_list[2*s], dq_list[2*s+1] = (
                np.where(sw[:, None], dq_list[2*s+1], dq_list[2*s  ]),
                np.where(sw[:, None], dq_list[2*s  ], dq_list[2*s+1]))
            if has_p:
                pb_list[2*s], pb_list[2*s+1] = (
                    np.where(sw, pb_list[2*s+1], pb_list[2*s  ]),
                    np.where(sw, pb_list[2*s  ], pb_list[2*s+1]))

        # --- Recompute per-pixel palette after swap, then measure SSE ---
        for s in range(3):
            ms = part == s
            dq_base[:, ms, :] = dq_list[2*s    ][:, None, :]
            dq_end [:, ms, :] = dq_list[2*s + 1][:, None, :]
        w     = weights[indices.astype(np.int32)].astype(np.float32)
        recon = np.floor((np.float32(64) - w[:, :, None]) * dq_base +
                          w[:, :, None] * dq_end + 32) / 64
        rgb_err = ((pix           - recon) ** 2).sum(axis=(1, 2))
        a_err   = ((pix_full[:, :, 3] - 255) ** 2).sum(axis=1)
        error   = rgb_err + a_err

        # --- Pack 128-bit bitstream ---
        lo = np.zeros(k, np.uint64); hi = np.zeros(k, np.uint64)
        _bw(lo, hi, 0, mode_bits, np.full(k, mode_val, np.uint8))
        off = mode_bits
        _bw(lo, hi, off, part_bits, np.full(k, pid, np.uint8)); off += part_bits
        # Endpoints: R0..R5, then G0..G5, then B0..B5
        for ch in range(3):
            for ei in range(6):
                #ei = 5 - ei
                _bw(lo, hi, off, cb, c_list[ei][:, ch]); off += cb
        # P-bits: one per endpoint, 6 total (mode 0 only)
        if has_p:
            for pb_arr in pb_list:
                _bw(lo, hi, off, 1, pb_arr); off += 1
        # Indices: anchor pixels get (idx_bits - 1) bits, all others idx_bits
        for i in range(16):
            s_i       = int(part[i])
            is_anchor = (i == anchor[s_i])
            nb        = idx_bits - 1 if is_anchor else idx_bits
            _bw(lo, hi, off, nb, indices[:, i]); off += nb
        assert off == 128, f"mode {mode} pid {pid} bits = {off}"

        out_blocks[sel] = _bc7_pack(lo, hi)
        out_error[sel]  = error

    return out_blocks, out_error


# ── BC7 block selector ────────────────────────────────────────────────────────

def _compress_bc7(blocks_rgba, pca=0, lite=False, nano=False, zero=False):
    """(n,16,4) uint8 → (n,16) uint8 BC7-encoded blocks."""
    pf = blocks_rgba.astype(np.float32)

    best_blk, best_err = _bc7_mode6(pf, pca)
    
    blk4, err4 = _bc7_mode4_1(pf, pca)
    better = err4 < best_err
    best_blk[better] = blk4[better]; best_err[better] = err4[better]

    if zero: return best_blk
    
    blk4, err4 = _bc7_mode4_0(pf, pca)
    better = err4 < best_err
    best_blk[better] = blk4[better]; best_err[better] = err4[better]

    blk5, err5 = _bc7_mode5(pf, pca)
    better = err5 < best_err
    best_blk[better] = blk5[better]; best_err[better] = err5[better]

    # Share partition search between modes 1 and 3
    pid = _bc7_best_partition(pf[:,:,:3], pf.shape[0], lite=lite, nano=nano)
    
    blk1, err1 = _bc7_encode_2subset(pf, mode=1, pca=pca, best_pid=pid, lite=lite, nano=nano)
    better = err1 < best_err
    best_blk[better] = blk1[better]; best_err[better] = err1[better]
    
    blk3, err3 = _bc7_encode_2subset(pf, mode=3, pca=pca, best_pid=pid, lite=lite, nano=nano)
    better = err3 < best_err
    best_blk[better] = blk3[better]; best_err[better] = err3[better]

    if nano: return best_blk

    # 3-subset modes (0 and 2) — RGB only, alpha forced to 255 in error metric
    pid3, pid3big = _bc7_best_partition_3subset(pf[:, :, :3], pf.shape[0], lite=lite)

    blk0, err0 = _bc7_encode_3subset(pf, mode=0, pca=pca, best_pid=pid3, lite=lite)
    better = err0 < best_err
    best_blk[better] = blk0[better]; best_err[better] = err0[better]

    blk2, err2 = _bc7_encode_3subset(pf, mode=2, pca=pca, best_pid=pid3big, lite=lite)
    better = err2 < best_err
    best_blk[better] = blk2[better]; best_err[better] = err2[better]

    return best_blk

# ── Per-level compression ─────────────────────────────────────────────────────

def _compress_level(arr, pixel_format, pca=0):
    """Compress one mip level.  arr: (H, W, C) uint8 (any H/W)."""
    arr = _pad4(arr)

    if pixel_format == "BC7" or pixel_format == "BC7lite" or pixel_format == "BC7nano" or pixel_format == "BC7zero":
        if arr.shape[2] < 4:
            arr = np.concatenate([arr[..., :3],
                                  np.full(arr.shape[:2] + (1,), 255, np.uint8)], axis=2)
        return _compress_bc7(_to_blocks(arr[:, :, :4]), pca=pca, lite=pixel_format == "BC7lite", nano=pixel_format == "BC7nano", zero=pixel_format == "BC7zero").tobytes()

    rgb = arr[..., :3]
    rgb_blocks = _to_blocks(rgb)
    color = _compress_bc1(rgb_blocks, pca=pca)

    if pixel_format == "DXT1":
        return color.tobytes()

    # DXT5: alpha block (8 B) then color block (8 B) per 4×4 tile
    if arr.shape[2] >= 4:
        alpha_blocks = _to_blocks(arr[..., 3:4]).squeeze(-1)
    else:
        alpha_blocks = np.full(rgb_blocks.shape[:2], 255, dtype=np.uint8)
    alpha = _compress_bc3_alpha(alpha_blocks)
    return np.concatenate([alpha, color], axis=1).tobytes()

# ── DDS header ─────────────────────────────────────────────────────────────────

def _write_header(f, width, height, pixel_format, mip_count):
    flags = (_DDSD_CAPS | _DDSD_HEIGHT | _DDSD_WIDTH |
             _DDSD_PIXELFORMAT | _DDSD_LINEARSIZE)
    caps = _DDSCAPS_TEXTURE

    if mip_count > 1:
        flags |= _DDSD_MIPMAPCOUNT
        caps |= _DDSCAPS_COMPLEX | _DDSCAPS_MIPMAP

    block_size = 8 if pixel_format == "DXT1" else 16
    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    linear_size = bw * bh * block_size

    fourcc = _FOURCC_DXT1 if pixel_format == "DXT1" else _FOURCC_DXT5

    f.write(b"DDS ")
    f.write(struct.pack("<7I", 124, flags, height, width,
                        linear_size, 0, mip_count))
    f.write(b"\x00" * 44)                         # dwReserved1[11]
    f.write(struct.pack("<2I", 32, _DDPF_FOURCC))  # pixelformat size + flags
    f.write(struct.pack("<I", fourcc))
    f.write(b"\x00" * 20)                          # bit counts / masks
    f.write(struct.pack("<5I", caps, 0, 0, 0, 0))


def _write_header_dx10(f, width, height, dxgi_format, mip_count):
    """Write DDS header with DX10 extension (required for BC7)."""
    flags = (_DDSD_CAPS | _DDSD_HEIGHT | _DDSD_WIDTH |
             _DDSD_PIXELFORMAT | _DDSD_LINEARSIZE)
    caps = _DDSCAPS_TEXTURE
    if mip_count > 1:
        flags |= _DDSD_MIPMAPCOUNT
        caps |= _DDSCAPS_COMPLEX | _DDSCAPS_MIPMAP

    bw = max(1, (width + 3) // 4)
    bh = max(1, (height + 3) // 4)
    linear_size = bw * bh * 16

    f.write(b"DDS ")
    f.write(struct.pack("<7I", 124, flags, height, width,
                        linear_size, 0, mip_count))
    f.write(b"\x00" * 44)
    fourcc_dx10 = struct.unpack("<I", b"DX10")[0]
    f.write(struct.pack("<2I", 32, _DDPF_FOURCC))
    f.write(struct.pack("<I", fourcc_dx10))
    f.write(b"\x00" * 20)
    f.write(struct.pack("<5I", caps, 0, 0, 0, 0))
    # DX10 extended header (20 bytes)
    f.write(struct.pack("<5I",
        dxgi_format,   # 98 = DXGI_FORMAT_BC7_UNORM
        3,             # D3D10_RESOURCE_DIMENSION_TEXTURE2D
        0, 1, 0))

# ── Public API ─────────────────────────────────────────────────────────────────

def save_dds(image, dest, pixel_format=None, mipmaps=True, mipmaps_linear=False, pca=0):
    """Save a PIL Image as a DDS file with BCn compression and mipmaps.

    Args:
        image:        PIL Image (any mode — converted automatically).
        dest:         File path (str / Path) or writable binary file object.
        pixel_format: ``"DXT1"`` (BC1), ``"DXT5"`` (BC3), or ``"BC7"`` (BPTC).
                      Auto-detected from image mode when *None*.
        mipmaps:      Generate a full mipmap chain down to 1×1 (default True).
        pca:          Number of power-iteration steps for PCA endpoint selection.
                      0 (default) uses fast bounding-box endpoints; 4 is a good
                      quality/speed trade-off.
    """
    if pixel_format is None:
        pixel_format = "DXT5" if image.mode in ("RGBA", "LA", "PA") else "DXT1"
    if pixel_format not in ("DXT1", "DXT5", "BC7", "BC7lite", "BC7nano", "BC7zero"):
        raise ValueError(f"pixel_format must be 'DXT1', 'DXT5', or 'BC7', got {pixel_format!r}")

    if pixel_format == "BC7" or pixel_format == "BC7lite" or pixel_format == "BC7nano" or pixel_format == "BC7zero":
        image = image.convert("RGBA")
    else:
        image = image.convert("RGBA" if pixel_format == "DXT5" else "RGB")
    w, h = image.size

    # Build mip chain
    levels = [image]
    if mipmaps:
        mw, mh = w, h
        if not mipmaps_linear:
            while mw > 1 or mh > 1:
                mw, mh = max(1, mw // 2), max(1, mh // 2)
                levels.append(image.resize((mw, mh), Image.Resampling.BOX))
        else:
            # linear-light mipmap generation
            # sRGB -> Linear
            arr = np.array(image, dtype=np.float32) / 255.0
            c_idx = slice(-1) if (arr.ndim == 3 and arr.shape[-1] in (2, 4)) else slice(None)
            arr[..., c_idx] = np.where(arr[..., c_idx] <= 0.04045, arr[..., c_idx] / 12.92, ((arr[..., c_idx] + 0.055) / 1.055) ** 2.4)

            # Pillow lacks native multichannel float, so we process as a list of single-channel 'F' images
            bands = [Image.fromarray(arr[..., i], 'F') for i in range(arr.shape[-1])] if arr.ndim == 3 else[Image.fromarray(arr, 'F')]

            while mw > 1 or mh > 1:
                mw, mh = max(1, mw // 2), max(1, mh // 2)
                
                # Use Pillow's native float resizing
                bands =[b.resize((mw, mh), Image.Resampling.BOX) for b in bands]
                
                # Linear -> sRGB for storage
                s_mip = np.stack([np.array(b) for b in bands], axis=-1) if arr.ndim == 3 else np.array(bands[0])
                s_mip[..., c_idx] = np.where(s_mip[..., c_idx] <= 0.0031308, s_mip[..., c_idx] * 12.92, 1.055 * (s_mip[..., c_idx] ** (1/2.4)) - 0.055)
                levels.append(Image.fromarray((np.clip(s_mip, 0, 1) * 255).astype(np.uint8)))

    # Write
    own_file = not hasattr(dest, "write")
    f = open(dest, "wb") if own_file else dest
    try:
        if pixel_format == "BC7" or pixel_format == "BC7lite" or pixel_format == "BC7nano" or pixel_format == "BC7zero":
            _write_header_dx10(f, w, h, 98, len(levels))
        else:
            _write_header(f, w, h, pixel_format, len(levels))
        for lvl in levels:
            f.write(_compress_level(np.asarray(lvl), pixel_format, pca=pca))
    finally:
        if own_file:
            f.close()