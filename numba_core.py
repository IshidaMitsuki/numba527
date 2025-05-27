"""
numba_core.py – Numba‑accelerated low‑level helpers
--------------------------------------------------
Version 0.3 – *fully nopython‑compatible*

All heavy bit‑twiddling is implemented here so that the remainder of the
AI can stay pure‑Python.  Functions fall back to slow Python if Numba is
missing, but when JITed we see 10‑20× speed‑ups.

Public API
~~~~~~~~~~
    >>> from numba_core import drop_piece, heights, KIND_TO_IDX

* **drop_piece(cols, kind_idx, rot, x) → (new_cols, cleared) | None**
* **heights(cols) → tuple[int, …] (len==10)**

`cols` is the usual 10‑tuple of 20‑bit column bitboards.
"""
from __future__ import annotations
from typing import Tuple

# ---------------------------------------------------------------------
# Safe import of Numba -------------------------------------------------
try:
    import numba as nb
    _njit = nb.njit(cache=True, fastmath=True)
except ModuleNotFoundError:
    def _njit(*a, **kw):  # type: ignore
        def deco(f):
            return f
        return deco

# ---------------------------------------------------------------------
# Shape table – flattened 28 entries (7 kinds × 4 rots) ----------------
_SHAPE_TABLE: Tuple[Tuple[Tuple[int, int], ...], ...] = (
    # I
    ((-1,0),(0,0),(1,0),(2,0)),
    ((1,-1),(1,0),(1,1),(1,2)),
    ((-1,1),(0,1),(1,1),(2,1)),
    ((0,-1),(0,0),(0,1),(0,2)),
    # O
    ((0,0),(1,0),(0,1),(1,1)),
    ((0,0),(1,0),(0,1),(1,1)),
    ((0,0),(1,0),(0,1),(1,1)),
    ((0,0),(1,0),(0,1),(1,1)),
    # T
    ((-1,0),(0,0),(1,0),(0,1)),
    ((0,-1),(0,0),(1,0),(0,1)),
    ((-1,0),(0,0),(1,0),(0,-1)),
    ((0,-1),(0,0),(-1,0),(0,1)),
    # S
    ((-1,0),(0,0),(0,1),(1,1)),
    ((0,1),(0,0),(1,0),(1,-1)),
    ((1,0),(0,0),(0,-1),(-1,-1)),
    ((0,-1),(0,0),(-1,0),(-1,1)),
    # Z
    ((0,0),(1,0),(-1,1),(0,1)),
    ((0,0),(0,-1),(1,1),(1,0)),
    ((0,0),(-1,0),(1,-1),(0,-1)),
    ((0,0),(0,1),(-1,-1),(-1,0)),
    # J
    ((-1,0),(0,0),(1,0),(-1,1)),
    ((0,1),(0,0),(0,-1),(1,1)),
    ((1,0),(0,0),(-1,0),(1,-1)),
    ((0,-1),(0,0),(0,1),(-1,-1)),
    # L
    ((-1,0),(0,0),(1,0),(1,1)),
    ((0,1),(0,0),(0,-1),(1,-1)),
    ((1,0),(0,0),(-1,0),(-1,-1)),
    ((0,-1),(0,0),(0,1),(-1,1)),
)

_KIND_TO_IDX = {
    'I': 0,
    'O': 1,
    'T': 2,
    'S': 3,
    'Z': 4,
    'J': 5,
    'L': 6,
}
KIND_TO_IDX = _KIND_TO_IDX  # re‑export

# ---------------------------------------------------------------------
# Height calculation ---------------------------------------------------
@_njit
def _calc_heights(cols):
    res0=res1=res2=res3=res4=res5=res6=res7=res8=res9=0
    for x in range(10):
        col=cols[x]; h=0
        while col:
            h+=1; col>>=1
        if   x==0: res0=h
        elif x==1: res1=h
        elif x==2: res2=h
        elif x==3: res3=h
        elif x==4: res4=h
        elif x==5: res5=h
        elif x==6: res6=h
        elif x==7: res7=h
        elif x==8: res8=h
        else:       res9=h
    return (res0,res1,res2,res3,res4,res5,res6,res7,res8,res9)

# ---------------------------------------------------------------------
# Shift helper (inlineable) -------------------------------------------
@_njit
def _shift_single(col, rows, nrows):
    for i in range(nrows):
        r = rows[i]
        lower = col & ((1 << r) - 1)
        upper = col >> (r + 1)
        col = lower | (upper << r)
    return col

# ---------------------------------------------------------------------
@_njit
def drop_piece(cols: Tuple[int, ...], kind_idx: int, rot: int, x: int):
    """Return (new_cols, cleared) or None if top‑out."""
    shape = _SHAPE_TABLE[kind_idx * 4 + rot]

    # landing y & max_dy ----------------------------------------------
    y = 0
    max_dy = shape[0][1]
    for i in range(4):
        dx, dy = shape[i]
        if dy > max_dy:
            max_dy = dy
        col = cols[x + dx]
        h = 0
        tmp = col
        while tmp:
            h += 1
            tmp >>= 1
        v = h - dy
        if v > y: y = v

    if y + max_dy >= 20:
        return None

    # place blocks -----------------------------------------------------
    nc0,nc1,nc2,nc3,nc4,nc5,nc6,nc7,nc8,nc9 = cols
    touched_rows = [0,0,0,0]
    ntouched = 0
    for i in range(4):
        dx,dy = shape[i]
        row = y + dy
        bit = 1 << row
        idx = x + dx
        if   idx==0: nc0 |= bit
        elif idx==1: nc1 |= bit
        elif idx==2: nc2 |= bit
        elif idx==3: nc3 |= bit
        elif idx==4: nc4 |= bit
        elif idx==5: nc5 |= bit
        elif idx==6: nc6 |= bit
        elif idx==7: nc7 |= bit
        elif idx==8: nc8 |= bit
        else:        nc9 |= bit
        touched_rows[ntouched] = row
        ntouched += 1

    # find full rows ---------------------------------------------------
    full_rows = [0,0,0,0]
    nfull = 0
    for i in range(ntouched):
        r = touched_rows[i]
        mask = 1 << r
        if ((nc0 & mask) and (nc1 & mask) and (nc2 & mask) and (nc3 & mask)
            and (nc4 & mask) and (nc5 & mask) and (nc6 & mask)
            and (nc7 & mask) and (nc8 & mask) and (nc9 & mask)):
            full_rows[nfull] = r
            nfull += 1
    if nfull == 0:
        return (nc0,nc1,nc2,nc3,nc4,nc5,nc6,nc7,nc8,nc9), 0

    # shift ------------------------------------------------------------
    nc0 = _shift_single(nc0, full_rows, nfull)
    nc1 = _shift_single(nc1, full_rows, nfull)
    nc2 = _shift_single(nc2, full_rows, nfull)
    nc3 = _shift_single(nc3, full_rows, nfull)
    nc4 = _shift_single(nc4, full_rows, nfull)
    nc5 = _shift_single(nc5, full_rows, nfull)
    nc6 = _shift_single(nc6, full_rows, nfull)
    nc7 = _shift_single(nc7, full_rows, nfull)
    nc8 = _shift_single(nc8, full_rows, nfull)
    nc9 = _shift_single(nc9, full_rows, nfull)

    return (nc0,nc1,nc2,nc3,nc4,nc5,nc6,nc7,nc8,nc9), nfull

# ---------------------------------------------------------------------
# Public wrappers ------------------------------------------------------

def heights(cols):
    return _calc_heights(cols)

__all__ = [
    'drop_piece', 'heights', 'KIND_TO_IDX',
]
