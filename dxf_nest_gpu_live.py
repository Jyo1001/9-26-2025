#!/usr/bin/env python3
# dxf_nest_gpu_live.py — Live HTML viewer + SSE controls (pause/resume/stop), optional CUDA/NumPy
# - GPU accel via PyTorch when available; fast CPU fallback via NumPy if found; else pure Python
# - Thickness grouping by filename prefix "<thickness><unit>-*.dxf" (e.g., 0.5in-*, 12mm-*)
# - Guaranteed spacing via geometric offsets (pyclipper) with pixel fallback; no shared-line cutting unless enabled
# - Output: per-thickness nested .dxf; optional split per sheet
# - Standalone HTML UI written to the folder; opens automatically; Pause/Resume/Stop & Save

# ================= Default Settings (overridable by CLI) =================
FOLDER = r"C:\Users\Jsudhakaran\OneDrive - GN Corporation Inc\Desktop\test\For waterjet cutting"

SHEET_W = 48.0
SHEET_H = 96.0
SHEET_MARGIN = 0.50
SHEET_GAP = 2.0

SPACING  = 0.125          # gap between parts (drawing units)
JOIN_TOL = 0.005
ARC_CHORD_TOL = 0.01

# Spacing accuracy controls
SPACING_ABS_TOL = 1e-4     # absolute tolerance (drawing units)
SPACING_REL_TOL = 0.01     # relative tolerance (fraction of spacing)

# Safety halo applied around each rasterized part (drawing units)
SAFETY_GAP = 0.01

FALLBACK_OPEN_AS_BBOX = True

ALLOW_ROTATE_90   = True
ALLOW_MIRROR      = False
USE_OBB_CANDIDATE = True

INSUNITS = 1  # 1=inches (DXF header), 4=mm

RECT_ALIGN_MODE = "prefer"  # "off" | "prefer" | "force"
RECT_ALIGN_TOL  = 1e-3

ALLOW_NEST_IN_HOLES = True
NEST_MODE = "bitmap"         # "bitmap" | "shelf"
PIXELS_PER_UNIT = 20

BITMAP_EVAL_WORKERS = None
BITMAP_DEVICE = None  # "cuda", "cuda:0", "cpu", etc.

SHUFFLE_TRIES = 5
SHUFFLE_SEED  = None

GROUP_BY_THICKNESS = False
THICKNESS_LABEL_UNITS = "auto"  # "auto"|"in"|"mm"
SPLIT_SHEETS = False
MERGE_LINES  = True         # shared-line cutting OFF by default (preserve gap)
ROTATION_STEP_DEG = 0.0

SAFETY_PX = 2
MIN_SPACING_PIXELS = 4

CLIPPER_SCALE = 1_000_000.0
CLIPPER_ARC_TOL = 0.001
CLIPPER_MITER = 4.0
MAX_AUTO_SCALE = 8192

_PRECISION_WARNING_EMITTED = False

# Live UI / server
UI_FILENAME = "nest_viewer.html"
HTTP_HOST   = "127.0.0.1"
HTTP_PORT   = 0  # 0 = auto-pick a free port
# ========================================================================

import os, math, re, sys, json, time, webbrowser, threading, traceback, datetime
from typing import List, Tuple, Dict, Optional, Any
from random import Random
from urllib.parse import urlparse, parse_qs

try:
    import pyclipper
except Exception:  # optional dependency for precise offsets
    pyclipper = None

PRECISION_OFFSETS_AVAILABLE = pyclipper is not None

# Fallback sample folder shipped with script
_REPO_SAMPLE_FOLDER = os.path.join(os.path.dirname(__file__), "For waterjet cutting")
if os.path.isdir(_REPO_SAMPLE_FOLDER):
    FOLDER = _REPO_SAMPLE_FOLDER

if not BITMAP_EVAL_WORKERS:
    cpu_count = os.cpu_count() or 1
    BITMAP_EVAL_WORKERS = max(1, cpu_count)

IS_WINDOWS = (os.name == "nt")

# Runtime toggles exposed in the HTML UI (key, label, global attr, description)
_UI_TOGGLE_DEFS = [
    ("allow_mirror", "Allow mirror / flip parts", "ALLOW_MIRROR", "Permit mirrored copies when searching poses."),
    ("allow_rotate_90", "Allow automatic 90° rotations", "ALLOW_ROTATE_90", "Include a 90° rotation candidate."),
    ("use_obb", "Use OBB seeding", "USE_OBB_CANDIDATE", "Seed placement with oriented bounding box pose."),
    ("allow_holes", "Allow nesting inside holes", "ALLOW_NEST_IN_HOLES", "Permit parts to be placed within other part holes."),
    ("fallback_bbox", "Fallback open profiles to bounding boxes", "FALLBACK_OPEN_AS_BBOX", "Treat open DXF profiles as rectangles."),
    ("group_by_thickness", "Group by thickness labels", "GROUP_BY_THICKNESS", "Nest files grouped by detected thickness."),
    ("split_sheets", "Split sheets into separate DXFs", "SPLIT_SHEETS", "Write one DXF per finished sheet."),
    ("merge_lines", "Merge touching lines", "MERGE_LINES", "Combine collinear edges for shared cutting."),
]


_UI_FIELD_DEFS = [
    {
        "key": "folder",
        "label": "DXF folder",
        "attr": "FOLDER",
        "type": "text",
        "description": "Directory containing DXF files and optional quantity/label text files.",
        "parser": lambda raw: os.path.abspath(os.path.expanduser(str(raw))),
    },
    {
        "key": "sheet_w",
        "label": "Sheet width",
        "attr": "SHEET_W",
        "type": "number",
        "min": 0.01,
        "step": 0.5,
        "parser": lambda raw: max(0.01, float(raw)),
    },
    {
        "key": "sheet_h",
        "label": "Sheet height",
        "attr": "SHEET_H",
        "type": "number",
        "min": 0.01,
        "step": 0.5,
        "parser": lambda raw: max(0.01, float(raw)),
    },
    {
        "key": "sheet_margin",
        "label": "Sheet margin",
        "attr": "SHEET_MARGIN",
        "type": "number",
        "min": 0.0,
        "step": 0.05,
        "parser": lambda raw: max(0.0, float(raw)),
    },
    {
        "key": "sheet_gap",
        "label": "Gap between exported sheets",
        "attr": "SHEET_GAP",
        "type": "number",
        "min": 0.0,
        "step": 0.1,
        "parser": lambda raw: max(0.0, float(raw)),
    },
    {
        "key": "spacing",
        "label": "Part spacing",
        "attr": "SPACING",
        "type": "number",
        "min": 0.0,
        "step": 0.01,
        "parser": lambda raw: max(0.0, float(raw)),
    },
    {
        "key": "safety_gap",
        "label": "Safety halo",
        "attr": "SAFETY_GAP",
        "type": "number",
        "min": 0.0,
        "step": 0.001,
        "parser": lambda raw: max(0.0, float(raw)),
    },
    {
        "key": "spacing_abs_tol",
        "label": "Spacing abs tol",
        "attr": "SPACING_ABS_TOL",
        "type": "number",
        "min": 1e-9,
        "step": 1e-5,
        "parser": lambda raw: max(1e-9, float(raw)),
    },
    {
        "key": "spacing_rel_tol",
        "label": "Spacing rel tol",
        "attr": "SPACING_REL_TOL",
        "type": "number",
        "min": 0.0,
        "step": 0.001,
        "parser": lambda raw: max(0.0, float(raw)),
    },
    {
        "key": "join_tol",
        "label": "Join tolerance",
        "attr": "JOIN_TOL",
        "type": "number",
        "min": 0.0,
        "step": 0.001,
        "parser": lambda raw: max(0.0, float(raw)),
    },
    {
        "key": "arc_chord_tol",
        "label": "Arc chord tolerance",
        "attr": "ARC_CHORD_TOL",
        "type": "number",
        "min": 0.0,
        "step": 0.001,
        "parser": lambda raw: max(0.0, float(raw)),
    },
    {
        "key": "rect_align_mode",
        "label": "Rectangular alignment",
        "attr": "RECT_ALIGN_MODE",
        "type": "select",
        "options": [
            {"value": "off", "label": "Off"},
            {"value": "prefer", "label": "Prefer"},
            {"value": "force", "label": "Force"},
        ],
        "parser": lambda raw: (str(raw).lower() if str(raw).lower() in {"off", "prefer", "force"} else RECT_ALIGN_MODE),
    },
    {
        "key": "rect_align_tol",
        "label": "Rectangular align tolerance",
        "attr": "RECT_ALIGN_TOL",
        "type": "number",
        "min": 0.0,
        "step": 1e-4,
        "parser": lambda raw: max(0.0, float(raw)),
    },
    {
        "key": "nest_mode",
        "label": "Nesting mode",
        "attr": "NEST_MODE",
        "type": "select",
        "options": [
            {"value": "bitmap", "label": "Bitmap (fast, dense)"},
            {"value": "shelf", "label": "Shelf (simple)"},
        ],
        "parser": lambda raw: (str(raw).lower() if str(raw).lower() in {"bitmap", "shelf"} else NEST_MODE),
    },
    {
        "key": "pixels_per_unit",
        "label": "Pixels per unit",
        "attr": "PIXELS_PER_UNIT",
        "type": "number",
        "min": 1,
        "step": 1,
        "parser": lambda raw: max(1, int(float(raw))),
    },
    {
        "key": "bitmap_workers",
        "label": "Bitmap workers",
        "attr": "BITMAP_EVAL_WORKERS",
        "type": "number",
        "min": 1,
        "step": 1,
        "allow_none": True,
        "parser": lambda raw: max(1, int(float(raw))),
    },
    {
        "key": "bitmap_device",
        "label": "Bitmap device",
        "attr": "BITMAP_DEVICE",
        "type": "text",
        "allow_none": True,
        "description": "PyTorch device string (e.g. cuda, cuda:0, cpu). Leave blank for auto.",
        "parser": lambda raw: (str(raw).strip() or None),
    },
    {
        "key": "shuffle_tries",
        "label": "Shuffle tries",
        "attr": "SHUFFLE_TRIES",
        "type": "number",
        "min": 1,
        "step": 1,
        "parser": lambda raw: max(1, int(float(raw))),
    },
    {
        "key": "shuffle_seed",
        "label": "Shuffle seed",
        "attr": "SHUFFLE_SEED",
        "type": "number",
        "allow_none": True,
        "parser": lambda raw: int(float(raw)),
    },
    {
        "key": "thickness_units",
        "label": "Thickness label units",
        "attr": "THICKNESS_LABEL_UNITS",
        "type": "select",
        "options": [
            {"value": "auto", "label": "Auto"},
            {"value": "in", "label": "Inches"},
            {"value": "mm", "label": "Millimetres"},
        ],
        "parser": lambda raw: (str(raw).lower() if str(raw).lower() in {"auto", "in", "mm"} else THICKNESS_LABEL_UNITS),
    },
    {
        "key": "rotation_step",
        "label": "Rotation step (deg)",
        "attr": "ROTATION_STEP_DEG",
        "type": "number",
        "min": 0.0,
        "step": 1.0,
        "parser": lambda raw: max(0.0, float(raw)),
    },
    {
        "key": "safety_px",
        "label": "Safety pixels",
        "attr": "SAFETY_PX",
        "type": "number",
        "min": 0,
        "step": 1,
        "parser": lambda raw: max(0, int(float(raw))),
    },
    {
        "key": "min_spacing_px",
        "label": "Minimum spacing pixels",
        "attr": "MIN_SPACING_PIXELS",
        "type": "number",
        "min": 1,
        "step": 1,
        "parser": lambda raw: max(1, int(float(raw))),
    },
    {
        "key": "insunits",
        "label": "DXF INSUNITS header",
        "attr": "INSUNITS",
        "type": "select",
        "options": [
            {"value": "in", "label": "Inches"},
            {"value": "mm", "label": "Millimetres"},
        ],
        "parser": lambda raw: (1 if str(raw).lower() != "mm" else 4),
        "to_ui": lambda val: "mm" if int(val)==4 else "in",
    },
]


def _ui_toggle_snapshot():
    snap = []
    g = globals()
    for key, label, attr, desc in _UI_TOGGLE_DEFS:
        snap.append({
            "key": key,
            "label": label,
            "value": bool(g.get(attr)),
            "description": desc,
        })
    return snap


def _ui_field_snapshot():
    snap = []
    g = globals()
    for field in _UI_FIELD_DEFS:
        attr = field["attr"]
        raw_val = g.get(attr)
        to_ui = field.get("to_ui")
        try:
            value = to_ui(raw_val) if to_ui else raw_val
        except Exception:
            value = raw_val
        if value is None:
            value = ""
        entry = {
            "key": field["key"],
            "label": field.get("label", field["key"]),
            "type": field.get("type", "text"),
            "value": value,
            "description": field.get("description"),
        }
        for extra in ("min", "max", "step", "placeholder"):
            if extra in field:
                entry[extra] = field[extra]
        options = field.get("options")
        if options:
            entry["options"] = options
        snap.append(entry)
    return snap


def _apply_toggle_config(cfg: Dict[str, Any]):
    if not isinstance(cfg, dict):
        return
    g = globals()
    for key, _label, attr, _desc in _UI_TOGGLE_DEFS:
        if key in cfg:
            g[attr] = bool(cfg[key])


def _apply_field_config(cfg: Dict[str, Any]):
    if not isinstance(cfg, dict):
        return
    g = globals()
    for field in _UI_FIELD_DEFS:
        key = field["key"]
        if key not in cfg:
            continue
        raw = cfg[key]
        if raw is None:
            if field.get("allow_none"):
                g[field["attr"]] = None
            continue
        if isinstance(raw, str) and raw.strip() == "":
            if field.get("allow_none"):
                g[field["attr"]] = None
            continue
        parser = field.get("parser")
        try:
            value = parser(raw) if parser else raw
        except Exception:
            continue
        g[field["attr"]] = value


# ---------- Tiny Win progress window (optional) ----------
if IS_WINDOWS:
    import ctypes
    user32  = ctypes.windll.user32
    gdi32   = ctypes.windll.gdi32
    kernel32= ctypes.windll.kernel32

    UINT = ctypes.c_uint; DWORD = ctypes.c_uint; INT = ctypes.c_int; LONG = ctypes.c_long
    ULONG_PTR = ctypes.c_size_t; LONG_PTR  = ctypes.c_ssize_t
    WPARAM = ULONG_PTR; LPARAM = LONG_PTR; LRESULT = LONG_PTR
    HWND = ctypes.c_void_p; HINSTANCE = ctypes.c_void_p; HICON = ctypes.c_void_p
    HCURSOR = ctypes.c_void_p; HBRUSH = ctypes.c_void_p; HMENU = ctypes.c_void_p
    LPCWSTR = ctypes.c_wchar_p

    WS_OVERLAPPEDWINDOW = 0x00CF0000; WS_VISIBLE = 0x10000000
    WS_CHILD = 0x40000000; WS_EX_TOPMOST = 0x00000008
    SW_SHOWNORMAL = 1; WM_DESTROY = 0x0002; PM_REMOVE = 0x0001
    SS_LEFT = 0x00000000; SS_NOPREFIX = 0x00000080; WHITE_BRUSH = 0

    class POINT(ctypes.Structure): _fields_=[("x", LONG), ("y", LONG)]
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, HWND, UINT, WPARAM, LPARAM)
    class WNDCLASS(ctypes.Structure):
        _fields_=[("style", UINT),("lpfnWndProc", WNDPROC),("cbClsExtra", INT),("cbWndExtra", INT),
                  ("hInstance", HINSTANCE),("hIcon", HICON),("hCursor", HCURSOR),
                  ("hbrBackground", HBRUSH),("lpszMenuName", LPCWSTR),("lpszClassName", LPCWSTR)]
    class MSG(ctypes.Structure):
        _fields_=[("hwnd", HWND),("message", UINT),("wParam", WPARAM),("lParam", LPARAM),
                  ("time", DWORD),("pt", POINT)]

    user32.DefWindowProcW.argtypes=[HWND, UINT, WPARAM, LPARAM]
    user32.DefWindowProcW.restype=LRESULT
    user32.RegisterClassW.argtypes=[ctypes.POINTER(WNDCLASS)]
    user32.CreateWindowExW.argtypes=[DWORD, LPCWSTR, LPCWSTR, DWORD, INT, INT, INT, INT, HWND, HMENU, HINSTANCE, ctypes.c_void_p]
    user32.CreateWindowExW.restype=HWND
    gdi32.GetStockObject.argtypes=[INT]; gdi32.GetStockObject.restype=HBRUSH
    DefWindowProcW = user32.DefWindowProcW

    class WinProgress:
        def __init__(self, title="Nesting DXF…", width=520, height=220):
            self.enabled=True; self.title=title; self.width=width; self.height=height
            self.hInstance = kernel32.GetModuleHandleW(None)
            self.hwnd = HWND(); self.hStatic = HWND(); self._wndproc=None
        def create(self):
            try:
                @WNDPROC
                def wndproc(hwnd, msg, wParam, lParam):
                    if msg==WM_DESTROY:
                        user32.PostQuitMessage(0); return LRESULT(0)
                    try: return DefWindowProcW(hwnd, msg, wParam, lParam)
                    except: return LRESULT(0)
                self._wndproc = wndproc
                cls = WNDCLASS(); cls.lpfnWndProc = self._wndproc; cls.hInstance=self.hInstance
                cls.hbrBackground = gdi32.GetStockObject(WHITE_BRUSH); cls.lpszClassName="PyNestProgress"
                try: user32.RegisterClassW(ctypes.byref(cls))
                except: pass
                sw=user32.GetSystemMetrics(0); sh=user32.GetSystemMetrics(1)
                x=max(0,(sw-self.width)//2); y=max(0,(sh-self.height)//2)
                self.hwnd=user32.CreateWindowExW(0,"PyNestProgress",self.title,
                    WS_OVERLAPPEDWINDOW|WS_VISIBLE,x,y,self.width,self.height,None,None,self.hInstance,None)
                self.hStatic=user32.CreateWindowExW(0,"STATIC","Loading…",WS_CHILD|WS_VISIBLE|SS_LEFT|SS_NOPREFIX,
                    12,12,self.width-24,self.height-24,self.hwnd,None,self.hInstance,None)
                user32.ShowWindow(self.hwnd, SW_SHOWNORMAL); self.pump()
            except: self.enabled=False
        def pump(self):
            if not self.enabled: return
            msg = MSG()
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                user32.TranslateMessage(ctypes.byref(msg)); user32.DispatchMessageW(ctypes.byref(msg))
        def update(self, text: str):
            if not self.enabled: return
            try: user32.SetWindowTextW(self.hStatic, text); self.pump()
            except: pass
        def close(self):
            if not self.enabled: return
            try: user32.DestroyWindow(self.hwnd); self.hwnd=HWND()
            except: pass
else:
    class WinProgress:
        def __init__(self,*_,**__): self.enabled=False
        def create(self): pass
        def update(self,_): pass
        def pump(self): pass
        def close(self): pass

# --------- logger ---------
_report_lines: List[str] = []
def log(line: str):
    print(line)
    _report_lines.append(line)

def _warn_precise_offsets_fallback():
    global _PRECISION_WARNING_EMITTED
    if not PRECISION_OFFSETS_AVAILABLE and not _PRECISION_WARNING_EMITTED:
        log("[WARN] pyclipper not available — spacing accuracy falls back to pixel dilation.")
        _PRECISION_WARNING_EMITTED = True

Point = Tuple[float,float]
Loop  = List[Point]
Seg   = Tuple[Point,Point]

# ---------- Torch (optional) ----------
try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None; F = None

class TorchMaskOps:
    def __init__(self, device: Optional[str] = None):
        if not torch: raise RuntimeError("PyTorch not available")
        if device is None: device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
    def zeros(self, H: int, W: int): return torch.zeros((H,W), dtype=torch.uint8, device=self.device)
    def mask_to_tensor(self, mask_01):
        H=len(mask_01); W=len(mask_01[0]) if H else 0
        t=torch.empty((H,W), dtype=torch.uint8, device=self.device)
        for y in range(H):
            row = mask_01[y]
            if isinstance(row,(bytearray,bytes)): row=list(row)
            t[y,:W]=torch.tensor(row, dtype=torch.uint8, device=self.device)
        return t
    def _disk_kernel(self, r:int):
        if r<=0: return torch.ones((1,1,1,1), dtype=torch.uint8, device=self.device)
        xs,ys=torch.meshgrid(torch.arange(-r,r+1,device=self.device), torch.arange(-r,r+1,device=self.device), indexing='ij')
        k=((xs*xs+ys*ys)<=(r*r)).to(torch.uint8)
        return k.view(1,1,k.shape[0],k.shape[1])
    def find_first_fit(self, occ_safe, test_mask_tensor):
        H,W=occ_safe.shape; ph,pw=test_mask_tensor.shape
        if H<ph or W<pw: return None
        x=occ_safe.unsqueeze(0).unsqueeze(0).to(torch.float32)
        k=test_mask_tensor.flip(0,1).unsqueeze(0).unsqueeze(0).to(torch.float32)
        heat=F.conv2d(x,k,stride=1); ok=(heat==0)
        if not torch.any(ok): return None
        yy,xx=torch.where(ok[0,0]); y=int(yy.min().item()); x=int(xx[yy==y].min().item()); return (x,y)
    def or_mask(self, occ, raw_mask, ox:int, oy:int):
        ph,pw=raw_mask.shape; occ[oy:oy+ph,ox:ox+pw]|=raw_mask
    def or_dilated(self, occ, raw_or_shell, ox:int, oy:int, r:int):
        ph,pw=raw_or_shell.shape
        tile=torch.zeros_like(occ); tile[oy:oy+ph, ox:ox+pw]=raw_or_shell
        if r>0:
            k=self._disk_kernel(r).to(torch.float32)
            y=(F.conv2d(tile.unsqueeze(0).unsqueeze(0).to(torch.float32),k,padding=r)>0).to(torch.uint8)
            tile=y[0,0]
        occ|=tile
    def count_true(self, occ)->int: return int(occ.sum().item())

# ---------- NumPy (optional) ----------
try:
    import numpy as np
except Exception:
    np = None

class NumpyMaskOps:
    """CPU acceleration using NumPy (no GPU required)."""
    def __init__(self):
        if np is None:
            raise RuntimeError("NumPy not available")
        self.device = "numpy"

    def zeros(self, H: int, W: int):
        return np.zeros((H, W), dtype=np.uint8)

    def mask_to_tensor(self, mask_01):
        H = len(mask_01); W = len(mask_01[0]) if H else 0
        arr = np.zeros((H, W), dtype=np.uint8)
        for y in range(H):
            row = mask_01[y]
            if isinstance(row, (bytes, bytearray)):
                arr[y, :W] = np.frombuffer(row, dtype=np.uint8, count=W)
            else:
                arr[y, :W] = np.asarray(row, dtype=np.uint8)
        return arr

    def find_first_fit(self, occ, test_mask):
        """Find first (x,y) where (occ AND test)==0 using FFT correlation."""
        H, W = occ.shape
        ph, pw = test_mask.shape
        if H < ph or W < pw:
            return None

        shape = (H + ph - 1, W + pw - 1)
        f1 = np.fft.rfftn(occ.astype(np.float32), shape)
        f2 = np.fft.rfftn(np.flipud(np.fliplr(test_mask.astype(np.float32))), shape)
        heat_full = np.fft.irfftn(f1 * f2, shape)
        valid = heat_full[ph - 1:H, pw - 1:W]
        ok = valid <= 0.5
        if not np.any(ok):
            return None
        yy, xx = np.where(ok)
        y = int(yy.min())
        x = int(xx[yy == y].min())
        return (x, y)

    def or_mask(self, occ, raw_mask, ox: int, oy: int):
        ph, pw = raw_mask.shape
        occ[oy:oy + ph, ox:ox + pw] |= raw_mask

    def or_dilated(self, occ, raw_or_shell, ox: int, oy: int, r: int):
        ph, pw = raw_or_shell.shape
        tile = np.zeros_like(occ, dtype=np.uint8)
        tile[oy:oy + ph, ox:ox + pw] = raw_or_shell
        if r <= 0:
            occ |= tile
            return
        H, W = occ.shape
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx*dx + dy*dy > r*r:
                    continue
                y0 = max(0, dy); y1 = min(H, H + dy)
                x0 = max(0, dx); x1 = min(W, W + dx)
                if y1 <= y0 or x1 <= x0:
                    continue
                occ[y0:y1, x0:x1] |= tile[y0 - dy:y1 - dy, x0 - dx:x1 - dx]

    def count_true(self, occ) -> int:
        return int(occ.sum())

def build_mask_ops(device_pref: Optional[str]):
    # Prefer PyTorch (CUDA or CPU). Else fallback to NumPy. Else None.
    if torch is not None:
        try:
            dev = device_pref.strip() if (device_pref and isinstance(device_pref, str)) else None
            if dev is None:
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            _ = torch.zeros(1, device=torch.device(dev))
            return TorchMaskOps(dev)
        except Exception:
            pass
    if np is not None:
        try:
            return NumpyMaskOps()
        except Exception:
            pass
    return None

# ---------- thickness + qty ----------
def _read_text(p):
    with open(p,'r',encoding='utf-8',errors='ignore') as f:
        return f.read().splitlines()

def _normalize_thickness_label(value: float, unit: str) -> str:
    s = f"{value:.4f}".rstrip('0').rstrip('.')
    return f"{s}{'in' if unit=='in' else 'mm'}"

def _parse_thickness_from_token(token: str, default_unit: str) -> Optional[Tuple[float,str]]:
    t = token.strip().lower()
    m = re.match(r'^([0-9]*\.?[0-9]+)\s*(?:("|in(?:ch(?:es)?)?|mm|millimet(?:er|re)s?))$', t)
    if m:
        val=float(m.group(1)); raw_u=m.group(2); unit='in' if (raw_u=='"' or (raw_u and 'in' in raw_u)) else 'mm'
        return val, unit
    m = re.match(r'^([0-9]*\.?[0-9]+)$', t)
    if m: return float(m.group(1)), default_unit
    return None

def _parse_thickness_from_basename(basename: str, default_unit: str) -> Optional[Tuple[float,str]]:
    token = basename.split('-', 1)[0]
    return _parse_thickness_from_token(token, default_unit)

def _convert_thickness_for_label(value: float, unit: str, label_units: str) -> Tuple[float, str]:
    unit = unit.lower()
    if label_units == "auto": return value, unit
    if label_units == "in":   return ((value/25.4) if unit=="mm" else value, "in")
    if label_units == "mm":   return ((value*25.4) if unit in ("in", '"') else value, "mm")
    return value, unit

def read_qty_for_dxf(folder: str, dxf_filename: str) -> int:
    base, _ = os.path.splitext(dxf_filename)
    for ext in ('.txt', '.TXT'):
        p = os.path.join(folder, base + ext)
        if os.path.isfile(p):
            try:
                lines=[ln.strip() for ln in open(p,'r',encoding='utf-8',errors='ignore') if ln.strip()]
                if not lines: return 1
                start = 1 if ('quantity' in lines[0].lower()) else 0
                total=0
                for ln in lines[start:]:
                    cells=[c.strip() for c in ln.split(',')]
                    token=cells[-1] if cells else ln
                    q=None
                    try: q=int(float(token))
                    except:
                        digs=''; 
                        for ch in ln:
                            if ch.isdigit(): digs+=ch
                            elif digs: break
                        if digs: q=int(digs)
                    if q and q>0: total+=q
                return total if total>0 else 1
            except: return 1
    return 1

def read_thickness_label(folder: str, dxf_filename: str, label_units: str) -> str:
    default_unit = 'in' if INSUNITS == 1 else 'mm'
    base, _ = os.path.splitext(dxf_filename)
    parsed = _parse_thickness_from_basename(os.path.basename(base), default_unit)
    if parsed:
        v,u = parsed
        vv, uu = _convert_thickness_for_label(v,u,label_units)
        return _normalize_thickness_label(vv, uu)
    return "unknown"

# ---------- DXF parse/join ----------
def _arc_points(cx,cy,r,a0_deg,a1_deg,chord_tol):
    a0=math.radians(a0_deg); a1=math.radians(a1_deg)
    while a1<a0: a1+=2*math.pi
    sweep=a1-a0
    if r<=0: return [(cx,cy)]
    dtheta=2*math.asin(max(0.0,min(1.0,chord_tol/(2*r)))) if chord_tol>0 else (math.pi/36)
    steps=max(2,int(math.ceil(sweep/max(dtheta,1e-6))))
    return [(cx+r*math.cos(a0+sweep*k/steps), cy+r*math.sin(a0+sweep*k/steps)) for k in range(steps+1)]

def _ellipse_points(cx, cy, mx, my, ratio, t0, t1, chord_tol):
    maj_len = math.hypot(mx, my)
    if maj_len <= 0: return [(cx,cy)]
    vx, vy = (-my/maj_len), (mx/maj_len)
    a_vecx, a_vecy = mx, my
    b_len = maj_len * max(0.0, ratio)
    b_vecx, b_vecy = vx * b_len, vy * b_len
    while t1 < t0: t1 += 2*math.pi
    sweep = t1 - t0
    steps = max(24, int(abs(sweep) / max(1e-6, 2*math.asin(min(1.0, chord_tol / max(1e-6, maj_len))))))
    pts=[]
    for k in range(steps+1):
        t = t0 + sweep * (k/steps)
        x = cx + a_vecx*math.cos(t) + b_vecx*math.sin(t)
        y = cy + a_vecy*math.cos(t) + b_vecy*math.sin(t)
        pts.append((x,y))
    return pts

def parse_entities(path: str):
    lines=_read_text(path)
    loops=[]; segs=[]
    in_entities=False
    in_lw=False; lw_pts=[]; lw_closed=False
    in_poly=False; poly_pts=[]; poly_closed=False
    in_spline=False; spline_fit=[]
    i=0; n=len(lines)
    def get(i): return lines[i].strip(), lines[i+1].strip()
    while i+1<n:
        code,val=get(i); i+=2
        if code=='0' and val=='SECTION':
            if i+1<n:
                c2,v2=get(i)
                if c2=='2' and v2=='ENTITIES': in_entities=True
            continue
        if code=='0' and val=='ENDSEC': in_entities=False; continue
        if not in_entities: continue
        if code=='0':
            # flush any open accumulators
            if in_lw:
                if lw_pts:
                    if lw_closed and lw_pts[0]!=lw_pts[-1]: lw_pts.append(lw_pts[0])
                    if len(lw_pts)>=4: loops.append(lw_pts)
                in_lw=False; lw_pts=[]; lw_closed=False
            if in_poly:
                if poly_pts:
                    if poly_closed and poly_pts[0]!=poly_pts[-1]: poly_pts.append(poly_pts[0])
                    if len(poly_pts)>=4: loops.append(poly_pts)
                in_poly=False; poly_pts=[]; poly_closed=False
            if in_spline:
                if len(spline_fit)>=3:
                    pts = list(spline_fit)
                    if pts[0]!=pts[-1]: pts.append(pts[0])
                    if len(pts)>=4: loops.append(pts)
                in_spline=False; spline_fit=[]
            if val=='LWPOLYLINE': in_lw=True; continue
            if val=='POLYLINE':   in_poly=True; continue
            if val=='SPLINE':     in_spline=True; spline_fit=[]; continue
            if val=='LINE':
                x1=y1=x2=y2=None
                while i+1<n:
                    c3,v3=get(i); i+=2
                    if c3=='0': i-=2; break
                    if c3=='10': x1=float(v3)
                    elif c3=='20': y1=float(v3)
                    elif c3=='11': x2=float(v3)
                    elif c3=='21': y2=float(v3)
                if None not in (x1,y1,x2,y2): segs.append(((x1,y1),(x2,y2)))
                continue
            if val in ('ARC','CIRCLE'):
                cx=cy=r=None; a0=0.0 if val=='CIRCLE' else None; a1=360.0 if val=='CIRCLE' else None
                while i+1<n:
                    c3,v3=get(i); i+=2
                    if c3=='0': i-=2; break
                    if   c3=='10': cx=float(v3)
                    elif c3=='20': cy=float(v3)
                    elif c3=='40': r =float(v3)
                    elif c3=='50': a0=float(v3)
                    elif c3=='51': a1=float(v3)
                if None not in (cx,cy,r,a0,a1):
                    pts=_arc_points(cx,cy,r,a0,a1,ARC_CHORD_TOL)
                    for k in range(len(pts)-1):
                        segs.append((pts[k],pts[k+1]))
                continue
            if val=='ELLIPSE':
                cx=cy=mx=my=ratio=None; t0=0.0; t1=2*math.pi
                while i+1<n:
                    c3,v3=get(i); i+=2
                    if c3=='0': i-=2; break
                    if c3=='10': cx=float(v3)
                    elif c3=='20': cy=float(v3)
                    elif c3=='11': mx=float(v3)
                    elif c3=='21': my=float(v3)
                    elif c3=='40': ratio=float(v3)
                    elif c3=='41': t0=float(v3)
                    elif c3=='42': t1=float(v3)
                if None not in (cx,cy,mx,my,ratio):
                    pts=_ellipse_points(cx,cy,mx,my,ratio,t0,t1,ARC_CHORD_TOL)
                    for k in range(len(pts)-1):
                        segs.append((pts[k], pts[k+1]))
                continue
            continue
        if in_lw:
            if code=='10':
                x=float(val)
                if i+1<n:
                    c2,v2=get(i); i+=2
                    if c2=='20': lw_pts.append((x:=float(x), y:=float(v2)) if False else (x,float(v2)))  # keep simple
                    else: i-=2
            elif code=='70':
                try: flags=int(val)
                except: flags=0
                lw_closed=bool(flags&1)
        elif in_poly:
            if code=='70':
                try: flags=int(val)
                except: flags=0
                poly_closed=bool(flags&1)
            elif code=='10':
                x=float(val)
                if i+1<n:
                    c2,v2=get(i); i+=2
                    if c2=='20': poly_pts.append((x,float(v2)))
                    else: i-=2
        elif in_spline:
            if code=='11':
                x=float(val)
                if i+1<n:
                    c2,v2=get(i); i+=2
                    if c2=='21': spline_fit.append((x,float(v2)))
                    else: i-=2
    if in_lw and lw_pts:
        if lw_closed and lw_pts[0]!=lw_pts[-1]: lw_pts.append(lw_pts[0])
        if len(lw_pts)>=4: loops.append(lw_pts)
    if in_poly and poly_pts:
        if poly_closed and poly_pts[0]!=poly_pts[-1]: poly_pts.append(poly_pts[0])
        if len(poly_pts)>=4: loops.append(poly_pts)
    if in_spline and len(spline_fit)>=3:
        pts=list(spline_fit)
        if pts[0]!=pts[-1]: pts.append(pts[0])
        if len(pts)>=4: loops.append(pts)
    return loops, segs

def join_segments_to_loops(segs: List[Seg], tol=JOIN_TOL) -> List[Loop]:
    if not segs: return []
    def key(pt): return (round(pt[0]/tol), round(pt[1]/tol))
    adj: Dict[tuple,List[tuple]]={}; used=[False]*len(segs)
    for idx,(a,b) in enumerate(segs):
        ka,kb=key(a),key(b)
        adj.setdefault(ka,[]).append((a,b,idx))
        adj.setdefault(kb,[]).append((b,a,idx))
    loops=[]
    for idx,(a0,b0) in enumerate(segs):
        if used[idx]: continue
        chain=[a0,b0]; used[idx]=True
        end=b0; kend=key(end)
        while True:
            nxt=None
            for a,b,j in adj.get(kend,[]):
                if used[j]: continue
                if abs(a[0]-end[0])<=tol and abs(a[1]-end[1])<=tol: nxt=(b,j); break
            if not nxt: break
            chain.append(nxt[0]); used[nxt[1]]=True; end=nxt[0]; kend=key(end)
        start=a0; kstart=key(start)
        while True:
            prv=None
            for a,b,j in adj.get(kstart,[]):
                if used[j]: continue
                if abs(b[0]-start[0])<=tol and abs(b[1]-start[1])<=tol: prv=(a,j); break
            if not prv: break
            chain.insert(0,prv[0]); used[prv[1]]=True; start=prv[0]; kstart=key(start)
        if len(chain)>=4 and abs(chain[0][0]-chain[-1][0])<=tol and abs(chain[0][1]-chain[-1][1])<=tol:
            if chain[0]!=chain[-1]: chain.append(chain[0])
            loops.append(chain)
    return loops

# ---------- geometry helpers ----------
def polygon_area(loop: Loop) -> float:
    s=0.0
    for i in range(len(loop)-1):
        x1,y1=loop[i]; x2,y2=loop[i+1]
        s += x1*y2 - x2*y1
    return 0.5*s

def bbox_of_points(pts: List[Tuple[float,float]]):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    return min(xs),min(ys),max(xs),max(ys)

def bbox_of_loops(loops: List[Loop]):
    pts=[p for lp in loops for p in lp]
    return bbox_of_points(pts) if pts else (0,0,0,0)

def translate_loop(loop: Loop, dx: float, dy: float) -> Loop:
    return [(x+dx,y+dy) for x,y in loop]

def mirror_loop(loop: Loop) -> Loop:
    mirrored = [(-x, y) for x, y in loop]
    minx = min((x for x, _ in mirrored), default=0.0)
    miny = min((y for _, y in mirrored), default=0.0)
    return [(x - minx, y - miny) for x, y in mirrored]

def rotate_loop(loop: Loop, theta: float) -> Loop:
    c,s=math.cos(theta), math.sin(theta)
    rot=[(x*c - y*s, x*s + y*c) for x,y in loop]
    minx=min(x for x,_ in rot); miny=min(y for _,y in rot)
    return [(x-minx,y-miny) for x,y in rot]

def _ensure_closed(loop: Loop) -> Loop:
    if not loop:
        return []
    if loop[0] != loop[-1]:
        return list(loop) + [loop[0]]
    return list(loop)

def _ensure_orientation(loop: Loop, ccw: bool) -> Loop:
    if len(loop) < 3:
        return list(loop)
    area = polygon_area(_ensure_closed(loop))
    if ccw and area < 0:
        return list(reversed(loop))
    if not ccw and area > 0:
        return list(reversed(loop))
    return list(loop)

def _offset_loops_precise(loops: List[Loop], delta: float) -> Optional[List[Loop]]:
    if delta == 0 or not loops:
        return [list(lp) for lp in loops]
    if pyclipper is None:
        return None
    co = pyclipper.PyclipperOffset(miterLimit=CLIPPER_MITER,
                                   arcTolerance=CLIPPER_ARC_TOL * CLIPPER_SCALE)
    added = False
    for idx, lp in enumerate(loops):
        if len(lp) < 3:
            continue
        want_ccw = (idx == 0)
        oriented = _ensure_orientation(lp, want_ccw)
        closed = _ensure_closed(oriented)
        path = [(int(round(x * CLIPPER_SCALE)), int(round(y * CLIPPER_SCALE)))
                for x, y in closed]
        if len(path) < 3:
            continue
        co.AddPath(path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        added = True
    if not added:
        return None
    result = co.Execute(delta * CLIPPER_SCALE)
    if not result:
        return []
    loops_out = []
    for path in result:
        pts = [(x / CLIPPER_SCALE, y / CLIPPER_SCALE) for x, y in path]
        pts = _ensure_closed(pts)
        loops_out.append(pts)
    loops_out.sort(key=lambda lp: abs(polygon_area(lp)), reverse=True)
    if loops_out:
        loops_out[0] = _ensure_orientation(loops_out[0], True)
        for i in range(1, len(loops_out)):
            loops_out[i] = _ensure_orientation(loops_out[i], False)
    return loops_out

def convex_hull(points: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    pts=sorted(set(points))
    if len(pts)<=1: return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]; upper=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0: lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0: upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]

def min_area_rect(points: List[Tuple[float,float]]):
    hull=convex_hull(points)
    if len(hull)<=1: return 0.0,0.0,0.0
    best=(float('inf'),0.0,0.0,0.0)
    for i in range(len(hull)):
        x1,y1=hull[i]; x2,y2=hull[(i+1)%len(hull)]
        theta=math.atan2(y2-y1, x2-x1)
        ct,st=math.cos(-theta), math.sin(-theta)
        xs=[px*ct - py*st for px,py in hull]
        ys=[px*st + py*ct for px,py in hull]
        w=max(xs)-min(xs); h=max(ys)-min(ys); area=w*h
        if area<best[0]: best=(area,w,h,theta)
    _,w,h,theta=best
    return w,h,theta

def is_rect_like_by_area(outer_loop, obb_w, obb_h, tol_frac=RECT_ALIGN_TOL) -> bool:
    rect_area = obb_w * obb_h
    if rect_area <= 0: return False
    poly_area = abs(polygon_area(outer_loop))
    return abs(poly_area - rect_area) <= tol_frac * rect_area

def split_outer_and_holes(loops: List[Loop]):
    if not loops: return None,[]
    idx=max(range(len(loops)), key=lambda i: abs(polygon_area(loops[i])))
    return loops[idx], [loops[i] for i in range(len(loops)) if i!=idx]

# ---------- Part ----------
class Part:
    _uid_counter = 0
    def __init__(self, name: str, loops: List[Loop], fallback_bbox: Optional[Tuple[float,float,float,float]]):
        if loops:
            minx,miny,maxx,maxy=bbox_of_loops(loops)
            loops0=[translate_loop(lp,-minx,-miny) for lp in loops]
        elif fallback_bbox is not None:
            minx,miny,maxx,maxy=fallback_bbox
            loops0=[[ (0,0),(maxx-minx,0),(maxx-minx,maxy-miny),(0,maxy-miny),(0,0) ]]
        else:
            loops0=[]
        self.name=name
        if not loops0:
            self.outer=None; self.holes=[]; self.w=self.h=0.0; self.obb_w=self.obb_h=self.obb_theta=0.0; return
        self.outer,self.holes = split_outer_and_holes(loops0)
        minx,miny,maxx,maxy=bbox_of_loops([self.outer])
        self.w=maxx-minx; self.h=maxy-miny
        self.obb_w,self.obb_h,self.obb_theta = min_area_rect(self.outer)
        self._cand_cache: Dict[Tuple[int,float,float,bool], Dict[str,Any]] = {}
        self.uid = Part._uid_counter; Part._uid_counter += 1

    def _axis_align_angles(self):
        a = (-self.obb_theta) % math.pi
        return [a, (a + math.pi/2) % math.pi]
    def is_rect_like(self) -> bool:
        return self.outer is not None and is_rect_like_by_area(self.outer, self.obb_w, self.obb_h)

    def candidate_angles(self):
        base = []
        if ROTATION_STEP_DEG and ROTATION_STEP_DEG > 0:
            step = math.radians(ROTATION_STEP_DEG)
            k = max(1, int(round(math.pi / step)))
            base = [(i*step) % math.pi for i in range(k)]
        else:
            base = [0.0]
            if ALLOW_ROTATE_90: base.append(math.pi/2)
            if USE_OBB_CANDIDATE and self.obb_w>0 and self.obb_h>0:
                a = self.obb_theta % math.pi
                base += [a, (a + math.pi/2) % math.pi]
        if RECT_ALIGN_MODE in ("prefer","force") and self.is_rect_like():
            axis = self._axis_align_angles()
            base = (axis if RECT_ALIGN_MODE=="force" else (axis + base))
        out=[]
        for a in base:
            if all(abs((a-b)%(math.pi))>math.radians(1) for b in out):
                out.append(a%(math.pi))
        return out

    def candidate_poses(self):
        angles = self.candidate_angles()
        mirrors = [False, True] if ALLOW_MIRROR else [False]
        seen=set(); poses=[]
        for mirror in mirrors:
            for ang in angles:
                key=(mirror, round((ang%(2*math.pi)),10))
                if key in seen: continue
                seen.add(key); poses.append((ang, mirror))
        return poses

    def oriented(self, theta: float, mirror: bool = False):
        if self.outer is None: return 0.0,0.0,[]
        loops_src = [self.outer] + self.holes
        if mirror: loops_src = [mirror_loop(lp) for lp in loops_src]
        if abs(theta)%(2*math.pi) > 1e-12:
            loops_src = [rotate_loop(lp, theta) for lp in loops_src]
        minx,miny,maxx,maxy=bbox_of_loops([loops_src[0]])
        return (maxx-minx),(maxy-miny),loops_src

# ---------- Raster helpers ----------
def _empty_mask(w:int, h:int): return [bytearray(w) for _ in range(h)]

def _mask_segments_and_fills(mask):
    segments=[]; fills=[]
    for row in mask:
        row_segments=[]; row_fills=[]
        start=-1
        row_len=len(row)
        for idx,val in enumerate(row):
            if val:
                if start==-1:
                    start=idx
            elif start!=-1:
                if idx>start:
                    row_segments.append((start, idx))
                    row_fills.append(b"\x01"*(idx-start))
                start=-1
        if start!=-1 and row_len>start:
            row_segments.append((start, row_len))
            row_fills.append(b"\x01"*(row_len-start))
        segments.append(row_segments)
        fills.append(row_fills)
    return segments, fills

def rasterize_polygon_to_mask(mask, w, h, pts_scaled):
    if not pts_scaled: return
    ys=[p[1] for p in pts_scaled]
    y0=max(0,int(math.floor(min(ys)))); y1=min(h-1,int(math.ceil(max(ys))))
    n=len(pts_scaled)
    for y in range(y0,y1+1):
        yscan=y+0.5; xs=[]
        for i in range(n):
            x1,y1 = pts_scaled[i]; x2,y2 = pts_scaled[(i+1)%n]
            if y1==y2: continue
            if y1>y2: x1,y1,x2,y2=x2,y2,x1,y1
            if y1 <= yscan and yscan < y2:
                t=(yscan-y1)/(y2-y1); xs.append(x1+t*(x2-x1))
        if not xs: continue
        xs.sort()
        for i in range(0,len(xs),2):
            x_start=int(math.floor(xs[i])); x_end=int(math.ceil(xs[i+1]))-1 if i+1<len(xs) else x_start
            if x_end<0 or x_start>=w: continue
            x_start=max(0,x_start); x_end=min(w-1,x_end)
            row=mask[y]
            for x in range(x_start,x_end+1): row[x]=1

def rasterize_loops(loops: List[Loop], scale: float):
    allpts=[p for lp in loops for p in lp]
    if not allpts: return _empty_mask(1,1),1,1
    minx,miny,maxx,maxy=bbox_of_points(allpts)
    loops0=[[(x-minx,y-miny) for (x,y) in lp] for lp in loops]
    pw=max(1,int(math.ceil((maxx-minx)*scale))); ph=max(1,int(math.ceil((maxy-miny)*scale)))
    mask=_empty_mask(pw,ph)
    if loops0:
        outer=loops0[0]; outer_px=[(x*scale,y*scale) for x,y in outer]
        rasterize_polygon_to_mask(mask,pw,ph,outer_px)
        for hole in loops0[1:]:
            hole_px=[(x*scale,y*scale) for x,y in hole]
            hmask=_empty_mask(pw,ph); rasterize_polygon_to_mask(hmask,pw,ph,hole_px)
            for y in range(ph):
                row=mask[y]; hr=hmask[y]
                for x in range(pw):
                    if hr[x]: row[x]=0
    return mask,pw,ph

def rasterize_outer_only(loops: List[Loop], scale: float):
    if not loops: return _empty_mask(1,1),1,1
    outer=loops[0]; minx,miny,maxx,maxy=bbox_of_points(outer)
    pw=max(1,int(math.ceil((maxx-minx)*scale))); ph=max(1,int(math.ceil((maxy-miny)*scale)))
    mask=_empty_mask(pw,ph)
    pts=[((x-minx)*scale,(y-miny)*scale) for (x,y) in outer]
    rasterize_polygon_to_mask(mask,pw,ph,pts)
    return mask,pw,ph

def dilate_mask(mask,w,h,r):
    if r<=0: return mask
    out=_empty_mask(w,h); offs=[]; rr=r*r
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            if dx*dx+dy*dy<=rr: offs.append((dx,dy))
    for y in range(h):
        row=mask[y]
        for x in range(w):
            if row[x]:
                for dx,dy in offs:
                    xx=x+dx; yy=y+dy
                    if 0<=xx<w and 0<=yy<h: out[yy][xx]=1
    return out

def _eff_scale(scale:int, spacing:float)->int:
    target = max(1, scale)
    feature = max(spacing, SAFETY_GAP, 1e-9)
    tol = max(SPACING_ABS_TOL, feature * SPACING_REL_TOL)
    target = max(target, int(math.ceil(1.0 / max(tol, 1e-9))))
    if spacing > 0:
        target = max(target, int(math.ceil(MIN_SPACING_PIXELS / max(spacing, 1e-9))))
    if SAFETY_GAP > 0:
        target = max(target, int(math.ceil(MIN_SPACING_PIXELS / max(SAFETY_GAP, 1e-9))))
    return min(target, MAX_AUTO_SCALE)

# ---------- Control / SSE Server ----------
class NestAbortPartial(Exception):
    def __init__(self, placements, sheets_used): super().__init__("Stopped by user"); self.placements=placements; self.sheets=sheets_used

class NestControl:
    def __init__(self):
        self.pause = threading.Event()
        self.stop  = threading.Event()
        self.status_lock = threading.Lock()
        self.status = {"phase":"idle"}
        self._start_lock = threading.Lock()
        self._start_event = threading.Event()
        self._start_payload: Dict[str, Any] = {}
        self._started = False
    def set_status(self, **kv):
        with self.status_lock:
            self.status.update(kv)
    def get_status(self):
        with self.status_lock:
            return dict(self.status)
    def request_start(self, payload: Dict[str, Any]):
        with self._start_lock:
            if self._started:
                return False
            self._start_payload = dict(payload or {})
            self._started = True
        self.stop.clear()
        self.pause.clear()
        self.set_status(phase="starting")
        self._start_event.set()
        return True
    def wait_for_start(self) -> Dict[str, Any]:
        self._start_event.wait()
        with self._start_lock:
            return dict(self._start_payload)

class SSEHub:
    def __init__(self): self._clients=[]; self._lock=threading.Lock()
    def attach(self, handler):
        with self._lock: self._clients.append(handler)
    def detach(self, handler):
        with self._lock:
            if handler in self._clients: self._clients.remove(handler)
    def broadcast(self, typ:str, payload:dict):
        data=json.dumps({"type":typ, **payload})
        dead=[]
        with self._lock:
            for h in self._clients:
                try:
                    h.wfile.write(b"data: "); h.wfile.write(data.encode("utf-8")); h.wfile.write(b"\n\n")
                    h.wfile.flush()
                except Exception:
                    dead.append(h)
            for h in dead:
                try: self._clients.remove(h)
                except: pass

from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

def write_standalone_html(path_on_disk: str):
    # Use doubled braces inside f-string for CSS/JS blocks.
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>DXF Nesting — Live</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  :root {{ --bg:#0b0f14; --fg:#e8eef6; --muted:#9fb3c8; --accent:#5ee1a2; --danger:#ff6b6b; --warn:#ffbb33; }}
  html,body {{ margin:0; height:100%; background:var(--bg); color:var(--fg); font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial; }}
  header {{ padding:12px 16px; display:flex; gap:12px; align-items:center; border-bottom:1px solid #223; }}
  .title {{ font-weight:700; letter-spacing:.2px; }}
  .spacer {{ flex:1; }}
  button {{ background:#1b2735; color:var(--fg); border:1px solid #2b3b52; border-radius:10px; padding:8px 12px; cursor:pointer; }}
  button:hover {{ border-color:#3f5b7a; }}
  button:active {{ transform: translateY(1px); }}
  button[disabled] {{ opacity:0.5; cursor:not-allowed; }}
  button.danger {{ border-color:var(--danger); color:var(--danger); }}
  button.accent {{ border-color:var(--accent); color:var(--accent); }}
  main {{ display:grid; grid-template-columns: 320px 1fr; min-height: calc(100% - 60px); }}
  aside {{ padding:14px; border-right:1px solid #223; }}
  section.viewer {{ position:relative; }}
  #canvasWrap {{ position:absolute; inset:0; display:flex; }}
  canvas {{ margin:auto; background:#0f1620; border:1px solid #223; border-radius:12px; }}
  .row {{ margin-bottom:12px; }}
  .label {{ color:var(--muted); font-size:12px; margin-bottom:4px; }}
  #progress {{ width:100%; height:14px; border-radius:8px; background:#121a24; overflow:hidden; border:1px solid #223; }}
  #bar {{ height:100%; width:0%; background:linear-gradient(90deg,var(--accent),#23a8f2); }}
  .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  select, input[type=number], input[type=text] {{ background:#0f1620; color:var(--fg); border:1px solid #223; border-radius:8px; padding:6px; width:100%; }}
  .small {{ font-size:12px; color:var(--muted); }}
  .pill {{ font-size:12px; padding:2px 8px; border:1px solid #2b3b52; border-radius:999px; }}
  .ok {{ border-color:var(--accent); color:var(--accent); }}
  .warn {{ border-color:var(--warn); color:var(--warn); }}
  #fieldsWrap {{ display:flex; flex-direction:column; gap:8px; margin:10px 0 8px; }}
  .fieldRow {{ display:flex; flex-direction:column; gap:4px; }}
  .fieldLabel {{ font-size:13px; color:var(--fg); }}
  .fieldHint {{ font-size:11px; color:var(--muted); }}
  #optionsWrap {{ display:flex; flex-direction:column; gap:6px; margin:6px 0 10px; }}
  .optCheck {{ display:flex; align-items:center; gap:8px; font-size:13px; }}
  .optCheck input {{ width:16px; height:16px; }}
</style>
</head>
<body>
<header>
  <div class="title">DXF Nesting — Live Viewer</div>
  <div class="pill" id="gpuPill">GPU: ?</div>
  <div class="pill" id="groupPill">Group: —</div>
  <div class="pill" id="sheetPill">Sheet: — / —</div>
  <div class="spacer"></div>
  <button id="pauseBtn">Pause</button>
  <button id="resumeBtn" class="accent">Resume</button>
  <button id="stopBtn" class="danger">Stop &amp; Save</button>
</header>
<main>
  <aside>
    <div class="row" id="configPanel">
      <div class="label">Run setup</div>
      <div class="small" id="configMsg">Loading options…</div>
      <div id="fieldsWrap"></div>
      <div id="optionsWrap"></div>
      <button id="startBtn" class="accent" disabled>Start Nesting</button>
    </div>
    <div class="row">
      <div class="label">Progress</div>
      <div id="progress"><div id="bar"></div></div>
      <div class="small"><span id="placed">0</span> / <span id="total">0</span> parts placed</div>
    </div>
    <div class="row">
      <div class="label">Status</div>
      <div class="mono" id="status">—</div>
    </div>
    <div class="row">
      <div class="label">View sheet</div>
      <select id="sheetSelect"></select>
    </div>
    <div class="row">
      <div class="label">Notes</div>
      <div class="small" id="notes">Gaps enforced (no shared-line cutting unless you turned it on).<br/>You can stop any time—current result will be saved.</div>
    </div>
    <div class="row">
      <div class="label">Outputs</div>
      <div class="small mono" id="outputs">—</div>
    </div>
  </aside>
  <section class="viewer">
    <div id="canvasWrap"><canvas id="cv" width="1280" height="720"></canvas></div>
  </section>
</main>
<script>
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
let W=48, H=96, M=0.5, scale=1;
let currentSheet = 0;
let total = 0, placed = 0;
const pauseBtn = document.getElementById('pauseBtn');
const resumeBtn = document.getElementById('resumeBtn');
const stopBtn = document.getElementById('stopBtn');
const startBtn = document.getElementById('startBtn');
const optionsWrap = document.getElementById('optionsWrap');
const fieldsWrap = document.getElementById('fieldsWrap');
const configMsg = document.getElementById('configMsg');
const sheetSel = document.getElementById('sheetSelect');

let runStarted = false;
let startRequested = false;

function updateInputDisabledState() {{
  const disabled = runStarted || startRequested;
  document.querySelectorAll('#configPanel input, #configPanel select').forEach(el => {{
    el.disabled = disabled;
  }});
}}

// Maintain sheet → paths in a Map
let sheets = new Map(); // sheetIndex -> {{paths: [ [ [x,y],... ] ], bbox:[W,H], margin:M}}

function setRunState(active) {{
  runStarted = active;
  [pauseBtn, resumeBtn, stopBtn].forEach(btn => {{ if (btn) btn.disabled = !active; }});
  updateInputDisabledState();
  if (active) {{
    startBtn.disabled = true;
    startBtn.textContent = 'Running…';
  }}
}}

function renderFields(fields) {{
  fieldsWrap.innerHTML = '';
  if (!Array.isArray(fields) || fields.length === 0) {{
    const msg = document.createElement('div');
    msg.className = 'small';
    msg.textContent = 'No configurable fields exposed.';
    fieldsWrap.appendChild(msg);
    updateInputDisabledState();
    return;
  }}
  for (const field of fields) {{
    const id = `fld_${{field.key}}`;
    const row = document.createElement('div');
    row.className = 'fieldRow';
    const label = document.createElement('label');
    label.className = 'fieldLabel';
    label.htmlFor = id;
    label.textContent = field.label || field.key;
    row.appendChild(label);
    let input;
    if (field.type === 'select' && Array.isArray(field.options)) {{
      const sel = document.createElement('select');
      sel.id = id;
      sel.dataset.key = field.key;
      sel.dataset.type = 'select';
      for (const opt of field.options) {{
        const optEl = document.createElement('option');
        optEl.value = opt.value;
        optEl.textContent = opt.label || opt.value;
        if (field.value !== undefined && field.value !== null && String(field.value) === String(opt.value)) {{
          optEl.selected = true;
        }}
        sel.appendChild(optEl);
      }}
      input = sel;
    }} else {{
      const inp = document.createElement('input');
      inp.id = id;
      inp.dataset.key = field.key;
      inp.type = field.type === 'number' ? 'number' : 'text';
      inp.dataset.type = inp.type;
      if (field.value !== undefined && field.value !== null) {{
        inp.value = field.value;
      }} else {{
        inp.value = '';
      }}
      if (field.min !== undefined) inp.min = field.min;
      if (field.max !== undefined) inp.max = field.max;
      if (field.step !== undefined) inp.step = field.step;
      if (field.placeholder) inp.placeholder = field.placeholder;
      input = inp;
    }}
    row.appendChild(input);
    if (field.description) {{
      const hint = document.createElement('div');
      hint.className = 'fieldHint';
      hint.textContent = field.description;
      row.appendChild(hint);
    }}
    fieldsWrap.appendChild(row);
  }}
  updateInputDisabledState();
}}

function renderOptions(opts) {{
  optionsWrap.innerHTML = '';
  if (!Array.isArray(opts) || opts.length === 0) {{
    const msg = document.createElement('div');
    msg.className = 'small';
    msg.textContent = 'No runtime toggles exposed.';
    optionsWrap.appendChild(msg);
    return;
  }}
  for (const opt of opts) {{
    const id = `opt_${{opt.key}}`;
    const label = document.createElement('label');
    label.className = 'optCheck';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.id = id;
    cb.dataset.key = opt.key;
    cb.checked = !!opt.value;
    const span = document.createElement('span');
    span.textContent = opt.label;
    if (opt.description) span.title = opt.description;
    label.appendChild(cb);
    label.appendChild(span);
    optionsWrap.appendChild(label);
  }}
  updateInputDisabledState();
}}

async function fetchConfig() {{
  try {{
    const resp = await fetch('/config');
    if (!resp.ok) throw new Error('HTTP '+resp.status);
    const data = await resp.json();
    if (Array.isArray(data.fields)) renderFields(data.fields);
    if (Array.isArray(data.options)) renderOptions(data.options);
    const phase = data.status && data.status.phase;
    if (!runStarted && (phase === 'waiting' || phase === 'idle')) {{
      configMsg.textContent = 'Adjust options then press Start.';
      startBtn.disabled = false;
      startBtn.textContent = 'Start Nesting';
    }}
  }} catch (err) {{
    configMsg.textContent = 'Failed to load options.';
  }}
}}

startBtn.addEventListener('click', async () => {{
  if (runStarted) return;
  const payload = {{}};
  optionsWrap.querySelectorAll('input[type=checkbox]').forEach(cb => {{
    payload[cb.dataset.key] = cb.checked;
  }});
  fieldsWrap.querySelectorAll('input, select').forEach(el => {{
    const key = el.dataset.key;
    if (!key) return;
    if (el.tagName === 'SELECT') {{
      payload[key] = el.value;
      return;
    }}
    if (el.type === 'number') {{
      payload[key] = el.value === '' ? null : el.value;
    }} else {{
      payload[key] = el.value;
    }}
  }});
  startRequested = true;
  updateInputDisabledState();
  startBtn.disabled = true;
  startBtn.textContent = 'Starting…';
  configMsg.textContent = 'Submitting…';
  try {{
    const res = await fetch('/control?cmd=start', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload)
    }});
    if (!res.ok) {{
      const text = await res.text();
      throw new Error(text || ('HTTP '+res.status));
    }}
    configMsg.textContent = 'Waiting for backend…';
  }} catch (err) {{
    startRequested = false;
    updateInputDisabledState();
    startBtn.disabled = false;
    startBtn.textContent = 'Start Nesting';
    configMsg.textContent = 'Start failed: '+err.message;
  }}
}});

pauseBtn.onclick = () => fetch('/control?cmd=pause', {{method:'POST'}});
resumeBtn.onclick = () => fetch('/control?cmd=resume', {{method:'POST'}});
stopBtn.onclick = () => fetch('/control?cmd=stop', {{method:'POST'}});

setRunState(false);
fetchConfig();

function fitScale() {{
  const pad = 20;
  const availW = cv.width - pad*2;
  const availH = cv.height - pad*2;
  const drawW = W + 2*M; const drawH = H + 2*M;
  scale = Math.min(availW/drawW, availH/drawH);
}}
function toXY(x,y) {{
  // origin bottom-left visual (flip Y)
  const pad = 20;
  const X = pad + (x)*scale;
  const Y = cv.height - (pad + (y)*scale);
  return [X,Y];
}}
function redraw() {{
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.fillStyle = '#0f1620'; ctx.fillRect(0,0,cv.width,cv.height);
  fitScale();
  // frame
  ctx.strokeStyle='#445'; ctx.lineWidth=2;
  let [x0,y0] = toXY(0,0);
  let [x1,y1] = toXY(W+2*M,0);
  let [x2,y2] = toXY(W+2*M,H+2*M);
  let [x3,y3] = toXY(0,H+2*M);
  ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.lineTo(x2,y2); ctx.lineTo(x3,y3); ctx.closePath(); ctx.stroke();
  // parts on current sheet
  const data = sheets.get(currentSheet);
  if (!data) return;
  ctx.strokeStyle='#5ee1a2'; ctx.lineWidth=1;
  for (const lp of data.paths) {{
    ctx.beginPath();
    for (let i=0;i<lp.length;i++) {{
      const [X,Y] = toXY(M+lp[i][0], M+lp[i][1]);
      if (i===0) ctx.moveTo(X,Y); else ctx.lineTo(X,Y);
    }}
    ctx.stroke();
  }}
}}
function setProgress(pPlaced, pTotal) {{
  placed=pPlaced; total=pTotal;
  document.getElementById('placed').textContent=placed;
  document.getElementById('total').textContent=pTotal;
  const pct = pTotal>0 ? (100*placed/pTotal) : 0;
  document.getElementById('bar').style.width = pct.toFixed(1)+'%';
}}
function ensureSheet(i) {{
  if (!sheets.has(i)) sheets.set(i, {{paths:[], bbox:[W,H], margin:M}});
  if (![...sheetSel.options].some(o => Number(o.value)===i)) {{
    const opt=document.createElement('option'); opt.value=String(i); opt.textContent='Sheet '+(i+1); sheetSel.appendChild(opt);
  }}
}}
sheetSel.addEventListener('change', () => {{ currentSheet = Number(sheetSel.value)||0; redraw(); }});

function setStatus(t) {{ document.getElementById('status').textContent = t; }}
function setGPU(ok) {{
  const pill=document.getElementById('gpuPill');
  pill.textContent='GPU: '+(ok?'ON':'OFF');
  pill.className = 'pill '+(ok?'ok':'warn');
}}
function setGroup(g) {{ const el=document.getElementById('groupPill'); el.textContent='Group: '+(g||'—'); }}
function setSheetPill(cur, total) {{ const el=document.getElementById('sheetPill'); el.textContent='Sheet: '+(cur+1)+' / '+(total||'—'); }}

const es = new EventSource('/events');
es.onmessage = (ev) => {{
  try {{
    const msg = JSON.parse(ev.data||'{{}}');
    if (msg.type==='waiting') {{
      startRequested = false;
      setRunState(false);
      if (Array.isArray(msg.fields)) renderFields(msg.fields);
      if (Array.isArray(msg.options)) renderOptions(msg.options);
      const text = msg.message || 'Waiting for Start…';
      configMsg.textContent = text;
      startBtn.disabled = false;
      startBtn.textContent = 'Start Nesting';
      setStatus(text);
      return;
    }}
    if (msg.type==='starting') {{
      startRequested = true;
      updateInputDisabledState();
      configMsg.textContent = 'Starting…';
      startBtn.disabled = true;
      startBtn.textContent = 'Starting…';
      setStatus('Starting…');
      return;
    }}
    if (msg.type==='options_applied') {{
      if (Array.isArray(msg.fields)) renderFields(msg.fields);
      if (Array.isArray(msg.options)) renderOptions(msg.options);
      return;
    }}
    if (msg.type==='hello') {{
      setGPU(!!msg.cuda);
      return;
    }}
    if (msg.type==='start') {{
      W=msg.sheet_w; H=msg.sheet_h; M=msg.margin; setGroup(msg.group||'—'); setStatus('Starting…');
      sheets.clear(); sheetSel.innerHTML=''; currentSheet=0; setProgress(0,msg.total_parts||0); redraw();
      // show first sheet slot immediately
      setRunState(true);
      configMsg.textContent = 'Run in progress…';
      ensureSheet(0); setSheetPill(0, 1); redraw();
      return;
    }}
    if (msg.type==='group') {{
      setGroup(msg.group||'—'); setStatus('Group '+(msg.group||'—')); setProgress(0,msg.total_parts||0);
      sheets.clear(); sheetSel.innerHTML=''; currentSheet=0; ensureSheet(0); redraw();
      return;
    }}
    if (msg.type==='sheet_opened') {{
      ensureSheet(msg.sheet_index||0); setSheetPill(msg.sheet_index||0, msg.total_sheets||0);
      setStatus('Opened new sheet '+(1+(msg.sheet_index||0)));
      redraw();
      return;
    }}
    if (msg.type==='place') {{
      ensureSheet(msg.sheet);
      const rec = sheets.get(msg.sheet);
      if (msg.loops) for (const lp of msg.loops) rec.paths.push(lp);
      setProgress(msg.placed||0, msg.total||0);
      setStatus('Placed: '+(msg.part||'part')+'  (sheet '+(msg.sheet+1)+')');
      redraw();
      return;
    }}
    if (msg.type==='progress') {{
      if (msg.text) setStatus(msg.text);
      return;
    }}
    if (msg.type==='done' || msg.type==='stopped') {{
      setStatus(msg.type==='done' ? 'Completed.' : 'Stopped — partial saved.');
      setRunState(false);
      startBtn.disabled = true;
      configMsg.textContent = msg.type==='done' ? 'Completed.' : 'Stopped — partial saved.';
      if (msg.outputs) {{
        document.getElementById('outputs').textContent = (msg.outputs||[]).map(o=>o[0]+'  (sheets:'+o[1]+')').join('\\n') || '—';
      }}
      return;
    }}
  }} catch(e) {{ console.warn(e); }}
}};
</script>
</body></html>"""
    with open(path_on_disk, "w", encoding="utf-8") as f:
        f.write(html)

class NestHTTPHandler(BaseHTTPRequestHandler):
    hub:SSEHub = None
    control:NestControl = None
    folder:str = None
    ui_path:str = None
    cuda_on:bool = False

    def _set_headers(self, code=200, ctype="text/html; charset=utf-8", extra=None):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        if extra:
            for k,v in (extra.items() if isinstance(extra,dict) else []): self.send_header(k,v)
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path)
        if p.path in ("/", "/index.html"):
            try:
                with open(self.ui_path, "r", encoding="utf-8") as f:
                    data = f.read().encode("utf-8")
                self._set_headers(200, "text/html; charset=utf-8"); self.wfile.write(data)
            except Exception as e:
                self._set_headers(500, "text/plain"); self.wfile.write(str(e).encode("utf-8"))
            return

        if p.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                # hello
                hello = json.dumps({"type":"hello","cuda":self.cuda_on})
                self.wfile.write(b"data: "); self.wfile.write(hello.encode("utf-8")); self.wfile.write(b"\n\n"); self.wfile.flush()
                # attach
                self.hub.attach(self)
                # keep open
                while True:
                    time.sleep(15)
                    try:
                        self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                    except Exception:
                        break
            finally:
                self.hub.detach(self)
            return

        if p.path == "/status":
            st = self.control.get_status()
            self._set_headers(200, "application/json"); self.wfile.write(json.dumps(st).encode("utf-8")); return

        if p.path == "/config":
            payload = {
                "options": _ui_toggle_snapshot(),
                "fields": _ui_field_snapshot(),
                "status": self.control.get_status(),
            }
            self._set_headers(200, "application/json"); self.wfile.write(json.dumps(payload).encode("utf-8")); return

        self._set_headers(404, "text/plain"); self.wfile.write(b"Not found")

    def do_POST(self):
        p = urlparse(self.path)
        if p.path == "/control":
            qs = parse_qs(p.query or "")
            cmd = (qs.get("cmd",[""])[0] or "").lower()
            if cmd == "pause":
                self.control.pause.set(); self._set_headers(200,"text/plain"); self.wfile.write(b"OK"); return
            if cmd == "resume":
                self.control.pause.clear(); self._set_headers(200,"text/plain"); self.wfile.write(b"OK"); return
            if cmd == "stop":
                self.control.stop.set();  self._set_headers(200,"text/plain"); self.wfile.write(b"OK"); return
            if cmd == "start":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except Exception:
                    length = 0
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                try:
                    payload = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    self._set_headers(400, "text/plain"); self.wfile.write(b"Invalid JSON"); return
                if not isinstance(payload, dict):
                    self._set_headers(400, "text/plain"); self.wfile.write(b"JSON body must be an object"); return
                if not self.control.request_start(payload):
                    self._set_headers(409, "text/plain"); self.wfile.write(b"Already started"); return
                self._set_headers(200, "application/json"); self.wfile.write(json.dumps({"status":"starting"}).encode("utf-8"))
                self.hub.broadcast("starting", {"options": payload})
                return
            self._set_headers(400,"text/plain"); self.wfile.write(b"Bad command"); return
        self._set_headers(404,"text/plain"); self.wfile.write(b"Not found")

def start_http_server(folder:str, ui_filename:str, cuda_on:bool, control:NestControl, hub:SSEHub,
                      port:int=0, host:str="127.0.0.1"):
    ui_path = None
    ui_dir_candidates = []
    if folder and os.path.isdir(folder):
        ui_dir_candidates.append(folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in ui_dir_candidates:
        ui_dir_candidates.append(script_dir)
    cwd = os.getcwd()
    if cwd not in ui_dir_candidates:
        ui_dir_candidates.append(cwd)

    last_exc = None
    for candidate in ui_dir_candidates:
        try:
            ui_path = os.path.join(candidate, ui_filename)
            write_standalone_html(ui_path)
            break
        except Exception as exc:
            last_exc = exc
            ui_path = None
    if ui_path is None:
        raise last_exc or RuntimeError("Unable to write UI HTML file")

    class _Server(ThreadingHTTPServer): daemon_threads=True
    NestHTTPHandler.hub = hub
    NestHTTPHandler.control = control
    NestHTTPHandler.folder = folder
    NestHTTPHandler.ui_path = ui_path
    NestHTTPHandler.cuda_on = cuda_on
    srv = _Server((host, port), NestHTTPHandler)
    host_bound, real_port = srv.server_address
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, host_bound, real_port

# ---------- placement (with live events + pause/stop checks) ----------
def bl_place(occ, mask_segments, pw):
    H=len(occ); W=len(occ[0]) if H>0 else 0
    ph=len(mask_segments)
    if H < ph or W < pw:
        return None
    max_y = H - ph + 1
    max_x = W - pw + 1
    for y in range(max_y):
        rows = occ[y:y+ph]
        x = 0
        while x < max_x:
            skip = 1
            blocked = False
            for yy, segs in enumerate(mask_segments):
                if not segs:
                    continue
                row = rows[yy]
                for start, end in segs:
                    hit = row.find(1, x + start, x + end)
                    if hit != -1:
                        skip = max(skip, hit - (x + start) + 1)
                        blocked = True
                        break
                if blocked:
                    break
            if not blocked:
                return (x, y)
            x += skip
    return None

def or_mask_inplace(occ, mask_segments, mask_fills, ox, oy):
    for y,segs in enumerate(mask_segments):
        if not segs: continue
        row=occ[oy+y]
        fills=mask_fills[y]
        for (start,end),fill in zip(segs,fills):
            if not fill:
                continue
            dst_start=ox+start
            row[dst_start:dst_start + (end-start)] = fill

def pack_bitmap_core(ordered_parts: List['Part'], W: float, H: float, spacing: float, scale: int,
                     progress=None, progress_total=None, progress_prefix="",
                     mask_ops: Optional[Any] = None,
                     control: Optional[NestControl] = None,
                     event_sink: Optional[callable] = None):
    Wpx=max(1,int(math.ceil(W*scale))); Hpx=max(1,int(math.ceil(H*scale)))
    spacing_px=int(math.ceil(max(0.0, spacing)*scale))
    safety_px = SAFETY_PX
    if SAFETY_GAP > 0:
        safety_px = max(safety_px, int(math.ceil(SAFETY_GAP * scale)))
    safety_units = max(SAFETY_GAP, safety_px / float(scale))
    sheets_occ_raw=[]; sheets_occ_safe=[]; sheets_out=[]; sheets_count=0
    def ensure_sheet():
        nonlocal sheets_count
        if len(sheets_occ_raw)<=sheets_count:
            if mask_ops:
                sheets_occ_raw.append(mask_ops.zeros(Hpx,Wpx)); sheets_occ_safe.append(mask_ops.zeros(Hpx,Wpx))
            else:
                sheets_occ_raw.append(_empty_mask(Wpx,Hpx));     sheets_occ_safe.append(_empty_mask(Wpx,Hpx))
            sheets_out.append([])
        return sheets_occ_raw[sheets_count], sheets_occ_safe[sheets_count], sheets_out[sheets_count]

    placed_count=0; total_parts=progress_total if (progress_total is not None) else len(ordered_parts)

    def check_ctrl():
        if control:
            while control.pause.is_set(): time.sleep(0.05)
            if control.stop.is_set():
                partial=[{'sheet': i, 'loops': pl['loops']} for i,out in enumerate(sheets_out) for pl in out]
                used=max((pl['sheet'] for out in sheets_out for pl in out), default=-1)+1
                raise NestAbortPartial(partial, used)

    for p in ordered_parts:
        check_ctrl()
        placed=False
        for ang,mirror in p.candidate_poses():
            check_ctrl()
            key=(scale,ang,mirror)

            if key not in p._cand_cache:
                w,h,loops=p.oriented(ang,mirror)
                raw,raw_w,raw_h=rasterize_loops(loops,scale)
                outer,outer_w,outer_h=rasterize_outer_only(loops,scale)
                if ALLOW_NEST_IN_HOLES:
                    base_loops = loops
                    base_mask, base_w, base_h = raw, raw_w, raw_h
                else:
                    base_loops = [loops[0]] if loops else []
                    base_mask, base_w, base_h = outer, outer_w, outer_h

                precise_test = _offset_loops_precise(base_loops, max(0.0, spacing) + safety_units)
                precise_occ = _offset_loops_precise(base_loops, safety_units)
                if precise_test is not None and precise_occ is not None:
                    test,test_w,test_h=rasterize_loops(precise_test, scale)
                    occ_pad,occ_w,occ_h=rasterize_loops(precise_occ, scale)
                else:
                    _warn_precise_offsets_fallback()
                    test=dilate_mask(base_mask, base_w, base_h, spacing_px + safety_px)
                    occ_pad=dilate_mask(base_mask, base_w, base_h, safety_px)
                    test_w,test_h = base_w, base_h
                    occ_w,occ_h = base_w, base_h
                test_segments,_=_mask_segments_and_fills(test)
                raw_segments,raw_fills=_mask_segments_and_fills(raw)
                occ_segments,occ_fills=_mask_segments_and_fills(occ_pad)
                p._cand_cache[key] = {
                    'loops':loops,
                    'raw':raw,
                    'occ':occ_pad,
                    'test':test,
                    'pw':test_w,
                    'ph':test_h,
                    'test_segments':test_segments,
                    'raw_segments':raw_segments,
                    'raw_fills':raw_fills,
                    'occ_segments':occ_segments,
                    'occ_fills':occ_fills
                }
            cand=p._cand_cache[key]
            if mask_ops:
                if 'test_tensor' not in cand: cand['test_tensor']=mask_ops.mask_to_tensor(cand['test'])
                if 'occ_tensor'  not in cand: cand['occ_tensor'] =mask_ops.mask_to_tensor(cand['occ'])
                if 'raw_tensor'  not in cand: cand['raw_tensor'] =mask_ops.mask_to_tensor(cand['raw'])
            attempt_sheet=sheets_count
            while True:
                check_ctrl()
                occ_raw,occ_safe,outlist=ensure_sheet()
                pos = (mask_ops.find_first_fit(occ_safe, cand['test_tensor']) if mask_ops
                       else bl_place(occ_safe, cand['test_segments'], cand['pw']))
                if pos is not None:
                    xpx,ypx=pos
                    if mask_ops:
                        mask_ops.or_mask(occ_raw, cand['raw_tensor'], xpx, ypx)
                        mask_ops.or_mask(occ_safe,cand['occ_tensor'], xpx, ypx)
                    else:
                        or_mask_inplace(occ_raw, cand['raw_segments'], cand['raw_fills'], xpx, ypx)
                        or_mask_inplace(occ_safe,cand['occ_segments'], cand['occ_fills'], xpx, ypx)
                    x_units=xpx/scale; y_units=ypx/scale
                    loops_t=[[ (x+x_units,y+y_units) for (x,y) in lp ] for lp in cand['loops']]
                    outlist.append({'sheet':sheets_count,'loops':loops_t})
                    placed=True; placed_count+=1
                    if event_sink:
                        event_sink("place", {"sheet":sheets_count,"loops":loops_t,"part":os.path.basename(p.name),
                                             "placed":placed_count,"total":total_parts})
                    if progress:
                        progress(f"{progress_prefix}Placing parts…\nPlaced: {placed_count}/{total_parts}\nCurrent sheet: {sheets_count+1}\nPart: {os.path.basename(p.name)}")
                    break
                else:
                    sheets_count+=1
                    if event_sink:
                        event_sink("sheet_opened", {"sheet_index": sheets_count})
                    if progress:
                        progress(f"{progress_prefix}Opening new sheet… now {sheets_count+1}\nPlaced: {placed_count}/{total_parts}")
                    if sheets_count>attempt_sheet+25: break
            if placed: break

        if not placed:
            sheets_count+=1
            occ_raw,occ_safe,outlist=ensure_sheet()

            ang,mirror=0.0,False; key=(scale,ang,mirror)

            if key not in p._cand_cache:
                w,h,loops=p.oriented(ang,mirror)
                raw,raw_w,raw_h=rasterize_loops(loops,scale)
                outer,outer_w,outer_h=rasterize_outer_only(loops,scale)
                if ALLOW_NEST_IN_HOLES:
                    base_loops = loops
                    base_mask, base_w, base_h = raw, raw_w, raw_h
                else:
                    base_loops = [loops[0]] if loops else []
                    base_mask, base_w, base_h = outer, outer_w, outer_h

                precise_test = _offset_loops_precise(base_loops, max(0.0, spacing) + safety_units)
                precise_occ = _offset_loops_precise(base_loops, safety_units)
                if precise_test is not None and precise_occ is not None:
                    test,test_w,test_h=rasterize_loops(precise_test, scale)
                    occ_pad,occ_w,occ_h=rasterize_loops(precise_occ, scale)
                else:
                    _warn_precise_offsets_fallback()
                    test=dilate_mask(base_mask, base_w, base_h, spacing_px + safety_px)
                    occ_pad=dilate_mask(base_mask, base_w, base_h, safety_px)
                    test_w,test_h = base_w, base_h
                    occ_w,occ_h = base_w, base_h
                test_segments,_=_mask_segments_and_fills(test)
                raw_segments,raw_fills=_mask_segments_and_fills(raw)
                occ_segments,occ_fills=_mask_segments_and_fills(occ_pad)
                p._cand_cache[key] = {
                    'loops':loops,
                    'raw':raw,
                    'occ':occ_pad,
                    'test':test,
                    'pw':test_w,
                    'ph':test_h,
                    'test_segments':test_segments,
                    'raw_segments':raw_segments,
                    'raw_fills':raw_fills,
                    'occ_segments':occ_segments,
                    'occ_fills':occ_fills
                }
            cand=p._cand_cache[key]
            if mask_ops:
                if 'raw_tensor' not in cand: cand['raw_tensor']=mask_ops.mask_to_tensor(cand['raw'])
                if 'occ_tensor' not in cand: cand['occ_tensor']=mask_ops.mask_to_tensor(cand['occ'])
                mask_ops.or_mask(occ_raw,cand['raw_tensor'],0,0)
                mask_ops.or_mask(occ_safe,cand['occ_tensor'],0,0)
            else:
                or_mask_inplace(occ_raw,cand['raw_segments'],cand['raw_fills'],0,0)
                or_mask_inplace(occ_safe,cand['occ_segments'],cand['occ_fills'],0,0)
            outlist.append({'sheet':sheets_count,'loops':p._cand_cache[key]['loops']})
            placed_count+=1
            if event_sink:
                event_sink("place", {"sheet":sheets_count,"loops":p._cand_cache[key]['loops'],
                                     "part":os.path.basename(p.name),"placed":placed_count,"total":total_parts})
            if progress:
                progress(f"Forced place on new sheet {sheets_count+1}\nPlaced: {placed_count}/{total_parts}")

    used_sheets = max((pl['sheet'] for out in sheets_out for pl in out), default=-1)+1
    if mask_ops:
        fill_pixels = 0
        for occ in sheets_occ_raw: fill_pixels += mask_ops.count_true(occ)
    else:
        fill_pixels = sum(sum(1 for v in row if v) for occ in sheets_occ_raw for row in occ)

    placements=[{'sheet':i,'loops':pl['loops']} for i,out in enumerate(sheets_out) for pl in out]
    return placements, used_sheets, fill_pixels

def _seq_key(order: List['Part']): return tuple(p.uid for p in order)

def _result_is_better(candidate, incumbent):
    if candidate is None: return False
    if incumbent is None: return True
    _, cs, cf = candidate; _, is_, if_ = incumbent
    if cs != is_: return cs < is_
    return cf > if_

def _mutate_order(order: List['Part'], rnd: Random) -> List['Part']:
    n=len(order)
    if n<=1: return list(order)
    op=rnd.random()
    if n==2: op=0.0
    if op<0.4:
        i,j=rnd.sample(range(n),2); new=list(order); new[i],new[j]=new[j],new[i]; return new
    elif op<0.75:
        i,j=rnd.sample(range(n),2); new=list(order); part=new.pop(i); new.insert(j,part); return new
    else:
        i,j=sorted(rnd.sample(range(n),2)); new=list(order); new[i:j+1]=reversed(new[i:j+1]); return new

def _anneal_order(initial_order: List['Part'], evaluate_fn, rnd: Random, sheet_penalty: int,
                  progress=None, label="", max_iters: Optional[int] = None, control: Optional[NestControl]=None):
    order=list(initial_order); best_order=list(order); best_result=evaluate_fn(best_order, allow_progress=False)
    current_order=list(order); current_result=best_result
    n=len(order)
    if n<=1: return best_order, best_result
    default_iters=max(8,min(24,n+4)); base_iters=max(5,min(default_iters,max_iters)) if max_iters is not None else default_iters
    temperature=max(1.0,n*0.4); cooling=0.9; stall_limit=None; stall=0
    def score(res):
        if res is None: return float('inf')
        _,sh,fi=res; return sh*sheet_penalty - fi
    for it in range(1, base_iters+1):
        if control and control.stop.is_set(): break
        while control and control.pause.is_set(): time.sleep(0.05)
        cand_order=_mutate_order(current_order,rnd)
        cand_result=evaluate_fn(cand_order, allow_progress=False)
        if _result_is_better(cand_result,current_result):
            current_order,current_result=cand_order,cand_result
        else:
            delta=score(cand_result)-score(current_result)
            accept_prob=1.0 if delta<0 else (math.exp(-delta/temperature) if temperature>0 else 0.0)
            if accept_prob>rnd.random(): current_order,current_result=cand_order,cand_result
        if _result_is_better(current_result,best_result):
            best_order,best_result=list(current_order),current_result
            if progress: progress(f"{label}Anneal improvement: sheets={best_result[1]}, fill={best_result[2]}")
            stall=0
        else: stall+=1
        if progress and it % max(6, base_iters//3)==0:
            progress(f"{label}Anneal {it}/{base_iters}: best sheets={best_result[1]}, fill={best_result[2]}")
        temperature*=cooling
        if temperature<1e-4: temperature=1e-4
        if stall_limit is None: stall_limit=max(3, base_iters//2)
        if stall>=stall_limit: break
    return best_order, best_result

def pack_bitmap_multi(parts: List['Part'], W: float, H: float, spacing: float, scale: int,
                      tries: int, seed: Optional[int], progress=None,
                      mask_ops: Optional[Any] = None,
                      control: Optional[NestControl]=None,
                      event_sink: Optional[callable]=None):
    base=[p for p in parts if p.outer is not None]
    base.sort(key=lambda p: abs(polygon_area(p.outer)), reverse=True)
    rnd=Random(seed) if seed is not None else Random()
    total_parts=len(base)
    if total_parts==0: return [], 0
    search_scale=scale
    if scale > 6:
        search_scale = max(6, scale // 2)  # coarse for trials → faster

    Wpx=max(1,int(math.ceil(W*scale))); Hpx=max(1,int(math.ceil(H*scale))); sheet_penalty=Wpx*Hpx*1000
    cache: Dict[Tuple[tuple,int,float], Tuple[List[dict],int,int]] = {}

    def evaluate(order: List['Part'], allow_progress: bool, prefix: str = "", use_scale: int = search_scale,
                 spacing_override: Optional[float] = None):
        eff_spacing = spacing if spacing_override is None else spacing_override
        spacing_tag = round(max(0.0, eff_spacing), 9)
        key=(_seq_key(order),use_scale,spacing_tag)
        if key in cache: return cache[key]
        allow_events = (spacing_override is None) and allow_progress and event_sink is not None
        res=pack_bitmap_core(order,W,H,eff_spacing,use_scale,
                             progress=(progress if allow_progress else None),
                             progress_total=total_parts if allow_progress else None,
                             progress_prefix=prefix if allow_progress else "",
                             mask_ops=mask_ops,
                             control=control,
                             event_sink=(event_sink if allow_events else None))
        cache[key]=res; return res

    best_result=None; best_order=None
    heuristic=[("Area-desc ", list(base)),
               ("Aspect-desc ", sorted(base, key=lambda p: max(p.w,p.h,p.obb_w,p.obb_h), reverse=True)),
               ("Tall-first ",  sorted(base, key=lambda p: p.h, reverse=True))]
    tries=max(1,tries)
    starts=[]
    for ho in heuristic:
        if len(starts)>=tries: break
        starts.append(ho)
    while len(starts)<tries:
        idx=len(starts)-len(heuristic)+1
        starts.append((f"Random {max(1,idx)} ", rnd.sample(base, len(base))))

    attempts=max(1,len(starts))
    anneal_limit=max(4,min(8,total_parts+max(1,tries//2)))
    last_start=None
    for t,(label,start_order) in enumerate(starts):
        if progress: progress(f"{label}placement trial {t+1}/{attempts}…")
        _ = evaluate(start_order, allow_progress=False, prefix=f"{label}Try {t+1}/{attempts}\n",
                     use_scale=search_scale, spacing_override=0.0)
        last_start=evaluate(start_order, allow_progress=False, prefix=f"{label}Try {t+1}/{attempts}\n", use_scale=search_scale)
        if anneal_limit<=0:
            order_after,result_after=start_order,last_start
        else:
            limit=anneal_limit if t==0 else (min(3,anneal_limit) if t<len(heuristic) else min(4,anneal_limit))
            if limit<=1:
                order_after,result_after=start_order,last_start
            else:
                order_after,_=_anneal_order(start_order,
                    lambda o, allow_progress=False: evaluate(o, allow_progress, prefix=label,
                                                             use_scale=search_scale, spacing_override=0.0),
                    rnd, sheet_penalty, progress=progress, label=label, max_iters=limit, control=control)
                result_after=evaluate(order_after, allow_progress=False, prefix=label, use_scale=search_scale)
        final = result_after if _result_is_better(result_after,last_start) else last_start
        final_order = order_after if final is result_after else start_order
        if _result_is_better(final,best_result):
            best_result=final; best_order=final_order
            if progress: progress(f"{label}New global best: sheets={best_result[1]}, fill={best_result[2]}")
        elif progress and best_result:
            progress(f"{label}Result sheets={final[1]}, fill={final[2]} (best remains sheets={best_result[1]}, fill={best_result[2]})")
    if best_result is None:
        best_result=last_start; best_order=starts[0][1] if starts else base
    final_order=best_order if best_order is not None else base
    final_result=evaluate(final_order, allow_progress=True, prefix="Final pass\n", use_scale=scale)
    return final_result[0], final_result[1]

# ---------- Shelf fallback ----------
def pack_shelves(parts: List['Part'], W: float, H: float, spacing: float,
                 control: Optional[NestControl]=None, event_sink: Optional[callable]=None):
    parts=sorted([p for p in parts if p.outer is not None], key=lambda p: max(p.w,p.h,p.obb_w,p.obb_h), reverse=True)
    placements=[]; sheet=0; shelf_y=0.0; shelf_h=0.0; cursor_x=0.0
    def new_sheet():
        nonlocal sheet,shelf_y,shelf_h,cursor_x
        sheet+=1; shelf_y=0.0; shelf_h=0.0; cursor_x=0.0
        if event_sink: event_sink("sheet_opened", {"sheet_index": sheet})
    for idx,p in enumerate(parts,1):
        if control and control.stop.is_set():
            raise NestAbortPartial(placements, (max((pl['sheet'] for pl in placements), default=-1))+1 )
        while control and control.pause.is_set(): time.sleep(0.05)
        cands=[]
        for ang, mirror in p.candidate_poses():
            w,h,_=p.oriented(ang, mirror); cands.append((ang, mirror, w, h))
        placed=False
        for ang, mirror, w, h in cands:
            if cursor_x + w + spacing <= W and shelf_y + max(shelf_h, h + spacing) <= H:
                _,_,loops = p.oriented(ang, mirror)
                loops_t=[[(x+cursor_x,y+shelf_y) for x,y in lp] for lp in loops]
                placements.append({'sheet':sheet,'loops':loops_t})
                if event_sink: event_sink("place", {"sheet":sheet,"loops":loops_t,"part":os.path.basename(p.name),"placed":idx,"total":len(parts)})
                cursor_x += w + spacing; shelf_h = max(shelf_h, h + spacing)
                placed=True; break
        if placed: continue
        shelf_y += shelf_h; cursor_x = 0.0; shelf_h = 0.0
        for ang, mirror, w, h in cands:
            if w + spacing <= W and shelf_y + h + spacing <= H:
                _,_,loops = p.oriented(ang, mirror)
                loops_t=[[(x+0.0,y+shelf_y) for x,y in lp] for lp in loops]
                placements.append({'sheet':sheet,'loops':loops_t})
                if event_sink: event_sink("place", {"sheet":sheet,"loops":loops_t,"part":os.path.basename(p.name),"placed":idx,"total":len(parts)})
                cursor_x = w + spacing; shelf_h = h + spacing
                placed=True; break
        if placed: continue
        new_sheet()
        ok=False
        for ang, mirror, w, h in cands:
            if w + spacing <= W and h + spacing <= H:
                _,_,loops = p.oriented(ang, mirror)
                loops_t=[[(x+0.0,y+0.0) for x,y in lp] for lp in loops]
                placements.append({'sheet':sheet,'loops':loops_t})
                if event_sink: event_sink("place", {"sheet":sheet,"loops":loops_t,"part":os.path.basename(p.name),"placed":idx,"total":len(parts)})
                cursor_x = w + spacing; shelf_h = h + spacing
                ok=True; break
        if not ok:
            _,_,loops = p.oriented(0.0, False)
            loops_t=[[(x,y) for x,y in lp] for lp in loops]
            placements.append({'sheet':sheet,'loops':loops_t})
            if event_sink: event_sink("place", {"sheet":sheet,"loops":loops_t,"part":os.path.basename(p.name),"placed":idx,"total":len(parts)})
            cursor_x = p.w + spacing; shelf_h = p.h + spacing
    sheets_used=(max((pl['sheet'] for pl in placements), default=-1))+1
    return placements, sheets_used

# ---------- Line merging & writers ----------
def merge_common_lines(placements: List[dict], tol=1e-4) -> List[dict]:
    def norm_seg(a,b):
        ax,ay=a; bx,by=b
        if (bx<ax) or (abs(bx-ax)<=tol and by<ay): ax,ay,bx,by=bx,by,ax,ay
        return (round(ax/tol), round(ay/tol), round(bx/tol), round(by/tol))
    keep={}
    for pl in placements:
        new_loops=[]
        for lp in pl['loops']:
            segs=[]
            for i in range(len(lp)-1):
                a=lp[i]; b=lp[i+1]
                key=norm_seg(a,b); keep[key]=keep.get(key,0)+1; segs.append((a,b))
            new_loops.append(segs)
        pl['__segs__']=new_loops
    out=[]
    for pl in placements:
        lines=[]
        for segs in pl['__segs__']:
            for a,b in segs:
                key=norm_seg(a,b)
                if keep.get(key,0)==1: lines.append((a,b))
        out.append({'sheet': pl['sheet'], 'lines': lines})
    return out

def write_r12_dxf(path, sheets, W, H, placements, margin, merge_lines=False):
    def w(f,c,v): f.write(f"{c}\n{v}\n")
    with open(path,'w',encoding='utf-8') as f:
        w(f,0,"SECTION"); w(f,2,"HEADER"); w(f,9,"$INSUNITS"); w(f,70,INSUNITS); w(f,0,"ENDSEC")
        w(f,0,"SECTION"); w(f,2,"TABLES"); w(f,0,"ENDSEC")
        w(f,0,"SECTION"); w(f,2,"ENTITIES")
        for s in range(sheets):
            sheet_ox = s*(W + 2*margin + SHEET_GAP)
            w(f,0,"POLYLINE"); w(f,8,"SHEET"); w(f,66,1); w(f,70,1)
            for x,y in [(sheet_ox,0),(sheet_ox+W+2*margin,0),(sheet_ox+W+2*margin,H+2*margin),(sheet_ox,H+2*margin),(sheet_ox,0)]:
                w(f,0,"VERTEX"); w(f,8,"SHEET"); w(f,10,x); w(f,20,y)
            w(f,0,"SEQEND")
        if merge_lines:
            merged = merge_common_lines(placements)
            for pl in merged:
                ox = pl['sheet']*(W + 2*margin + SHEET_GAP) + margin
                oy = margin
                for (a,b) in pl['lines']:
                    w(f,0,"LINE"); w(f,8,"NEST")
                    w(f,10,a[0]+ox); w(f,20,a[1]+oy)
                    w(f,11,b[0]+ox); w(f,21,b[1]+oy)
        else:
            for pl in placements:
                ox = pl['sheet']*(W + 2*margin + SHEET_GAP) + margin
                oy = margin
                for lp in pl['loops']:
                    w(f,0,"POLYLINE"); w(f,8,"NEST"); w(f,66,1); w(f,70,1)
                    for x,y in ((x+ox,y+oy) for x,y in lp):
                        w(f,0,"VERTEX"); w(f,8,"NEST"); w(f,10,x); w(f,20,y)
                    w(f,0,"SEQEND")
        w(f,0,"ENDSEC"); w(f,0,"EOF")

def write_split_sheets(base_path_no_ext: str, placements: List[dict], total_sheets: int,
                       W: float, H: float, margin: float, merge_lines=False):
    for s in range(total_sheets):
        sub=[{'sheet':0,'loops':pl['loops']} for pl in placements if pl['sheet']==s]
        out=f"{base_path_no_ext}-s{s+1}.dxf"
        write_r12_dxf(out,1,W,H,sub,margin,merge_lines=merge_lines)

# ---------- main ----------
def main_live():
    prog = WinProgress("Nesting DXF… (HTML live viewer)", 520, 220); prog.create()

    control = NestControl()
    hub = SSEHub()

    # start HTTP server + open viewer (GPU status filled in after configuration)
    srv, bound_host, port = start_http_server(FOLDER, UI_FILENAME, False, control, hub, HTTP_PORT, HTTP_HOST)
    open_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::", "") else bound_host
    url=f"http://{open_host}:{port}/"
    try: webbrowser.open(url)
    except: pass
    if bound_host != open_host:
        log(f"[INFO] Live viewer at: {url} (server bound to {bound_host})")
    else:
        log(f"[INFO] Live viewer at: {url}")

    wait_text = "Viewer ready — adjust options in the browser and press Start."
    control.set_status(phase="waiting")
    hub.broadcast("waiting", {"message": wait_text, "options": _ui_toggle_snapshot(), "fields": _ui_field_snapshot()})
    prog.update(wait_text)

    # status helpers
    def progress_cb(text: str):
        prog.update(text)
        hub.broadcast("progress", {"text": text})

    def event_sink(kind: str, payload: dict):
        if kind=="place":
            hub.broadcast("place", payload)
        elif kind=="sheet_opened":
            hub.broadcast("sheet_opened", payload)

    # Wait for the UI to kick off the run
    start_config = control.wait_for_start()
    _apply_field_config(start_config)
    _apply_toggle_config(start_config)
    applied_opts = _ui_toggle_snapshot()
    applied_fields = _ui_field_snapshot()
    hub.broadcast("options_applied", {"options": applied_opts, "fields": applied_fields})
    for opt in applied_opts:
        log(f"[INFO] {opt['label']}: {'ON' if opt['value'] else 'OFF'}")

    # configure accelerator now that device selection may have changed
    mask_ops = build_mask_ops(BITMAP_DEVICE)
    accel_note = "Acceleration: CPU bitmap evaluator"; using_cuda=False
    if mask_ops:
        dev = getattr(mask_ops, "device", "cpu"); dev_type = getattr(dev, "type", str(dev))
        if str(dev_type).lower()=="cuda": using_cuda=True; accel_note=f"Acceleration: CUDA GPU ({dev}) via PyTorch"
        elif str(dev).lower()=="numpy": accel_note = "Acceleration: NumPy (CPU)"
        else: accel_note=f"Acceleration: PyTorch device {dev}"
    NestHTTPHandler.cuda_on = using_cuda
    hub.broadcast("hello", {"cuda": using_cuda})

    if not os.path.isdir(FOLDER):
        msg=f"[ERROR] Folder not found: {FOLDER}"
        log(msg)
        control.set_status(phase="done")
        hub.broadcast("progress", {"text": "Folder not found."})
        hub.broadcast("done", {"outputs": []})
        prog.update("Folder not found."); prog.close(); return

    dxf_files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith(".dxf") and f.lower()!="nested.dxf"])
    if not dxf_files:
        msg=f"[WARN] No .dxf files found in: {FOLDER}"
        log(msg)
        control.set_status(phase="done")
        hub.broadcast("progress", {"text": "No .dxf files found."})
        hub.broadcast("done", {"outputs": []})
        prog.update("No .dxf files found."); prog.close(); return

    W_eff=SHEET_W-2*SHEET_MARGIN; H_eff=SHEET_H-2*SHEET_MARGIN
    if W_eff<=0 or H_eff<=0:
        msg=f"[ERROR] SHEET_MARGIN={SHEET_MARGIN} leaves no usable area on a {SHEET_W}×{SHEET_H} sheet."
        log(msg)
        control.set_status(phase="done")
        hub.broadcast("progress", {"text": msg})
        hub.broadcast("done", {"outputs": []})
        prog.update(msg); prog.close(); return

    eff_scale = _eff_scale(PIXELS_PER_UNIT, SPACING)

    control.set_status(phase="reading")
    hub.broadcast("progress", {"text":"Reading DXFs…"})
    prog.update(f"Reading DXFs… 0/{len(dxf_files)}")

    # parse all parts + groups
    all_parts=[]; groups={}; skipped=0
    for idx,fn in enumerate(dxf_files,1):
        prog.update(f"Reading DXFs… {idx}/{len(dxf_files)}  {fn}")
        path=os.path.join(FOLDER,fn)
        loops,segs=parse_entities(path)
        if not loops and segs: loops=join_segments_to_loops(segs,JOIN_TOL)
        fallback_bbox=None
        if not loops and segs and FALLBACK_OPEN_AS_BBOX:
            pts=[pt for a,b in segs for pt in (a,b)]
            if pts:
                minx,miny,maxx,maxy=bbox_of_points(pts)
                if maxx>minx and maxy>miny: fallback_bbox=(minx,miny,maxx,maxy)
        if not loops and fallback_bbox is None:
            log(f"[WARN] {fn}: no closed loops; skipped."); skipped+=1; continue
        p=Part(fn,loops,fallback_bbox)
        if p.outer is None or p.w<=0 or p.h<=0:
            log(f"[WARN] {fn}: zero-sized; skipped."); skipped+=1; continue
        qty=read_qty_for_dxf(FOLDER,fn)
        thk_label=read_thickness_label(FOLDER,fn,THICKNESS_LABEL_UNITS)
        for _ in range(qty):
            all_parts.append(p); groups.setdefault(thk_label,[]).append(p)

    if not all_parts:
        log("[WARN] Nothing to nest.")
        hub.broadcast("progress", {"text":"Nothing to nest."})
        control.set_status(phase="done")
        hub.broadcast("done", {"outputs": []})
        prog.update("Nothing to nest."); prog.close(); return

    outputs=[]
    hub.broadcast("hello", {"cuda": using_cuda})

    def do_one_batch(parts: List[Part], group_label: str):
        total=len(parts)
        hub.broadcast("start", {"sheet_w":W_eff,"sheet_h":H_eff,"margin":SHEET_MARGIN,"total_parts":total,"group":group_label})
        hub.broadcast("sheet_opened", {"sheet_index": 0})
        control.set_status(phase="nest", group=group_label, total=total, placed=0)
        if NEST_MODE.lower()=="bitmap":
            try:
                if SHUFFLE_TRIES>1:
                    placements, sheets = pack_bitmap_multi(parts, W_eff, H_eff, SPACING, eff_scale,
                                                           SHUFFLE_TRIES, SHUFFLE_SEED,
                                                           progress=progress_cb, mask_ops=mask_ops,
                                                           control=control, event_sink=event_sink)
                else:
                    res = pack_bitmap_core(parts, W_eff, H_eff, SPACING, eff_scale,
                                           progress=progress_cb, progress_total=len(parts),
                                           mask_ops=mask_ops, control=control, event_sink=event_sink)
                    placements, sheets = res[0], res[1]
            except NestAbortPartial as nb:
                placements, sheets = nb.placements, nb.sheets
                if SPLIT_SHEETS:
                    base_no_ext=os.path.join(FOLDER, f"{group_label}-nested")
                    write_split_sheets(base_no_ext, placements, sheets, W_eff, H_eff, SHEET_MARGIN, merge_lines=MERGE_LINES)
                    outputs.append((base_no_ext+"-s*", sheets))
                else:
                    out=os.path.join(FOLDER, f"{group_label}-nested.dxf")
                    write_r12_dxf(out, sheets, W_eff, H_eff, placements, SHEET_MARGIN, merge_lines=MERGE_LINES)
                    outputs.append((out, sheets))
                hub.broadcast("stopped", {"outputs": outputs})
                control.set_status(phase="stopped")
                return False  # stopped
        else:
            try:
                placements, sheets = pack_shelves(parts, W_eff, H_eff, SPACING, control=control, event_sink=event_sink)
            except NestAbortPartial as nb:
                placements, sheets = nb.placements, nb.sheets
                if SPLIT_SHEETS:
                    base_no_ext=os.path.join(FOLDER, f"{group_label}-nested")
                    write_split_sheets(base_no_ext, placements, sheets, W_eff, H_eff, SHEET_MARGIN, merge_lines=MERGE_LINES)
                    outputs.append((base_no_ext+"-s*", sheets))
                else:
                    out=os.path.join(FOLDER, f"{group_label}-nested.dxf")
                    write_r12_dxf(out, sheets, W_eff, H_eff, placements, SHEET_MARGIN, merge_lines=MERGE_LINES)
                    outputs.append((out, sheets))
                hub.broadcast("stopped", {"outputs": outputs})
                control.set_status(phase="stopped")
                return False

        if sheets<=0:
            log(f"[WARN] Parts @ {group_label} exist, but none fit.")
            return True

        if SPLIT_SHEETS:
            base_no_ext=os.path.join(FOLDER, f"{group_label}-nested")
            write_split_sheets(base_no_ext, placements, sheets, W_eff, H_eff, SHEET_MARGIN, merge_lines=MERGE_LINES)
            outputs.append((base_no_ext+"-s*", sheets))
        else:
            out=os.path.join(FOLDER, f"{group_label}-nested.dxf")
            write_r12_dxf(out, sheets, W_eff, H_eff, placements, SHEET_MARGIN, merge_lines=MERGE_LINES)
            outputs.append((out, sheets))
        return True

    if GROUP_BY_THICKNESS:
        for thk_label, parts in sorted(groups.items(), key=lambda kv: kv[0]):
            hub.broadcast("group", {"group": thk_label, "total_parts": len(parts)})
            ok = do_one_batch(parts, thk_label)
            if not ok: break
    else:
        ok = do_one_batch(all_parts, "all")

    # Report
    report_path = os.path.join(FOLDER, "nest_report.txt")
    _report_lines.insert(0,"=== Nesting Report ===")
    for out,sheets in outputs: _report_lines.append(f"Saved: {out}  | Sheets: {sheets}")
    _report_lines.append(f"Mode: {NEST_MODE}")
    _report_lines.append(f"Margin: {SHEET_MARGIN}")
    _report_lines.append(f"Spacing: {SPACING}")
    _report_lines.append(f"Safety halo: {SAFETY_GAP}")
    _report_lines.append(f"Spacing tolerances: abs={SPACING_ABS_TOL}, rel={SPACING_REL_TOL}")
    _report_lines.append(f"Minimum spacing pixels: {MIN_SPACING_PIXELS}")
    _report_lines.append(f"Resolution (eff): {eff_scale} px/unit")
    _report_lines.append(f"Shuffle tries: {SHUFFLE_TRIES}{'' if SHUFFLE_SEED is None else f' (seed {SHUFFLE_SEED})'}")
    _report_lines.append(f"Rect-align mode: {RECT_ALIGN_MODE}")
    _report_lines.append(f"Allow mirror: {ALLOW_MIRROR}")
    _report_lines.append(f"Allow nest in holes: {ALLOW_NEST_IN_HOLES}")
    _report_lines.append(f"Thickness label units: {THICKNESS_LABEL_UNITS}")
    _report_lines.append(f"Split sheets: {SPLIT_SHEETS}")
    _report_lines.append(f"Merge common lines: {MERGE_LINES}")
    _report_lines.append(f"INSUNITS header: {'inches' if INSUNITS==1 else 'mm'}")
    _report_lines.append(accel_note)
    try:
        with open(report_path,"w",encoding="utf-8") as rf: rf.write("\n".join(_report_lines))
    except Exception as e:
        print(f"[WARN] Could not write report: {e}")

    control.set_status(phase="done")
    hub.broadcast("done", {"outputs": outputs})
    prog.update("Done. You can close this window."); prog.close()

# ---------- crash log ----------
def _write_crash(folder: str, ex: BaseException):
    try:
        path=os.path.join(folder if os.path.isdir(folder) else os.getcwd(), "nest_error.txt")
        with open(path,"a",encoding="utf-8") as f:
            f.write(f"=== Crash {datetime.datetime.now().isoformat()} ===\n")
            traceback.print_exc(file=f); f.write("\n")
        print(f"[ERROR] A crash log was written to: {path}")
    except: pass

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DXF Nesting with Live HTML viewer (SSE).")
    parser.add_argument("--folder", default=FOLDER, help="Folder with DXFs and optional qty TXTs.")
    parser.add_argument("--sheet", nargs=2, metavar=("WIDTH","HEIGHT"), type=float, default=(SHEET_W, SHEET_H))
    parser.add_argument("--margin", type=float, default=SHEET_MARGIN)
    parser.add_argument("--spacing", type=float, default=SPACING)
    parser.add_argument("--safety-gap", type=float, default=SAFETY_GAP,
                        help="Extra halo around parts in drawing units for alias protection.")
    parser.add_argument("--spacing-abs-tol", type=float, default=SPACING_ABS_TOL,
                        help="Absolute tolerance allowed when quantizing spacing to pixels.")
    parser.add_argument("--spacing-rel-tol", type=float, default=SPACING_REL_TOL,
                        help="Relative tolerance (fraction of spacing) for pixel quantization.")
    parser.add_argument("--min-spacing-px", type=int, default=MIN_SPACING_PIXELS,
                        help="Minimum pixels dedicated to spacing halo when rasterizing.")
    parser.add_argument("--nest-mode", choices=["bitmap","shelf"], default=NEST_MODE)
    parser.add_argument("--pixels-per-unit", type=int, default=PIXELS_PER_UNIT)
    parser.add_argument("--tries", type=int, default=SHUFFLE_TRIES)
    parser.add_argument("--seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument("--workers", type=int, default=BITMAP_EVAL_WORKERS)
    parser.add_argument("--device", default=BITMAP_DEVICE, help="PyTorch device (e.g., 'cuda','cuda:0','cpu').")
    parser.add_argument("--allow-mirror", dest="allow_mirror", action="store_true", default=ALLOW_MIRROR)
    parser.add_argument("--no-mirror", dest="allow_mirror", action="store_false")
    parser.add_argument("--allow-hole-nesting", dest="allow_holes", action="store_true", default=ALLOW_NEST_IN_HOLES)
    parser.add_argument("--forbid-hole-nesting", dest="allow_holes", action="store_false")
    parser.add_argument("--rect-align", choices=["off","prefer","force"], default=RECT_ALIGN_MODE)
    parser.add_argument("--group-by-thickness", dest="group_by_thickness", action="store_true", default=GROUP_BY_THICKNESS)
    parser.add_argument("--no-group-by-thickness", dest="group_by_thickness", action="store_false")
    parser.add_argument("--thickness-label-units", choices=["auto","in","mm"], default=THICKNESS_LABEL_UNITS)
    parser.add_argument("--split-sheets", dest="split_sheets", action="store_true", default=SPLIT_SHEETS)
    parser.add_argument("--no-split-sheets", dest="split_sheets", action="store_false")
    parser.add_argument("--merge-lines", dest="merge_lines", action="store_true", default=MERGE_LINES)
    parser.add_argument("--no-merge-lines", dest="merge_lines", action="store_false")
    parser.add_argument("--rotation-step-deg", type=float, default=ROTATION_STEP_DEG)
    parser.add_argument("--insunits", choices=["in","mm"], default=("in" if INSUNITS==1 else "mm"))
    parser.add_argument("--port", type=int, default=HTTP_PORT, help="HTTP port for the viewer (0=auto).")
    parser.add_argument("--host", default=HTTP_HOST, help="Host/interface for the viewer server (default 127.0.0.1).")
    parser.add_argument("--pause-on-exit", dest="pause_on_exit", action="store_true", default=False)
    args = parser.parse_args()

    # apply args
    FOLDER = os.path.abspath(args.folder)
    SHEET_W, SHEET_H = map(float, args.sheet)
    SHEET_MARGIN = float(args.margin); SPACING=float(args.spacing)
    SAFETY_GAP = max(0.0, float(args.safety_gap))
    SPACING_ABS_TOL = max(1e-9, float(args.spacing_abs_tol))
    SPACING_REL_TOL = max(0.0, float(args.spacing_rel_tol))
    MIN_SPACING_PIXELS = max(1, int(args.min_spacing_px))
    NEST_MODE = args.nest_mode
    PIXELS_PER_UNIT = max(1, int(args.pixels_per_unit))
    SHUFFLE_TRIES = max(1, int(args.tries)); SHUFFLE_SEED=args.seed
    BITMAP_EVAL_WORKERS=args.workers; BITMAP_DEVICE=args.device
    ALLOW_MIRROR=args.allow_mirror; ALLOW_NEST_IN_HOLES=args.allow_holes
    RECT_ALIGN_MODE=args.rect_align; GROUP_BY_THICKNESS=args.group_by_thickness
    THICKNESS_LABEL_UNITS=args.thickness_label_units
    SPLIT_SHEETS=args.split_sheets; MERGE_LINES=args.merge_lines
    ROTATION_STEP_DEG=max(0.0, float(args.rotation_step_deg))
    INSUNITS = 1 if args.insunits=="in" else 4
    HTTP_PORT = int(args.port)
    HTTP_HOST = args.host

    try:
        main_live()
    except Exception as ex:
        _write_crash(FOLDER, ex)
        traceback.print_exc()
    finally:
        if args.pause_on_exit and IS_WINDOWS:
            try: input("\nPress Enter to exit…")
            except: pass
