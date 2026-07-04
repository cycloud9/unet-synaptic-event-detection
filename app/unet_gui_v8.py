# -*- coding: utf-8 -*-
"""
EPSC/IPSC U-Net GUI (v10_Ultimate_Kinetics)
【全面重构】：
1. 废弃曲线拟合，采用 SimplyFire 的滑动均值交叉法寻找 Onset。
2. 引入“自然基线 + 波谷截断 + 3*Tau投射”的三墙合一防拖尾虚高机制。
3. 动态 EPSC/IPSC 窗口参数，完美兼容快速和极慢速动力学。
4. 全面引入底部数据表 (Treeview) 与波形/列表的双向联动。
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import traceback
import csv
import struct  

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Use your own inference utilities
import unet_ola_fp_fn_en as ens  
import unet_ola_fp_fn_ipsc as ens_ipsc  
import abf2csv

# ==== Configuration ====
MODEL_PATH_EPSC = None
MODEL_PATH_IPSC = str(Path(__file__).with_name("best_ipsc_slowkinetics.keras"))

DEFAULT_EVENT_THR = 0.3
ZOOM_WINDOW_SEC_EPSC = 0.04  
ZOOM_WINDOW_SEC_IPSC = 0.10  

DEFAULT_MAIN_WINLEN_EPSC = 5.0
DEFAULT_MAIN_WINLEN_IPSC = 10.0

POTENTIAL_GAP_MS = 0.3         
POTENTIAL_EXCLUDE_MS = 1.0     
POTENTIAL_MIN_SEP_MS = 0.8     

# ==============================================================================
# 核心动力学提取模块 (动态双轨参数 + SimplyFire交叉寻点 + 三墙截断)
# ==============================================================================

def compute_event_kinetics_absolute(trace_data, peak_idx, fs, direction=-1, next_peak_idx=None, is_ipsc=False):
    """
    1. 前瞻滑动均值交叉法 (Moving Average Crossing) 寻找严谨物理 Onset。
    2. 局部动态基线锁定。
    3. “三墙合一” 强制截断 (自然回归、波谷截断、3*Tau 投射) 防治面积虚高。
    """
    dt_ms = 1000.0 / fs
    
    # --- EPSC / IPSC 双轨动态参数配置 ---
    if is_ipsc:
        smooth_win_ms = 4.0      # IPSC: 强力平滑过滤通道开放噪声
        peak_search_win_ms = 6.0 # 大范围找峰
        ma_win_ms = 8.0          # 宏观前瞻找起点
        base_win_ms = 4.0        # 稳定的长基线
    else:
        smooth_win_ms = 1.0      # EPSC: 轻度平滑
        peak_search_win_ms = 2.0 # 精准找峰
        ma_win_ms = 2.0          # 短前瞻
        base_win_ms = 1.5        # 短基线
        
    smooth_win_pts = max(3, int(smooth_win_ms * fs / 1000.0))
    ma_win_pts = max(5, int(ma_win_ms * fs / 1000.0))
    base_win_pts = int(base_win_ms * fs / 1000.0)

    # 扩大提取窗口，确保长尾 IPSC 和滑动平均有足够数据
    pre_search_pts = int((ma_win_ms + base_win_ms + 10.0) * fs / 1000.0) 
    post_search_pts = int(150.0 * fs / 1000.0) if is_ipsc else int(100.0 * fs / 1000.0)
    
    start_idx = max(0, peak_idx - pre_search_pts)
    end_idx = min(len(trace_data), peak_idx + post_search_pts)
    
    # 统一翻转为正向 (向上) 处理
    raw_local = trace_data[start_idx:end_idx] * direction
    local_peak_initial = peak_idx - start_idx
    
    if len(raw_local) < 10: return None
        
    # 1. 动态去毛刺平滑
    kernel = np.ones(smooth_win_pts) / smooth_win_pts
    smooth_local = np.convolve(raw_local, kernel, mode='same')
    smooth_local[:smooth_win_pts] = raw_local[:smooth_win_pts]
    smooth_local[-smooth_win_pts:] = raw_local[-smooth_win_pts:]
    
    # 2. 锁定平滑曲线上的真实物理重心 (Peak)
    search_win = int(peak_search_win_ms * fs / 1000.0)
    p_start = max(0, local_peak_initial - search_win)
    p_end = min(len(smooth_local), local_peak_initial + search_win)
    if p_end <= p_start: return None
    true_peak_rel = p_start + np.argmax(smooth_local[p_start:p_end])
    
    # =========================================================
    # 3. SimplyFire 前瞻滑动均值交叉法 (Onset 寻找)
    # =========================================================
    onset_rel = true_peak_rel
    for i in range(true_peak_rel - 1, ma_win_pts, -1):
        ma_val = np.mean(smooth_local[i - ma_win_pts : i])
        if smooth_local[i] <= ma_val: # 跌穿前方滑动均值，回到基线
            onset_rel = i
            break
            
    # =========================================================
    # 4. 局部动态基线 (Local Dynamic Baseline)
    # =========================================================
    b_start = max(0, onset_rel - base_win_pts)
    if onset_rel > b_start:
        baseline_val = np.mean(smooth_local[b_start:onset_rel])
    else:
        baseline_val = smooth_local[0]
        
    amplitude = smooth_local[true_peak_rel] - baseline_val
    if amplitude <= 2.0: return None 
        
    level_10 = baseline_val + 0.10 * amplitude
    level_50 = baseline_val + 0.50 * amplitude
    level_90 = baseline_val + 0.90 * amplitude
    level_37 = baseline_val + 0.3678 * amplitude 
    
    # =========================================================
    # 5. 重叠波谷截断防爆墙 (Drop-line Wall)
    # =========================================================
    wall_rel = len(smooth_local)
    if next_peak_idx is not None and next_peak_idx > peak_idx:
        next_rel = next_peak_idx - start_idx
        if next_rel > true_peak_rel:
            search_valley_end = min(next_rel + int(2.0*fs/1000.0), len(smooth_local))
            wall_rel = true_peak_rel + np.argmin(smooth_local[true_peak_rel:search_valley_end])
            
    wall_rel = min(wall_rel, len(smooth_local))
    
    # =========================================================
    # 6. Decay 37% 与 3*Tau 投射截断墙
    # =========================================================
    valid_decay = smooth_local[true_peak_rel:wall_rel]
    
    t_37_idx_arr = np.where(valid_decay <= level_37)[0]
    if len(t_37_idx_arr) > 0:
        decay_37_rel = true_peak_rel + t_37_idx_arr[0]
        decay_tau_ms = (decay_37_rel - true_peak_rel) * dt_ms
        
        # O-Score 演化法则：根据 1Tau 推算 3Tau (95% 衰减) 作为强制截断边界
        tau_pts = decay_37_rel - true_peak_rel
        tau_wall_rel = true_peak_rel + int(3.0 * tau_pts) 
        
        # 三墙合一：重叠波谷 和 3*Tau 投射墙，谁更近用谁！
        wall_rel = min(wall_rel, tau_wall_rel)
    else:
        decay_37_rel = None
        decay_tau_ms = np.nan 
        
    # Rise 10-90
    rise_phase_full = smooth_local[onset_rel:true_peak_rel+1]
    r10_arr = np.where(rise_phase_full <= level_10)[0]
    r90_arr = np.where(rise_phase_full <= level_90)[0]
    r50_arr = np.where(rise_phase_full <= level_50)[0]
    
    t_10_rel = onset_rel + r10_arr[-1] if len(r10_arr) > 0 else onset_rel
    t_90_rel = onset_rel + r90_arr[-1] if len(r90_arr) > 0 else true_peak_rel
    rise_50_rel = onset_rel + r50_arr[-1] if len(r50_arr) > 0 else onset_rel
    rise_ms = (t_90_rel - t_10_rel) * dt_ms

    t_50_idx_arr = np.where(valid_decay <= level_50)[0]
    if len(t_50_idx_arr) > 0:
        decay_50_rel = true_peak_rel + t_50_idx_arr[0]
        half_width_ms = (decay_50_rel - rise_50_rel) * dt_ms
    else:
        half_width_ms = np.nan
        
    # =========================================================
    # 7. 真实面积积分 (Area) - 终结虚高
    # =========================================================
    # wall_rel 已被波谷和 3*Tau 保护。在墙内寻找自然回归点。
    valid_decay_for_area = smooth_local[true_peak_rel:wall_rel]
    baseline_crossings = np.where(valid_decay_for_area <= baseline_val)[0]
    
    if len(baseline_crossings) > 0:
        end_rel = true_peak_rel + baseline_crossings[0] 
    else:
        end_rel = wall_rel 
        
    # 直接对扣除基线后的原始波形进行物理积分
    signal_area_array = raw_local[onset_rel:end_rel] - baseline_val
    
    # 删掉 [ < 0] = 0 这一句！让向下的毛刺自然去抵消向上的毛刺！
    area_fc = np.trapz(signal_area_array) * dt_ms
    
    abs_onset = start_idx + onset_rel
    abs_peak = start_idx + true_peak_rel
    abs_decay37 = start_idx + decay_37_rel if decay_37_rel is not None else None
    abs_end = start_idx + end_rel
    
    return {
        'Amplitude_pA': amplitude,
        'Rise_10_90_ms': rise_ms,
        'Decay_Tau_ms': decay_tau_ms,
        'Half_Width_ms': half_width_ms,
        'Area_fC': area_fc,
        'Baseline_pA': baseline_val * direction,
        
        'abs_onset': abs_onset,
        'abs_peak': abs_peak,
        'abs_decay37': abs_decay37,
        'abs_end': abs_end,
        'plot_baseline': baseline_val * direction,
        'plot_level_37': level_37 * direction
    }

# ==============================================================================
# GUI Class Definition
# ==============================================================================

class EPSCGUI:
    def __init__(self, master):
        self.master = master
        self.polarity_var = tk.StringVar(value="epsc")
        self.master.title("U-Net GUI (EPSC/IPSC) - Visual Kinetics")

        self.model_path_eps = MODEL_PATH_EPSC
        self.model_path_ips = MODEL_PATH_IPSC
        self.model_eps = None
        self.model_ips = None

        self.csv_path = None
        self.events_export_path = None  

        self.trace_raw = None  
        self.trace = None      
        self.prob = None
        self.rate = None
        self.fs = None         
        self.fs_orig = None    
        self.T_sec = None
        self.time_axis = None
        self.event_thr = DEFAULT_EVENT_THR

        self.events_red_idx = np.array([], dtype=int)     
        self.events_blue_idx = np.array([], dtype=int)    

        self.view_mode_var = tk.StringVar(value="events")
        self.display_items = []
        self.cached_kinetics = {} # 缓存红色事件的动力学参数

        self.fig = None
        self.canvas = None
        self.ax_trace = None
        self.ax_prob = None
        self.trace_line = None
        self.prob_line = None
        self.threshold_line = None
        self.event_vlines = []  

        self.sel_line_trace = None
        self.sel_line_prob = None
        self._mpl_cid = None  

        self.zoom_fig = None
        self.zoom_canvas = None
        self.zoom_ax_trace = None
        self.zoom_ax_prob = None

        self.add_event_mode = False
        self._disable_tree_sync = False # 防止双向联动死循环

        # --- UI Layout Starts ---
        self.path_label = tk.Label(master, text="No CSV or ABF file selected", width=60, anchor="w")
        self.path_label.pack(padx=10, pady=5)

        btn_frame = tk.Frame(master)
        btn_frame.pack(padx=10, pady=5)

        self.btn_choose = tk.Button(btn_frame, text="Select trace CSV/ABF", command=self.choose_file)
        self.btn_choose.pack(side=tk.LEFT, padx=5)

        tk.Label(btn_frame, text="Polarity:").pack(side=tk.LEFT, padx=(10, 2))
        self.polarity_menu = tk.OptionMenu(btn_frame, self.polarity_var, "epsc", "ipsc", command=lambda _: self.on_polarity_changed())
        self.polarity_menu.config(width=6)
        self.polarity_menu.pack(side=tk.LEFT, padx=2)

        self.btn_run = tk.Button(btn_frame, text="Run U-Net Inference", command=self.run_unet)
        self.btn_run.pack(side=tk.LEFT, padx=5)

        self.btn_export = tk.Button(btn_frame, text="Export Events", command=self.export_events)
        self.btn_export.pack(side=tk.LEFT, padx=5)

        self.btn_choose_model = tk.Button(btn_frame, text="Select Model", command=self.choose_model)
        self.btn_choose_model.pack(side=tk.LEFT, padx=5)

        self.btn_add_event = tk.Button(btn_frame, text="Add Event", command=self.enable_add_event)
        self.btn_add_event.pack(side=tk.LEFT, padx=5)

        self.log_text = tk.Text(master, height=6, width=90)
        self.log_text.pack(padx=10, pady=5, fill=tk.X)

        self.center_frame = tk.Frame(master)
        self.center_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 0))

        # Left side: Plots
        self.plot_frame = tk.Frame(self.center_frame)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side: Events list & Zoom
        self.side_frame = tk.Frame(self.center_frame, width=320)
        self.side_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.side_frame.pack_propagate(False)

        tk.Label(self.side_frame, text="Events List", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=(0, 3))

        mode_frame = tk.Frame(self.side_frame)
        mode_frame.pack(fill=tk.X, padx=5, pady=(0, 3))

        tk.Label(mode_frame, text="Show:").pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Events", variable=self.view_mode_var, value="events", command=self.on_view_mode_changed).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(mode_frame, text="Potential", variable=self.view_mode_var, value="potential", command=self.on_view_mode_changed).pack(side=tk.LEFT, padx=2)
        tk.Radiobutton(mode_frame, text="All", variable=self.view_mode_var, value="all", command=self.on_view_mode_changed).pack(side=tk.LEFT, padx=2)

        nav_frame = tk.Frame(self.side_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=(0, 3))
        self.btn_prev = tk.Button(nav_frame, text="↑ Prev", width=9, command=self.go_prev_event)
        self.btn_prev.pack(side=tk.LEFT, padx=2)
        self.btn_next = tk.Button(nav_frame, text="↓ Next", width=9, command=self.go_next_event)
        self.btn_next.pack(side=tk.LEFT, padx=2)
        self.btn_promote = tk.Button(nav_frame, text="Promote", width=10, command=self.promote_selected_to_red)
        self.btn_promote.pack(side=tk.RIGHT, padx=2)

        list_frame = tk.Frame(self.side_frame)
        list_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=3)

        self.events_listbox = tk.Listbox(list_frame, height=8, width=38)
        self.events_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.events_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.events_listbox.config(yscrollcommand=scrollbar.set)

        self.events_listbox.bind("<<ListboxSelect>>", self.on_event_selected)
        self.events_listbox.bind("<Up>", self.on_key_prev_next)
        self.events_listbox.bind("<Down>", self.on_key_prev_next)
        self.events_listbox.bind("<Return>", self.on_enter_key)
        self.events_listbox.bind("<Double-Button-1>", self.on_double_click_list)
        self.events_listbox.bind("<Button-3>", self.on_list_right_click)
        self.events_listbox.bind("<Delete>", self.on_delete_key)
        self.events_listbox.bind("<BackSpace>", self.on_delete_key) # 兼容 Mac 键盘

        tk.Label(self.side_frame, text="Single Event Zoom", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=(8, 0))

        self.zoom_plot_frame = tk.Frame(self.side_frame)
        self.zoom_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=3)

        zoom_frame = tk.Frame(master)
        zoom_frame.pack(fill=tk.X, padx=10, pady=0)

        self.start_time_var = tk.DoubleVar(value=0.0)
        self.winlen_var = tk.DoubleVar(value=DEFAULT_MAIN_WINLEN_EPSC)  

        tk.Label(zoom_frame, text="Window length (s):").grid(row=0, column=0, sticky="w")
        self.winlen_spin = tk.Spinbox(zoom_frame, from_=0.5, to=20.0, increment=0.5, width=6, textvariable=self.winlen_var, command=self.on_window_change)
        self.winlen_spin.grid(row=0, column=1, sticky="w", padx=5)

        tk.Label(zoom_frame, text="Start time (s):").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.start_scale = tk.Scale(zoom_frame, from_=0.0, to=0.0, orient=tk.HORIZONTAL, resolution=0.1, length=400, variable=self.start_time_var, command=lambda v: self.on_window_change())
        self.start_scale.grid(row=0, column=3, sticky="we", padx=5)
        zoom_frame.columnconfigure(3, weight=1)
        
        # --- Bottom UI: Treeview Data Table ---
        table_frame = tk.Frame(master)
        table_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        columns = ("id", "time", "amp", "rise", "decay", "hw", "area", "base")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
        self.tree.heading("id", text="Event ID")
        self.tree.column("id", width=60, anchor="center")
        self.tree.heading("time", text="Time (s)")
        self.tree.column("time", width=100, anchor="center")
        self.tree.heading("amp", text="Amplitude (pA)")
        self.tree.column("amp", width=120, anchor="center")
        self.tree.heading("rise", text="Rise 10-90 (ms)")
        self.tree.column("rise", width=120, anchor="center")
        self.tree.heading("decay", text="Decay Tau (ms)")
        self.tree.column("decay", width=120, anchor="center")
        self.tree.heading("hw", text="Half-Width (ms)")
        self.tree.column("hw", width=120, anchor="center")
        self.tree.heading("area", text="Area (fC)")
        self.tree.column("area", width=100, anchor="center")
        self.tree.heading("base", text="Baseline (pA)")
        self.tree.column("base", width=100, anchor="center")
        
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind("<ButtonRelease-1>", self.on_treeview_click)

        self.log("Ready: choose EPSC/IPSC polarity, then select a CSV/ABF and run inference.")

    # ===== Utils & Models =====
    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        print(msg)

    @staticmethod
    def _insert_sorted_unique(arr: np.ndarray, value: int) -> np.ndarray:
        if arr is None or len(arr) == 0: return np.array([int(value)], dtype=int)
        value = int(value)
        pos = int(np.searchsorted(arr, value))
        if pos < len(arr) and int(arr[pos]) == value: return arr
        return np.insert(arr, pos, value)

    @staticmethod
    def _remove_value(arr: np.ndarray, value: int) -> np.ndarray:
        if arr is None or len(arr) == 0: return arr
        value = int(value)
        pos = int(np.searchsorted(arr, value))
        if pos < len(arr) and int(arr[pos]) == value: return np.delete(arr, pos)
        mask = arr != value
        return arr[mask]

    def choose_file(self):
        path = filedialog.askopenfilename(
            title=f"Select {self.polarity_var.get().upper()} trace CSV or ABF",
            filetypes=[("CSV files", "*.csv"), ("ABF files", "*.abf"), ("All files", "*.*")],
        )
        if path:
            self.csv_path = Path(path)
            self.path_label.config(text=str(self.csv_path))
            self.log(f"Selected file: {self.csv_path}")
            if self.csv_path.suffix.lower() == ".abf":
                self.log("ABF file detected, converting to CSV...")
                try:
                    trace_csv = abf2csv.convert_abf_to_trace_csv(self.csv_path)
                    self.csv_path = trace_csv
                    self.path_label.config(text=str(self.csv_path))
                    self.log(f"ABF file converted to CSV: {self.csv_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to convert ABF to CSV: {e}")

    def choose_model(self):
        path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Keras Model files", "*.keras"), ("All files", "*.*")])
        if path:
            pol = self.polarity_var.get()
            if pol == "ipsc":
                global MODEL_PATH_IPSC
                MODEL_PATH_IPSC = path
                self.model_path_ips = MODEL_PATH_IPSC
                self.model_ips = None  
                self.log(f"Selected IPSC model file: {MODEL_PATH_IPSC}")
            else:
                global MODEL_PATH_EPSC
                MODEL_PATH_EPSC = path
                self.model_path_eps = MODEL_PATH_EPSC
                self.model_eps = None  
                self.log(f"Selected EPSC model file: {MODEL_PATH_EPSC}")

    def on_polarity_changed(self):
        pol = self.polarity_var.get()
        self.master.title(f"U-Net GUI ({pol.upper()})")
        self.btn_choose.config(text=f"Select {pol.upper()} trace CSV or ABF")
        try:
            self.winlen_var.set(DEFAULT_MAIN_WINLEN_IPSC if pol == "ipsc" else DEFAULT_MAIN_WINLEN_EPSC)
            self.start_time_var.set(0.0)
            if self.T_sec is not None: self.update_plot_window()
        except Exception: pass
        self.clear_selected_event_highlight()
        if self.trace is not None:
            self.ax_trace.set_title(f"{pol.upper()} trace with detected events")
            self.recompute_all_kinetics()
            self.canvas.draw_idle()

    def get_zoom_window_sec(self):
        return ZOOM_WINDOW_SEC_IPSC if self.polarity_var.get() == "ipsc" else ZOOM_WINDOW_SEC_EPSC

    def load_model_if_needed(self):
        pol = self.polarity_var.get()
        if pol == "ipsc":
            if self.model_ips is not None: return
            mp = self.model_path_ips or MODEL_PATH_IPSC
            self.log("Loading IPSC U-Net model...")
            try:
                ens_ipsc.ENS2_TEMPERATURE = ens_ipsc._resolve_temperature(getattr(ens_ipsc, "METRICS_JSON", None), default=1.0)
            except Exception:
                ens_ipsc.ENS2_TEMPERATURE = 1.0
            self.model_ips = ens_ipsc.load_ens2_model_robust(mp)
            self.log("IPSC model loaded successfully.")
        else:
            if self.model_eps is not None: return
            mp = self.model_path_eps or MODEL_PATH_EPSC
            self.log("Loading EPSC U-Net model...")
            try:
                ens.ENS2_TEMPERATURE = ens._resolve_temperature(getattr(ens, "METRICS_JSON", None), default=1.0)
            except Exception:
                ens.ENS2_TEMPERATURE = 1.0
            self.model_eps = ens.load_ens2_model_robust(mp)
            self.log("EPSC model loaded successfully.")

    def compute_potential_events(self, prob: np.ndarray, red_events: np.ndarray) -> np.ndarray:
        if prob is None or len(prob) == 0 or self.fs is None: return np.array([], dtype=int)
        fs = float(self.fs)
        gap_samp = max(0, int(round(POTENTIAL_GAP_MS * 1e-3 * fs)))
        excl_samp = max(0, int(round(POTENTIAL_EXCLUDE_MS * 1e-3 * fs)))
        min_sep_samp = max(1, int(round(POTENTIAL_MIN_SEP_MS * 1e-3 * fs)))

        mask = prob > float(self.event_thr)
        if not np.any(mask): return np.array([], dtype=int)

        if gap_samp > 0:
            mask_f = mask.copy()
            n = len(mask_f)
            i = 0
            while i < n:
                if not mask_f[i]:
                    j = i
                    while j < n and not mask_f[j]: j += 1
                    if (j - i) <= gap_samp and (i - 1) >= 0 and mask_f[i - 1] and j < n and mask_f[j]:
                        mask_f[i:j] = True
                    i = j
                else: i += 1
            mask = mask_f

        candidates = []
        i, n = 0, len(mask)
        while i < n:
            if mask[i]:
                j = i
                while j < n and mask[j]: j += 1
                seg = prob[i:j]
                if len(seg) > 0: candidates.append(i + int(np.argmax(seg)))
                i = j
            else: i += 1

        if not candidates: return np.array([], dtype=int)
        candidates = np.array(sorted(set(candidates)), dtype=int)

        if red_events is not None and len(red_events) > 0 and excl_samp > 0:
            red_sorted = np.sort(red_events)
            keep = []
            for c in candidates:
                pos = int(np.searchsorted(red_sorted, c))
                near = (pos < len(red_sorted) and abs(red_sorted[pos] - c) <= excl_samp) or \
                       (pos > 0 and abs(red_sorted[pos - 1] - c) <= excl_samp)
                if not near: keep.append(c)
            candidates = np.array(keep, dtype=int)

        if len(candidates) == 0: return candidates

        kept = [candidates[0]]
        for c in candidates[1:]:
            if abs(c - kept[-1]) >= min_sep_samp:
                kept.append(c)
            elif float(prob[c]) > float(prob[kept[-1]]):
                kept[-1] = c
        return np.array(kept, dtype=int)

    def run_unet(self):
        try:
            if self.csv_path is None:
                messagebox.showwarning("Warning", "Please select a CSV file first.")
                return

            self.load_model_if_needed()
            self.log("Loading CSV and dynamically detecting sampling rate...")
            pol = self.polarity_var.get()
            infer = ens_ipsc if pol == "ipsc" else ens
            model = self.model_ips if pol == "ipsc" else self.model_eps

            trace_raw, fs_orig = infer.load_trace_from_csv(self.csv_path)
            self.fs_orig = float(fs_orig)
            self.trace_raw = trace_raw  
            self.log(f"Detected original Fs: {self.fs_orig} Hz. Retained {len(self.trace_raw)} raw data points.")
            
            trace, fs = infer.resample_to_target_fs(trace_raw, fs_orig, infer.ENS2_TARGET_FS)
            T_sec = len(trace) / fs

            self.log("Running ENS² + blob-based inference...")
            detected_events, prob, rate = infer.blob_based_inference(trace, fs, model, polarity=pol) if pol == "ipsc" else infer.blob_based_inference(trace, fs, model)
            
            self.trace = trace
            self.prob = prob
            self.rate = rate
            self.fs = float(fs)
            self.T_sec = float(T_sec)
            self.time_axis = np.arange(len(trace)) / fs

            self.events_red_idx = np.sort(np.array(detected_events, dtype=int))
            self.events_blue_idx = np.sort(self.compute_potential_events(self.prob, self.events_red_idx))

            self.log(f"Inference done. Red events: {len(self.events_red_idx)}, Blue events: {len(self.events_blue_idx)}")

            self.init_or_update_main_figure()
            
            # 全量计算动力学参数，刷新树状表
            self.recompute_all_kinetics()
            self.populate_events_list()

            self.winlen_var.set(5.0)
            self.start_scale.config(from_=0.0, to=max(0.0, self.T_sec - 5.0))
            self.start_time_var.set(0.0)
            self.update_plot_window()
            self.clear_selected_event_highlight()

        except Exception as e:
            self.log(f"Error occurred:\n{traceback.format_exc()}")
            messagebox.showerror("Error", f"Failed to run inference:\n{e}")

    # ==============================================================================
    # 全量参数缓存与底部表格同步
    # ==============================================================================
    def recompute_all_kinetics(self):
            if self.trace is None or len(self.events_red_idx) == 0:
                self.tree.delete(*self.tree.get_children())
                self.cached_kinetics.clear()
                return
                
            # 备份老缓存
            old_cache = self.cached_kinetics.copy()
            self.cached_kinetics.clear()
            
            # Treeview 的 UI 刷新极快，全部重绘不影响性能，瓶颈在算法
            self.tree.delete(*self.tree.get_children())
            
            direction = -1 if self.polarity_var.get() == "epsc" else 1
            is_ipsc = (self.polarity_var.get() == "ipsc")
            
            for i, idx in enumerate(self.events_red_idx):
                idx = int(idx)
                next_idx = self.events_red_idx[i+1] if i + 1 < len(self.events_red_idx) else None
                
                # 核心优化：如果该峰之前算过，且它后面的邻居没变（意味着截断墙位置绝对不变），直接光速复用！
                if idx in old_cache and old_cache[idx].get('_next_idx') == next_idx:
                    kin = old_cache[idx]
                else:
                    kin = compute_event_kinetics_absolute(self.trace, idx, self.fs, direction=direction, next_peak_idx=next_idx, is_ipsc=is_ipsc)
                    if kin:
                        kin['_next_idx'] = next_idx  # 在字典里悄悄打个标记
                        
                if kin:
                    self.cached_kinetics[idx] = kin
                    t_s = float(self.time_axis[idx])
                    self.tree.insert("", "end", iid=str(idx), values=(
                        f"E{i}", f"{t_s:.4f}", f"{kin['Amplitude_pA']:.2f}", 
                        f"{kin['Rise_10_90_ms']:.2f}", f"{kin['Decay_Tau_ms']:.2f}", 
                        f"{kin['Half_Width_ms']:.2f}", f"{kin['Area_fC']:.2f}", 
                        f"{kin['Baseline_pA']:.2f}"
                    ))

    # ===== Plots & Interactions =====
    def init_or_update_main_figure(self):
        if self.fig is None:
            self.fig = Figure(figsize=(7, 5), dpi=100)
            self.ax_trace = self.fig.add_subplot(211)
            self.ax_prob = self.fig.add_subplot(212, sharex=self.ax_trace)

            self.trace_line, = self.ax_trace.plot(self.time_axis, self.trace, "k-", linewidth=0.5)
            self.prob_line, = self.ax_prob.plot(self.time_axis, self.prob, "b-", linewidth=0.8)
            self.threshold_line = self.ax_prob.axhline(self.event_thr, linestyle="--", color="orange")

            self.ax_trace.set_ylabel("Current (pA)")
            self.ax_prob.set_ylabel("Probability")
            self.ax_prob.set_xlabel("Time (s)")
            self.fig.tight_layout()
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self._mpl_cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click_main_plot)
        else:
            self.trace_line.set_data(self.time_axis, self.trace)
            self.prob_line.set_data(self.time_axis, self.prob)
            self.ax_trace.relim()
            self.ax_trace.autoscale_view()
            self.ax_prob.relim()
            self.ax_prob.autoscale_view()
            self.clear_selected_event_highlight()
            self.canvas.draw()
        
        pol = self.polarity_var.get()
        self.ax_trace.set_title(f"{pol.upper()} trace with detected events")
        self.update_vlines()

    def update_vlines(self):
        for line in self.event_vlines: line.remove()
        self.event_vlines.clear()

        if len(self.events_red_idx) > 0:
            for t in self.time_axis[self.events_red_idx]:
                self.event_vlines.extend([self.ax_trace.axvline(t, color="r", alpha=0.6), self.ax_prob.axvline(t, color="r", alpha=0.6)])

        if len(self.events_blue_idx) > 0:
            for t in self.time_axis[self.events_blue_idx]:
                self.event_vlines.extend([self.ax_trace.axvline(t, color="b", alpha=0.3, linestyle=":"), self.ax_prob.axvline(t, color="b", alpha=0.3, linestyle=":")])

        self.canvas.draw_idle()

    # ==============================================================================
    # 核心可视化重构：三点一线一阴影 (还原 U-Net 锚点)
    # ==============================================================================
    def update_zoom_plot(self, event_idx_in_trace: int, kind: str):
        if self.zoom_fig is None:
            self.zoom_fig = Figure(figsize=(2.8, 3.5), dpi=100)
            self.zoom_ax_trace = self.zoom_fig.add_subplot(211)
            self.zoom_ax_prob = self.zoom_fig.add_subplot(212, sharex=self.zoom_ax_trace)
            self.zoom_fig.tight_layout()
            self.zoom_canvas = FigureCanvasTkAgg(self.zoom_fig, master=self.zoom_plot_frame)
            self.zoom_canvas.draw()
            self.zoom_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.zoom_ax_trace.clear()
        self.zoom_ax_prob.clear()

        if event_idx_in_trace is None:
            self.zoom_ax_trace.text(0.5, 0.5, "No event", ha="center", va="center")
            self.zoom_canvas.draw_idle()
            return

        idx = int(event_idx_in_trace)
        anchor_t = self.time_axis[idx]
        
        w_samp = int(self.get_zoom_window_sec() * self.fs)
        start = max(0, idx - w_samp // 2)
        end = min(len(self.trace), idx + w_samp // 2)

        # 1. 底层原汁原味的黑线和 U-Net Prob map
        self.zoom_ax_trace.plot(self.time_axis[start:end], self.trace[start:end], "k-", linewidth=1.0, zorder=1)
        self.zoom_ax_prob.plot(self.time_axis[start:end], self.prob[start:end], "b-", linewidth=1.0)
        
        # 2. 保留 U-Net 原始锚点指示线 (贯穿两张子图)
        line_color = "r" if kind == "red" else "b"
        self.zoom_ax_trace.axvline(anchor_t, color=line_color, linestyle="--", alpha=0.5, zorder=0)
        self.zoom_ax_prob.axvline(anchor_t, color=line_color, linestyle="--", alpha=0.5, zorder=0)
        
        # 如果是红色事件，叠加三点一阴影物理投影
        if kind == "red" and idx in self.cached_kinetics:
            kin = self.cached_kinetics[idx]
            direction = -1 if self.polarity_var.get() == "epsc" else 1
            
            # 画局部动态基线 (绿色虚线) 与 37% 阈值线 (橙色虚线)
            self.zoom_ax_trace.axhline(kin['plot_baseline'], color='g', linestyle='--', alpha=0.6, zorder=2)
            self.zoom_ax_trace.axhline(kin['plot_level_37'], color='orange', linestyle='--', alpha=0.6, zorder=2)
            
            abs_onset = kin['abs_onset']
            abs_peak = kin['abs_peak']
            abs_decay37 = kin['abs_decay37']
            abs_end = kin['abs_end']
            
            # 三点定位投影
            self.zoom_ax_trace.plot(self.time_axis[abs_onset], self.trace[abs_onset], 'gx', markersize=8, markeredgewidth=2, zorder=4)
            self.zoom_ax_trace.plot(self.time_axis[abs_peak], self.trace[abs_peak], 'rx', markersize=8, markeredgewidth=2, zorder=4)
            if abs_decay37 is not None:
                self.zoom_ax_trace.plot(self.time_axis[abs_decay37], kin['plot_level_37'], 'o', color='hotpink', markersize=6, zorder=4)
                
            # 积分阴影面积
            fill_t = self.time_axis[abs_onset:abs_end+1]
            fill_y = self.trace[abs_onset:abs_end+1]
            base_arr = np.full_like(fill_y, kin['plot_baseline'])
            
            if direction == -1: # EPSC
                self.zoom_ax_trace.fill_between(fill_t, fill_y, base_arr, where=(fill_y <= base_arr), color='red', alpha=0.3, zorder=3)
            else: # IPSC
                self.zoom_ax_trace.fill_between(fill_t, fill_y, base_arr, where=(fill_y >= base_arr), color='red', alpha=0.3, zorder=3)
                
            # 右下角增加高逼格半透明参数展板
            text_str = (
                f"Amp: {kin['Amplitude_pA']:.1f} pA\n"
                f"Rise: {kin['Rise_10_90_ms']:.1f} ms\n"
                f"Decay: {kin['Decay_Tau_ms']:.1f} ms\n"
                f"Area: {kin['Area_fC']:.1f} fC"
            )
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            self.zoom_ax_trace.text(0.95, 0.95, text_str, transform=self.zoom_ax_trace.transAxes, 
                                    fontsize=8, verticalalignment='top', horizontalalignment='right', bbox=props, zorder=5)
            
            self.zoom_ax_trace.set_title(f"Event @ {anchor_t:.4f}s", fontsize=9)
        else:
            self.zoom_ax_trace.set_title(f"Potential Event @ {anchor_t:.4f}s", fontsize=9)
            
        self.zoom_canvas.draw_idle()

    def on_window_change(self, *args):
        if self.trace is not None: self.update_plot_window()

    def update_plot_window(self):
        try:
            start, wlen = self.start_time_var.get(), self.winlen_var.get()
            self.ax_trace.set_xlim(start, start + wlen)
            self.ax_prob.set_xlim(start, start + wlen)
            self.canvas.draw_idle()
        except Exception: pass

    def on_click_main_plot(self, event):
            if event.inaxes not in [self.ax_trace, self.ax_prob] or self.trace is None or event.xdata is None: return
    
            if self.add_event_mode:
                self.perform_add_event(event.xdata)
                return
    
            click_idx = int(event.xdata * self.fs)
            candidates = []
            if len(self.events_red_idx) > 0:
                dist = np.abs(self.events_red_idx - click_idx)
                candidates.append((dist.min(), self.events_red_idx[np.argmin(dist)], "red"))
            if len(self.events_blue_idx) > 0:
                dist = np.abs(self.events_blue_idx - click_idx)
                candidates.append((dist.min(), self.events_blue_idx[np.argmin(dist)], "blue"))
    
            if not candidates: return
            best = min(candidates, key=lambda x: x[0])
            
            # 命中范围 (50ms 内)
            if best[0] <= 0.05 * self.fs:
                # 1. 鼠标左键 (默认): 正常高亮并联动数据表
                if event.button == 1:
                    self.highlight_event(best[1], best[2])
                    
                # 2. 鼠标右键: 找回老版本的删除(降级)功能，并且可以反向晋升！
                elif event.button == 3:
                    # 先让程序选中它（为了触发底层的双向联动）
                    self.highlight_event(best[1], best[2])
                    # 如果点的是红点 -> 删掉它(降级为蓝点)
                    if best[2] == "red":
                        self.demote_selected_to_blue()
                    # 如果点的是蓝点 -> 捡回来(晋升为红点)
                    else:
                        self.promote_selected_to_red()

    def highlight_event(self, idx, kind):
            t = self.time_axis[idx]
            if self.sel_line_trace: self.sel_line_trace.remove()
            if self.sel_line_prob: self.sel_line_prob.remove()
            
            c = "r" if kind == "red" else "b"
            self.sel_line_trace = self.ax_trace.axvline(t, color=c, linewidth=2, alpha=0.8)
            self.sel_line_prob = self.ax_prob.axvline(t, color=c, linewidth=2, alpha=0.8)
            self.canvas.draw_idle()
            self.update_zoom_plot(idx, kind)
    
            # 联动 1: 同步右侧 Events List
            for i, item in enumerate(self.display_items):
                if item["sample_idx"] == idx and item["kind"] == kind:
                    self.events_listbox.selection_clear(0, tk.END)
                    self.events_listbox.selection_set(i)
                    self.events_listbox.activate(i)  # <--- 加上这一行！彻底锁死键盘焦点，拒绝跳跃！
                    self.events_listbox.see(i)
                    break
                    
            # 联动 2: 同步底部 Treeview 数据表
            if kind == "red" and not self._disable_tree_sync:
                self._disable_tree_sync = True
                self.tree.selection_remove(self.tree.selection())
                if self.tree.exists(str(idx)):
                    self.tree.selection_set(str(idx))
                    self.tree.see(str(idx))
                self._disable_tree_sync = False

    def clear_selected_event_highlight(self):
        if self.sel_line_trace: self.sel_line_trace.remove(); self.sel_line_trace = None
        if self.sel_line_prob: self.sel_line_prob.remove(); self.sel_line_prob = None
        if self.zoom_ax_trace: self.zoom_ax_trace.clear(); self.zoom_ax_prob.clear(); self.zoom_canvas.draw_idle()

    # ===== List & Treeview Logic =====
    def on_view_mode_changed(self):
        self.populate_events_list()

    def populate_events_list(self):
            if self.trace is None: return
            self.events_listbox.delete(0, tk.END)
            self.display_items = []
            mode = self.view_mode_var.get()
            
            items = []
            if mode in ["events", "all"]:
                items.extend([{"idx": i, "kind": "red"} for i in self.events_red_idx])
            if mode in ["potential", "all"]:
                items.extend([{"idx": i, "kind": "blue"} for i in self.events_blue_idx])
            
            items.sort(key=lambda x: x["idx"])
    
            # ====== 修改的部分：恢复老版本的详细格式，并保留 V8 的颜色标识 ======
            for i, it in enumerate(items):
                idx = it["idx"]
                kind = it["kind"]
                t_s = float(self.time_axis[idx])
                tag = "R" if kind == "red" else "B"
                
                # 生成形如: "0001 [R]  t=    9.40 ms  (idx=94)" 的标签
                label = f"{i+1:04d} [{tag}]  t={t_s*1000.0:8.2f} ms  (idx={idx})"
                
                self.events_listbox.insert(tk.END, label)
                # 保留 V8 版本中对字体颜色的控制 (红/蓝)
                self.events_listbox.itemconfig(tk.END, {'fg': kind}) 
                self.display_items.append({"kind": kind, "sample_idx": idx})

    def on_event_selected(self, event):
        sel = self.events_listbox.curselection()
        if sel: self.highlight_event(self.display_items[sel[0]]["sample_idx"], self.display_items[sel[0]]["kind"])

    def on_treeview_click(self, event):
        if self._disable_tree_sync: return
        sel = self.tree.selection()
        if sel:
            idx = int(sel[0])
            self._disable_tree_sync = True # 防止反向触发循环
            self.highlight_event(idx, "red")
            self._disable_tree_sync = False

    def on_double_click_list(self, event):
        sel = self.events_listbox.curselection()
        if sel:
            t = self.time_axis[self.display_items[sel[0]]["sample_idx"]]
            self.start_time_var.set(min(max(0, t - self.winlen_var.get()/2), self.start_scale.cget("to")))
            self.update_plot_window()

    def on_key_prev_next(self, event): self.master.after(50, lambda: self.on_event_selected(None))
    def on_enter_key(self, event):
        sel = self.events_listbox.curselection()
        if sel: self.promote_selected_to_red() if self.display_items[sel[0]]["kind"] == "blue" else self.demote_selected_to_blue()

    def on_list_right_click(self, event):
        idx = self.events_listbox.nearest(event.y)
        self.events_listbox.selection_clear(0, tk.END); self.events_listbox.selection_set(idx)
        self.on_event_selected(None)
        self.demote_selected_to_blue() if self.display_items[idx]["kind"] == "red" else self.promote_selected_to_red()
    def on_delete_key(self, event):
            sel = self.events_listbox.curselection()
            if not sel: return
            
            idx_in_list = int(sel[0])
            item = self.display_items[idx_in_list]
            sample_idx = int(item["sample_idx"])
            kind = item["kind"]
            
            # 1. 彻底剔除数据
            if kind == "red":
                self.events_red_idx = self._remove_value(self.events_red_idx, sample_idx)
                self.recompute_all_kinetics() 
            else:
                self.events_blue_idx = self._remove_value(self.events_blue_idx, sample_idx)
                
            self.log(f"Completely deleted {kind} event: t={self.time_axis[sample_idx]*1000:.2f} ms")
            
            # 2. 刷新所有 UI 和线条
            self.populate_events_list()
            self.update_vlines()
            self.clear_selected_event_highlight()
            
            # 3. 智能焦点保持：删完后自动选中下一个，如果删的是最后一个就往上选
            if self.display_items:
                new_sel = min(idx_in_list, len(self.display_items) - 1)
                next_item = self.display_items[new_sel]
                # 直接调用底层函数，一步到位完成 选中 + 聚焦 + 滚动 + 联动
                self.highlight_event(next_item["sample_idx"], next_item["kind"])
    def enable_add_event(self):
        self.add_event_mode = True
        self.master.config(cursor="crosshair")
        self.log("Click on trace to ADD a RED event.")

    def perform_add_event(self, t_click):
        idx = min(max(0, int(t_click * self.fs)), len(self.trace)-1)
        self.events_red_idx = self._insert_sorted_unique(self.events_red_idx, idx)
        self.events_blue_idx = self._remove_value(self.events_blue_idx, idx)
        self.log(f"Added event at {t_click:.4f} s")
        self.add_event_mode = False
        self.master.config(cursor="")
        self.recompute_all_kinetics()
        self.populate_events_list(); self.update_vlines(); self.highlight_event(idx, "red")

    def promote_selected_to_red(self):
        sel = self.events_listbox.curselection()
        if sel and self.display_items[sel[0]]["kind"] == "blue":
            idx = self.display_items[sel[0]]["sample_idx"]
            self.events_blue_idx = self._remove_value(self.events_blue_idx, idx)
            self.events_red_idx = self._insert_sorted_unique(self.events_red_idx, idx)
            self.recompute_all_kinetics()
            self.populate_events_list(); self.update_vlines(); self.highlight_event(idx, "red")

    def demote_selected_to_blue(self):
        sel = self.events_listbox.curselection()
        if sel and self.display_items[sel[0]]["kind"] == "red":
            idx = self.display_items[sel[0]]["sample_idx"]
            self.events_red_idx = self._remove_value(self.events_red_idx, idx)
            self.events_blue_idx = self._insert_sorted_unique(self.events_blue_idx, idx)
            self.recompute_all_kinetics()
            self.populate_events_list(); self.update_vlines(); self.highlight_event(idx, "blue")

    def go_prev_event(self):
        if len(self.events_red_idx) > 0:
            mask = self.time_axis[self.events_red_idx] < (self.start_time_var.get() + self.winlen_var.get()/2 - 0.001)
            if np.any(mask): self._jump_to_event(self.events_red_idx[mask][-1], "red")

    def go_next_event(self):
        if len(self.events_red_idx) > 0:
            mask = self.time_axis[self.events_red_idx] > (self.start_time_var.get() + self.winlen_var.get()/2 + 0.001)
            if np.any(mask): self._jump_to_event(self.events_red_idx[mask][0], "red")

    def _jump_to_event(self, idx, kind):
        self.start_time_var.set(min(max(0, self.time_axis[idx] - self.winlen_var.get()/2), self.start_scale.cget("to")))
        self.update_plot_window(); self.highlight_event(idx, kind)

    # ==============================================================================
    # 彻底告别虚高的导出逻辑 (对接原汁原味高频采样数据)
    # ==============================================================================
    
    def _write_events_csv(self, path: Path):
        if self.events_red_idx is None or len(self.events_red_idx) == 0: return

        raw_data = self.trace_raw 
        raw_fs = self.fs_orig
        direction = -1 if self.polarity_var.get() == "epsc" else 1
        is_ipsc = (self.polarity_var.get() == "ipsc")
        scale_ratio = (raw_fs / self.fs) if (raw_fs and self.fs) else 1.0

        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "event_index", "time_s", "time_ms", "sample_index", "native_sample_index", 
                "Baseline_pA", "Amplitude_pA", "Rise_10_90_ms", "Decay_Tau_ms", "Half_Width_ms", "Area_fC"
            ])
            
            for i, idx in enumerate(self.events_red_idx):
                idx = int(idx)
                if idx < len(self.time_axis):
                    t_s = float(self.time_axis[idx])
                    native_idx = int(round(idx * scale_ratio))
                    
                    if i + 1 < len(self.events_red_idx):
                        next_native_idx = int(round(self.events_red_idx[i+1] * scale_ratio))
                    else:
                        next_native_idx = None
                    
                    # 导出时在原版高频数据上跑同样的寻谷截断逻辑
                    kin = compute_event_kinetics_absolute(raw_data, native_idx, raw_fs, direction=direction, next_peak_idx=next_native_idx, is_ipsc=is_ipsc)
                    
                    if kin:
                        base, amp, rise, decay, hw, area = (
                            kin['Baseline_pA'], kin['Amplitude_pA'], 
                            kin['Rise_10_90_ms'], kin['Decay_Tau_ms'], 
                            kin['Half_Width_ms'], kin['Area_fC']
                        )
                    else:
                        base = amp = rise = decay = hw = area = np.nan
                        
                    writer.writerow([i, t_s, t_s * 1000.0, idx, native_idx, base, amp, rise, decay, hw, area])

    def _write_native_evt(self, path_evt: Path, path_template: Path):
        try:
            with path_template.open('rb') as f: header = bytearray(f.read(2048))
            header[1986:1990] = struct.pack('<I', len(self.events_red_idx))
            ratio = (self.fs_orig / self.fs) if (self.fs_orig and self.fs) else 1.0
            
            records = bytearray()
            for idx in self.events_red_idx:
                idx = int(idx)
                peak_index = int(idx * ratio)
                start_index = max(0, peak_index - int(2.0 * self.fs_orig / 1000.0))
                amp = float(self.trace[idx]) if idx < len(self.trace) else 0.0
                
                rec = bytearray(64)
                struct.pack_into('<f', rec, 0, float(peak_index)) 
                struct.pack_into('<f', rec, 16, amp)       
                struct.pack_into('<I', rec, 56, start_index)
                struct.pack_into('<I', rec, 60, peak_index)       
                records.extend(rec)
            
            with path_evt.open('wb') as f: f.write(header); f.write(records)
            self.log(f"Successfully generated perfect native EVT: {path_evt}")
            
        except Exception as e:
            self.log(f"Failed to generate EVT: {e}")

    def _write_minianalysis_asc(self, path: Path):
        if self.events_red_idx is None or len(self.events_red_idx) == 0: return
        with path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter='\t')
            for idx in self.events_red_idx:
                idx = int(idx)
                if idx < len(self.time_axis) and idx < len(self.trace):
                    writer.writerow([f"{float(self.time_axis[idx]) * 1000.0:.5f}", f"{float(self.trace[idx]):.5f}"])

    def export_events(self):
        if self.events_red_idx is None or len(self.events_red_idx) == 0:
            messagebox.showinfo("Export Events", "No RED events to export.")
            return

        default_name = self.csv_path.with_name(self.csv_path.stem + "_Visual_Kinetics_Ultimate.csv") if self.csv_path else Path("Visual_Kinetics_Ultimate.csv")

        path_str = filedialog.asksaveasfilename(title="Save Events CSV", defaultextension=".csv", initialfile=str(default_name.name), filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path_str: return

        path_csv = Path(path_str)
        try:
            self._write_events_csv(path_csv)
            self.events_export_path = path_csv
            self.log(f"Exported kinetics CSV to {path_csv}")
            
            path_asc = path_csv.with_suffix(".asc")
            self._write_minianalysis_asc(path_asc)
            self.log(f"Exported ASCII to {path_asc}")
            
            if messagebox.askyesno("Generate EVT", "Do you want to generate a perfect binary .EVT file?"):
                template_path = filedialog.askopenfilename(title="Select Template .EVT file", filetypes=[("EVT Files", "*.EVT"), ("All Files", "*.*")])
                if template_path: self._write_native_evt(path_csv.with_suffix(".EVT"), Path(template_path))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export:\n{e}")

def main():
    root = tk.Tk()
    app = EPSCGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()