# -*- coding: utf-8 -*-
r"""
unet_ola_fp_fn_ipsc.py
双模型两阶段（clean_ipsc + slowkinetics）推理 -> 并集 -> 与GT对齐计算 P / R / F1
- 读取: C:\Users\Michael\Desktop\lab1\2019-05-17\19518008_trace.csv 及同目录下 GT _events.csv
- 模型: clean_ipsc 与 slowkinetics 两个 .keras
- 推理: OLA（overlap-add）方式滑窗推理，输出概率轨迹
- 并集: 两模型检测结果合并并在 merge_within_ms 内去重
- 评估: 与 GT 对齐（match_tol_ms）计算 P / R / F1
"""

import os, json, zipfile, time, tempfile, shutil
from pathlib import Path
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, convolve, resample_poly

# SciPy compatibility: gaussian is under scipy.signal.windows in many versions
try:
    from scipy.signal.windows import gaussian  # SciPy >= 1.2 typically
except Exception:
    # Fallback: implement gaussian window ourselves
    import numpy as np
    def gaussian(M, std, sym=True):
        """
        Minimal replacement for scipy.signal.windows.gaussian.
        M: window length
        std: standard deviation (in samples)
        """
        if M <= 0:
            return np.array([])
        n = np.arange(0, M) - (M - 1.0) / 2.0
        w = np.exp(-0.5 * (n / float(std)) ** 2)
        return w

import tensorflow as tf

# ==============================
# 全局配置
# ==============================

ENS2_TARGET_FS = 10000.0  # 统一采样率 Hz

PARAMS_CONFIG = {
    "ipsc": {
        "thr_high": 0.60,
        "thr_low":  0.30,
        "min_peak_distance_ms": 9.0,
        "gaussian_sigma_ms":   2.0,
        "merge_within_ms":     5.0,
        "match_tol_ms":        8.0,
        "win_ms":             256.0,
        "hop_ms":              32.0,
        "batch_size":           32,
    }
}
POLARITY = "ipsc"
CURRENT_PARAMS = PARAMS_CONFIG[POLARITY]

# 路径（按你的机器；保留沙箱 fallback）
CLEAN_MODEL_PATH_WIN = r"C:\Users\Michael\Desktop\lab1\Unet\best_ens2_clean_ipsc.keras"
SLOW_MODEL_PATH_WIN  = r"C:\Users\Michael\Desktop\lab1\Unet\best_ipsc_slowkinetics.keras"
CLEAN_MODEL_PATH_FALLBACK = "/mnt/data/best_ens2_clean_ipsc.keras"
SLOW_MODEL_PATH_FALLBACK  = "/mnt/data/best_ipsc_slowkinetics.keras"

DATA_DIR  = r"C:\Users\Michael\Desktop\lab1\2019-05-17"
TARGET_ID = "19518008"

# ==============================
# 自定义层：时间轴对齐
# ==============================

class Match1DLike(tf.keras.layers.Layer):
    """
    将 x1 在 time 维度上裁剪/零填充到与 x2 的长度一致：(B, T1, C) -> (B, T2, C)
    """
    def call(self, inputs):
        x1, x2 = inputs
        t1 = tf.shape(x1)[1]
        t2 = tf.shape(x2)[1]
        def crop():
            return x1[:, :t2, :]
        def pad():
            pad_len = t2 - t1
            paddings = tf.stack([[0,0], [0,pad_len], [0,0]])
            return tf.pad(x1, paddings)
        return tf.cond(t1 >= t2, crop, pad)

# ==============================
# 读取 .keras 的输入长度（T）
# ==============================

def _read_input_len_from_keras(keras_path: str, default_len: int = 480) -> int:
    """从 .keras 包中解析 InputLayer 的 (None, T, 1) -> 拿到 T。失败则用默认值。"""
    try:
        with zipfile.ZipFile(keras_path, "r") as zf:
            cand = [n for n in zf.namelist() if n.endswith(("config.json","model.json"))]
            if not cand:
                return default_len
            cfg = json.loads(zf.read(cand[0]).decode("utf-8"))
            layers = cfg.get("config", {}).get("layers", []) or cfg.get("layers", [])
            for L in layers:
                if L.get("class_name") == "InputLayer":
                    shp = L.get("config",{}).get("batch_input_shape") or L.get("config",{}).get("batch_shape")
                    if shp and len(shp) == 3:
                        return int(shp[1])
    except Exception:
        pass
    return default_len

# ==============================
# 模型结构（膨胀瓶颈 U-Net，双头输出）
# ==============================

def build_clean_ens2_unet(input_len: int = 480, base: int = 32) -> tf.keras.Model:
    """
    重建与训练时一致的双头 U-Net（膨胀卷积 bottleneck 版本）
    """
    inp = tf.keras.Input(shape=(input_len, 1), name="input_trace")

    # Encoder
    def conv_block(x, ch, i):
        x = tf.keras.layers.Conv1D(ch, 3, padding="same", name=f"enc_{i}_conv1")(x)
        x = tf.keras.layers.BatchNormalization(name=f"enc_{i}_bn1")(x)
        x = tf.keras.layers.ReLU(name=f"enc_{i}_relu1")(x)
        x = tf.keras.layers.Conv1D(ch, 3, padding="same", name=f"enc_{i}_conv2")(x)
        x = tf.keras.layers.BatchNormalization(name=f"enc_{i}_bn2")(x)
        x = tf.keras.layers.ReLU(name=f"enc_{i}_relu2")(x)
        x = tf.keras.layers.Dropout(0.0, name=f"enc_{i}_dropout")(x)
        return x

    e1 = conv_block(inp, base, 1);   p1 = tf.keras.layers.MaxPooling1D(2, name="pool_1")(e1)
    e2 = conv_block(p1, base*2, 2);  p2 = tf.keras.layers.MaxPooling1D(2, name="pool_2")(e2)
    e3 = conv_block(p2, base*4, 3)

    # Dilated bottleneck
    bottleneck_channels = base * 8  # 256 when base=32
    def dilated(x, rate):
        y = tf.keras.layers.Conv1D(bottleneck_channels, 7, padding='same',
                                   dilation_rate=rate, name=f"bottleneck_dilated_{rate}")(x)
        y = tf.keras.layers.BatchNormalization(name=f"bottleneck_bn_dilated_{rate}")(y)
        y = tf.keras.layers.ReLU(name=f"bottleneck_relu_dilated_{rate}")(y)
        return y
    d2, d4, d8, d16 = dilated(e3,2), dilated(e3,4), dilated(e3,8), dilated(e3,16)
    b = tf.keras.layers.Concatenate(name="bottleneck_concat")([d2, d4, d8, d16])
    b = tf.keras.layers.Conv1D(bottleneck_channels, 1, padding='same', name="bottleneck_compress")(b)
    b = tf.keras.layers.BatchNormalization(name=f"bottleneck_bn_compress")(b)
    b = tf.keras.layers.ReLU(name=f"bottleneck_relu_compress")(b)
    b = tf.keras.layers.Dropout(0.0, name="bottleneck_dropout")(b)

    # Decoder
    def decoder_block(x, ch, i):
        x = tf.keras.layers.Conv1D(ch, 3, padding="same", name=f"dec_{i}_conv1")(x)
        x = tf.keras.layers.BatchNormalization(name=f"dec_{i}_bn1")(x)
        x = tf.keras.layers.ReLU(name=f"dec_{i}_relu1")(x)
        x = tf.keras.layers.Conv1D(ch, 3, padding="same", name=f"dec_{i}_conv2")(x)
        x = tf.keras.layers.BatchNormalization(name=f"dec_{i}_bn2")(x)
        x = tf.keras.layers.ReLU(name=f"dec_{i}_relu2")(x)
        x = tf.keras.layers.Dropout(0.0, name=f"dec_{i}_dropout")(x)
        return x

    d3 = decoder_block(b, base*4, 3)
    u2 = tf.keras.layers.UpSampling1D(2, name="up_2")(d3)
    c2 = Match1DLike(name="crop_2")([u2, e2])
    d2 = decoder_block(tf.keras.layers.Concatenate(name="cat_2")([c2, e2]), base*2, 2)
    u1 = tf.keras.layers.UpSampling1D(2, name="up_1")(d2)
    c1 = Match1DLike(name="crop_1")([u1, e1])
    d1 = decoder_block(tf.keras.layers.Concatenate(name="cat_1")([c1, e1]), base, 1)

    # 双头输出
    prob = tf.keras.layers.Conv1D(1, 1, activation='sigmoid', name="prob_head")(d1)
    rate = tf.keras.layers.Conv1D(1, 1, activation='relu',    name="rate_head")(d1)
    return tf.keras.Model(inp, [prob, rate], name="ipsc_ens2_slowkinetics")

# ==============================
# Loader：直读失败则从 .keras 解出 .weights.h5 -> 重建按名加载
# ==============================

ENS2_OUTPUTS_LOGIT = False  # 若模型输出是logit则外部再sigmoid，一般 False

def _extract_weights_h5_from_keras(keras_path: str):
    """
    从 Keras 3 的 .keras 单文件（zip 包）中提取权重 .h5
    返回: (weights_h5_path, temp_dir)；若找不到返回 (None, None)
    """
    try:
        with zipfile.ZipFile(keras_path, "r") as zf:
            names = zf.namelist()
            cands = [n for n in names if n.lower().endswith((".weights.h5", ".h5"))
                     and any(k in n.lower() for k in ("weight", "variable", "model"))]
            if not cands:
                return None, None
            cands.sort(key=len)
            tmpdir = Path(tempfile.mkdtemp(prefix="keras_unpack_"))
            out = tmpdir / Path(cands[0]).name
            with zf.open(cands[0]) as src, open(out, "wb") as dst:
                dst.write(src.read())
            return str(out), str(tmpdir)
    except Exception:
        return None, None

def load_ens2_model_robust(path: str):
    """
    先尝试直接 load_model（若 Keras3 则 safe_mode=False 允许 Lambda）；
    若失败：从 .keras 内提取 .weights.h5，重建结构并按名加载（by_name=True, skip_mismatch=True）。
    """
    custom_objects = {'Match1DLike': Match1DLike}
    path = str(path)

    # 目录（SavedModel）
    if os.path.isdir(path):
        model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
    else:
        try:
            print("尝试直接 load_model(..., safe_mode=False)")
            model = tf.keras.models.load_model(path, custom_objects=custom_objects,
                                               compile=False, safe_mode=False)
            print("直接加载成功")
        except Exception as e:
            print(f"直接加载失败：{e}")
            print("回退到：从 .keras 解出权重 -> 重建结构 -> 按名加载")
            T = _read_input_len_from_keras(path, default_len=480)
            print(f"检测到输入长度: {T} 时间点")

            # 重建（膨胀瓶颈版）
            model = build_clean_ens2_unet(T, base=32)

            # 从 .keras 解出权重 .h5 并加载
            wfile, tmpdir = _extract_weights_h5_from_keras(path)
            try:
                if wfile is None:
                    raise RuntimeError("未在 .keras 内找到权重 .h5 文件")
                model.load_weights(wfile, by_name=True, skip_mismatch=True)
                print(f"权重加载完成：{wfile}")
            except Exception as e2:
                # 清理并抛出更清晰的提示
                if tmpdir and os.path.isdir(tmpdir):
                    shutil.rmtree(tmpdir, ignore_errors=True)
                raise RuntimeError(
                    "从 .keras 提取权重失败。建议：提供同名 .weights.h5，或在旧环境加载，或先转存为 SavedModel。\n"
                    f"原始错误：{e2}"
                )
            finally:
                if tmpdir and os.path.isdir(tmpdir):
                    shutil.rmtree(tmpdir, ignore_errors=True)

    # 检测 prob_head 激活（一般已有 sigmoid）
    global ENS2_OUTPUTS_LOGIT
    try:
        prob_head = next(l for l in model.layers if l.name == "prob_head")
        act_name = getattr(prob_head.activation, "__name__", str(prob_head.activation))
        ENS2_OUTPUTS_LOGIT = ("linear" in act_name.lower()) or ("none" in act_name.lower())
        print(f"检测到 prob_head 激活: {act_name}, need_sigmoid={ENS2_OUTPUTS_LOGIT}")
    except Exception as e:
        print(f"激活检测失败，保守设置 need_sigmoid=False: {e}")
        ENS2_OUTPUTS_LOGIT = False

    return model

# ==============================
# I/O 与前处理
# ==============================

def _first_exist(*paths):
    for p in paths:
        if p and Path(p).exists():
            return p
    return paths[0]

def _infer_fs_from_time_ms(time_ms):
    time_ms = np.asarray(time_ms, dtype=float)
    if len(time_ms) < 3:
        return None
    diffs = np.diff(time_ms)
    med = np.median(diffs)
    if med <= 0:
        return None
    fs = 1000.0 / med
    return fs

def load_trace_from_csv(csv_path):
    """
    读取 trace：
    - 支持 (time_ms, value)、(time_s, value) 或单列 value（默认 fs=10k）
    - 根据列名判断时间单位：含 'ms' 按毫秒，含 's'/'sec' 且不含 'ms' 按秒
    - 自动从时间差估计采样率 fs
    返回：trace(np.float32), fs(Hz)
    """
    df = pd.read_csv(csv_path)
    cols_lower = [str(c).strip().lower() for c in df.columns]
    df.columns = cols_lower

    time_candidates = [c for c in cols_lower if "time" in c]
    sig_candidates  = [c for c in cols_lower if c in ("trace","value","signal","current_pa","current(pa)")]
    if not sig_candidates and len(cols_lower) >= 2:
        sig_candidates = [cols_lower[1]]

    if time_candidates and sig_candidates:
        tcol = time_candidates[0]
        scol = sig_candidates[0]
        t = df[tcol].astype(float).to_numpy()
        x = df[scol].astype(float).to_numpy().astype(np.float32)
        tname = tcol
        if ("ms" in tname):
            dt = np.median(np.diff(t)); fs = 1000.0 / dt if dt > 0 else ENS2_TARGET_FS
        elif (("time_s" in tname) or ("sec" in tname) or (("s" in tname) and ("ms" not in tname))):
            dt = np.median(np.diff(t)); fs = 1.0 / dt if dt > 0 else ENS2_TARGET_FS
        else:
            dt = np.median(np.diff(t))
            if dt > 0 and dt < 0.02: fs = 1.0 / dt
            elif dt > 0:            fs = 1000.0 / dt
            else:                   fs = ENS2_TARGET_FS
        return x, float(fs)
    elif len(cols_lower) == 1:
        x = df[cols_lower[0]].astype(float).to_numpy().astype(np.float32)
        return x, float(ENS2_TARGET_FS)
    else:
        t = df.iloc[:, 0].astype(float).to_numpy()
        x = df.iloc[:, 1].astype(float).to_numpy().astype(np.float32)
        dt = np.median(np.diff(t))
        if dt > 0 and dt < 0.02: fs = 1.0 / dt
        elif dt > 0:            fs = 1000.0 / dt
        else:                   fs = ENS2_TARGET_FS
        return x, float(fs)

def load_events_from_csv(csv_path):
    """
    读取 GT 事件时间（毫秒），鲁棒处理：
    - 支持类似 '169 Time (ms)' 的列名（模糊匹配包含 'time' 且包含 'ms'）
    - 清洗千分位逗号、小尾巴单位与杂字
    - to_numeric(errors='coerce')，无法解析的行丢弃并打印样例
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(csv_path)

    orig_cols = list(df.columns)
    lower_cols = [str(c).strip().lower() for c in orig_cols]
    df.columns = lower_cols

    cand_ms = [c for c in df.columns if ('time' in c and 'ms' in c)]
    if cand_ms: col = cand_ms[0]
    else:
        cand_t = [c for c in df.columns if 'time' in c]
        col = cand_t[0] if cand_t else df.columns[0]

    s = df[col].astype(str)
    s = s.str.replace(",", "", regex=False).str.strip()  # 去千分位逗号
    s_clean = s.str.replace(r"[^0-9+\-\.eE]", "", regex=True)
    vals = pd.to_numeric(s_clean, errors="coerce")

    bad_mask = vals.isna()
    bad_cnt = int(bad_mask.sum())
    if bad_cnt > 0:
        examples = s[bad_mask].head(5).tolist()
        print(f"⚠️  GT时间列中有 {bad_cnt} 条无法解析（已忽略）。样例：{examples}")

    vals = vals.dropna().astype(float).to_numpy()
    return [{"time_ms": float(x)} for x in vals]

def resample_to_target_fs(x, fs_in, fs_target):
    """
    用 resample_poly 做重采样（ratio 逼近为 up/down）
    """
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(fs_in) or fs_in <= 0:
        return x.copy(), float(fs_target)
    if not np.isfinite(fs_target) or fs_target <= 0:
        fs_target = ENS2_TARGET_FS
    if abs(fs_in - fs_target) / fs_target < 1e-6:
        return x.copy(), float(fs_in)

    ratio = float(fs_target) / float(fs_in)
    frac = Fraction(ratio).limit_denominator(1000)
    up, down = int(frac.numerator), int(frac.denominator)
    up = max(1, up); down = max(1, down)
    y = resample_poly(x, up, down).astype(np.float32)
    return y, float(fs_target)

def robust_standardize(x):
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    z = x - med
    mad = np.median(np.abs(z)) * 1.4826
    scale = mad if mad > 1e-6 else max(np.std(x), 1e-3)
    return z / scale

# ==============================
# OLA 推理
# ==============================

def make_windows_ola(x, fs, win_ms, hop_ms):
    win_len = int(round(win_ms * 1e-3 * fs))
    hop_len = int(round(hop_ms * 1e-3 * fs))
    win_len = max(win_len, 16)
    hop_len = max(hop_len, 1)

    L = len(x)
    if L <= win_len:
        pad_total = win_len - L
    else:
        n_hops = int(np.ceil((L - win_len) / hop_len)) + 1
        L_target = (n_hops - 1) * hop_len + win_len
        pad_total = max(0, L_target - L)

    x_pad = np.pad(x, (0, pad_total), mode='edge')
    windows = []
    i = 0
    while i + win_len <= len(x_pad):
        windows.append(x_pad[i:i+win_len])
        i += hop_len

    windows = np.stack(windows, axis=0).astype(np.float32)
    windows = windows[..., np.newaxis]  # (N, L, 1)
    return windows, win_len, hop_len, pad_total

def ola_reconstruct(probs_win, win_len, hop_len, pad_len, L_orig):
    """
    OLA 重建（稳健版）：
    - 自动以 probs_win 的第二维作为窗口长度，避免与 win_len 不一致导致的 broadcast 错误
    - 忽略 pad_len，直接在最后对齐到 L_orig
    """
    probs_win = _ensure_NL(probs_win)        # (N, Lw)
    N, Lw = probs_win.shape                  # Lw = 实际每窗输出长度
    win_len_eff = int(Lw)

    total_len = (N - 1) * hop_len + win_len_eff
    out = np.zeros(total_len, dtype=np.float32)
    cnt = np.zeros(total_len, dtype=np.float32)

    pos = 0
    for k in range(N):
        out[pos:pos + win_len_eff] += probs_win[k]
        cnt[pos:pos + win_len_eff] += 1.0
        pos += hop_len

    cnt[cnt == 0] = 1.0
    out = out / cnt

    # 对齐到原始长度（更稳；pad_len 差异也不再影响）
    if L_orig is not None and L_orig > 0:
        out = out[:L_orig]

    return out.astype(np.float32)


def gaussian_smooth(x, fs, sigma_ms):
    sigma_samp = max(1, int(round((sigma_ms * 1e-3) * fs)))
    k_len = int(6 * sigma_samp) + 1
    if k_len < 7: k_len = 7
    win = gaussian(k_len, std=sigma_samp); win = win / np.sum(win)
    y = convolve(x, win, mode='same')
    return y.astype(np.float32)

def _ensure_NL(y):
    """
    把模型窗口输出压成二维 (N, L):
    - (N, L, 1) -> (N, L)
    - (N, L, 1, 1) -> (N, L)
    - 其它奇怪形状尽量 squeeze，最后兜底到二维
    """
    y = np.asarray(y)
    if y.ndim == 4 and y.shape[-1] == 1 and y.shape[-2] == 1:
        y = y[..., 0, 0]
    if y.ndim == 3 and y.shape[-1] == 1:
        y = y[..., 0]
    y = np.squeeze(y)
    if y.ndim == 1:  # 万一是 (L,)（只有一个窗口）
        y = y[None, :]
    if y.ndim != 2:
        # 兜底：强行取前两维作为 (N, L)
        y = y.reshape(y.shape[0], -1)
    return y.astype(np.float32)


def model_predict_windows(model, X, batch_size=32):
    """
    批量前向，返回二维 (N, L) 的窗口概率：
    - 模型是双头输出 -> 只取第一个头（prob）
    - 无论是 (N,L) / (N,L,1) / (N,L,1,1) 都压成 (N,L)
    """
    y = model.predict(X, batch_size=batch_size, verbose=0)
    if isinstance(y, (list, tuple)):
        y = y[0]  # 只用概率头
    return _ensure_NL(y)

def _match_count_ms(pred_ms, gt_ms, tol_ms):
    """在时间单位=毫秒的空间里计算 TP/FP/FN。pred_ms, gt_ms 都应是升序。"""
    pred_ms = np.asarray(pred_ms, dtype=float)
    gt_ms   = np.asarray(gt_ms,   dtype=float)
    used = np.zeros(pred_ms.size, dtype=bool)
    tp = 0
    for g in gt_ms:
        k = np.searchsorted(pred_ms, g)
        cands = []
        if k < pred_ms.size: cands.append(k)
        if k > 0:            cands.append(k-1)
        for idx in cands:
            if not used[idx] and abs(pred_ms[idx] - g) <= tol_ms:
                used[idx] = True
                tp += 1
                break
    fp = int((~used).sum())
    fn = int(gt_ms.size - tp)
    return tp, fp, fn

def best_time_offset_ms(pred_idx, fs, gt_ms, tol_ms=8.0, search_ms=80.0, step_ms=0.5):
    """
    在 [-search_ms, +search_ms] 范围内网格搜索一个全局时间偏移，使 TP 最大。
    返回: (best_offset_ms, best_metrics_dict)
    """
    if len(pred_idx) == 0 or len(gt_ms) == 0:
        return 0.0, {"precision":0.0, "recall":0.0, "f1":0.0, "tp":0, "fp":len(pred_idx), "fn":len(gt_ms)}
    pred_ms = (np.asarray(pred_idx, dtype=float) / float(fs)) * 1000.0
    pred_ms.sort()
    gt_ms_sorted = np.sort(np.asarray(gt_ms, dtype=float))

    offs = np.arange(-search_ms, search_ms + 1e-9, step_ms, dtype=float)
    best = None
    best_off = 0.0
    for d in offs:
        tp, fp, fn = _match_count_ms(pred_ms + d, gt_ms_sorted, tol_ms)
        # 以 TP 优先，其次用更高 F1 打破平手
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*p*r / (p + r) if (p + r) else 0.0
        key = (tp, f1)
        if (best is None) or (key > best[0]):
            best = (key, {"precision":p,"recall":r,"f1":f1,"tp":tp,"fp":fp,"fn":fn})
            best_off = float(d)
    return best_off, best[1]


def _to_prob01(arr):
    """把输出转成 [0,1] 概率。
    - 若全局 ENS2_OUTPUTS_LOGIT=True -> 做 sigmoid
    - 或数值范围像 logit（比如<-5..>5） -> 自动做 sigmoid
    """
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0: return a
    need_sig = ENS2_OUTPUTS_LOGIT
    if not need_sig:
        a_min, a_max = float(np.min(a)), float(np.max(a))
        if (a_min < -4.0 and a_max > 4.0) or (a_min < -10.0) or (a_max > 10.0):
            need_sig = True
    if need_sig:
        a = 1.0 / (1.0 + np.exp(-a))
    return np.clip(a, 0.0, 1.0)

def blob_based_inference(x, fs, model, polarity="ipsc", params=None):
    """
    根据模型 input_shape[1] 自动设置窗口长度；若取不到，则退回到 params['win_ms']。
    """
    if params is None:
        params = CURRENT_PARAMS

    # 1) 取模型的窗口长度（样本数）并转换为毫秒
    T_model = _model_win_len_samples(model)  # e.g., 480
    if T_model is not None and fs > 0:
        win_ms_eff = (float(T_model) / float(fs)) * 1000.0
    else:
        win_ms_eff = float(params["win_ms"])  # fallback

    # 2) OLA 切窗（使用 win_ms_eff，而不是固定的 params['win_ms']）
    z = robust_standardize(x)
    windows, win_len, hop_len, pad_len = make_windows_ola(
        z, fs, win_ms=win_ms_eff, hop_ms=params["hop_ms"]
    )

    # 3) 前向
    probs_win = model_predict_windows(model, windows, batch_size=params["batch_size"])
    probs_win = _to_prob01(probs_win)
    prob = ola_reconstruct(probs_win, win_len, hop_len, pad_len, len(x))
    prob = _to_prob01(prob) 
    prob_s = gaussian_smooth(prob, fs, sigma_ms=params["gaussian_sigma_ms"])

    # 4) 峰值检测
    min_dist = int(round(params["min_peak_distance_ms"] * 1e-3 * fs))
    min_dist = max(1, min_dist)
    height = params["thr_high"]
    if polarity == "ipsc":
        peaks, _ = find_peaks(prob_s, distance=min_dist, height=height)
        events_idx = peaks.astype(int)
    else:
        inv = -prob_s
        peaks, _ = find_peaks(inv, distance=min_dist, height=height)
        events_idx = peaks.astype(int)

    extras = {"prob_raw": prob, "prob_smooth": prob_s, "win_ms_used": win_ms_eff}
    return events_idx, prob_s, extras

# ==============================
# 并集 & 评估
# ==============================

def union_events(e1, e2, fs, merge_within_ms=5.0):
    e1 = np.asarray(e1, dtype=int)
    e2 = np.asarray(e2, dtype=int)
    if e1.size == 0 and e2.size == 0:
        return np.array([], dtype=int)
    if e1.size and e2.size:
        all_idx = np.sort(np.unique(np.concatenate([e1, e2])))
    else:
        all_idx = e1 if e2.size == 0 else e2
    if all_idx.size == 0:
        return all_idx
    gap = int(round(float(merge_within_ms) * 1e-3 * float(fs)))
    gap = max(gap, 1)
    merged = [all_idx[0]]
    for t in all_idx[1:]:
        if (t - merged[-1]) <= gap:
            continue
        merged.append(t)
    return np.array(merged, dtype=int)

def eval_events(pred_idx, fs, gt_times_ms, tol_ms=8.0):
    if len(gt_times_ms) == 0:
        return 0, len(pred_idx), 0
    pred_ms = (np.asarray(pred_idx, dtype=float) / float(fs)) * 1000.0
    pred_ms = np.sort(pred_ms)
    gt_ms = np.sort(np.asarray(gt_times_ms, dtype=float))
    tol = float(tol_ms)
    used = np.zeros(len(pred_ms), dtype=bool)
    tp = 0
    for g in gt_ms:
        k = np.searchsorted(pred_ms, g)
        candidates = []
        if k < len(pred_ms): candidates.append(k)
        if k > 0:            candidates.append(k-1)
        for idx in candidates:
            if not used[idx] and abs(pred_ms[idx] - g) <= tol:
                used[idx] = True
                tp += 1
                break
    fp = int((~used).sum())
    fn = int(len(gt_ms) - tp)
    return tp, fp, fn

def compute_metrics(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*p*r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}

def _model_win_len_samples(model):
    """从模型的 input_shape 取窗口长度（样本数），拿不到就返回 None。"""
    try:
        # 兼容 Keras 2/3：可能是 TensorShape 或元组
        ishape = getattr(model, "input_shape", None)
        if isinstance(ishape, (list, tuple)):
            # 可能是多输入，取第一个
            if isinstance(ishape[0], (list, tuple)):
                ishape = ishape[0]
        if ishape is None:
            return None
        T = int(ishape[1])
        return T if T and T > 0 else None
    except Exception:
        return None


# ==============================
# 主流程
# ==============================

def main():
    clean_model_path = _first_exist(CLEAN_MODEL_PATH_WIN, CLEAN_MODEL_PATH_FALLBACK)
    slow_model_path  = _first_exist(SLOW_MODEL_PATH_WIN,  SLOW_MODEL_PATH_FALLBACK)
    trace_csv  = Path(DATA_DIR) / f"{TARGET_ID}_trace.csv"
    events_csv = Path(DATA_DIR) / f"{TARGET_ID}_events.csv"

    if not trace_csv.exists():
        print(f"❌ Trace 不存在: {trace_csv}"); return
    if not events_csv.exists():
        print(f"❌ GT Events 不存在: {events_csv}"); return
    if not Path(clean_model_path).exists():
        print(f"❌ clean_ipsc 模型不存在: {clean_model_path}"); return
    if not Path(slow_model_path).exists():
        print(f"❌ slowkinetics 模型不存在: {slow_model_path}"); return

    params = CURRENT_PARAMS
    EVAL_TOL_MS    = params["match_tol_ms"]
    MERGE_WITHIN_MS= params["merge_within_ms"]

    print("=== 双模型两阶段推理 + 并集评估 (IPSC) ===")
    print(f"  clean_ipsc   : {clean_model_path}")
    print(f"  slowkinetics : {slow_model_path}")
    print(f"  数据         : {trace_csv.name} / {events_csv.name}")
    print(f"  匹配容差     : {EVAL_TOL_MS} ms | 并集去重: {MERGE_WITHIN_MS} ms\n")

    # 载入数据
    raw_trace, fs_in = load_trace_from_csv(trace_csv)
    gt_events = load_events_from_csv(events_csv)
    gt_ms_all = [e["time_ms"] for e in gt_events]

    print(f"原始Trace: {len(raw_trace)/fs_in:.2f}s @ {fs_in:.1f}Hz,  GT={len(gt_ms_all)}")
    trace, fs = resample_to_target_fs(raw_trace, fs_in, ENS2_TARGET_FS)
    print(f"重采样: -> {len(trace)/fs:.2f}s @ {fs:.1f}Hz\n")

    test_duration_ms = (len(trace) / fs) * 1000.0
    gt_ms = [t for t in gt_ms_all if 0.0 <= t <= test_duration_ms]
    print(f"测试范围内 GT 事件: {len(gt_ms)}\n")

    # 加载模型
    print("加载模型 (Stage A: clean_ipsc)...")
    model_clean = load_ens2_model_robust(clean_model_path)
    print("加载模型 (Stage B: slowkinetics)...")
    model_slow  = load_ens2_model_robust(slow_model_path)

    # 阶段A：clean_ipsc 推理
    print("\n[Stage A] clean_ipsc 推理 ...")
    t0 = time.time()
    det_clean, prob_clean, _ = blob_based_inference(trace, fs, model_clean, polarity=POLARITY, params=params)
    tA = time.time() - t0
    print(f"[Stage A] 完成: {len(det_clean)} 事件，耗时 {tA:.2f}s")

    # 阶段B：slowkinetics 推理
    print("\n[Stage B] slowkinetics 推理 ...")
    t0 = time.time()
    det_slow, prob_slow, _ = blob_based_inference(trace, fs, model_slow, polarity=POLARITY, params=params)
    tB = time.time() - t0
    print(f"[Stage B] 完成: {len(det_slow)} 事件，耗时 {tB:.2f}s")

    # 并集
    det_union = union_events(det_clean, det_slow, fs, merge_within_ms=MERGE_WITHIN_MS)
    print(f"\n[UNION] 并集事件: {len(det_union)} (去重阈值 {MERGE_WITHIN_MS} ms)")

    # 评估
    print("\n=== 事件级评估（与 GT 对齐）===")
    def _eval_and_log(name, det_idx):
        tp, fp, fn = eval_events(det_idx, fs, gt_ms, tol_ms=EVAL_TOL_MS)
        m = compute_metrics(tp, fp, fn)
        print(f"  {name:>12s}  |  TP={tp:4d}  FP={fp:4d}  FN={fn:4d}  "
              f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
        return m

    _eval_and_log("clean_ipsc", det_clean)
    _eval_and_log("slowkinetics", det_slow)
    mU = _eval_and_log("UNION", det_union)
    print("\n=== 偏移诊断（全局时间平移）===")
    for name, det in [("clean_ipsc", det_clean), ("slowkinetics", det_slow), ("UNION", det_union)]:
        off, met = best_time_offset_ms(det, fs, gt_ms, tol_ms=EVAL_TOL_MS, search_ms=80.0, step_ms=0.5)
        print(f"  {name:>12s}  |  best_offset = {off:+.1f} ms  "
            f"→  TP={met['tp']:4d} FP={met['fp']:4d} FN={met['fn']:4d}  "
            f"P={met['precision']:.3f} R={met['recall']:.3f} F1={met['f1']:.3f}")

    # 导出并集预测
    out_csv = Path(DATA_DIR) / f"{TARGET_ID}_pred_union.csv"
    if len(det_union) > 0:
        times_ms = (det_union.astype(float) / float(fs)) * 1000.0
        pd.DataFrame({"Time (ms)": times_ms}).to_csv(out_csv, index=False)
        print(f"\n已保存并集预测到: {out_csv}")

    print("\n=== 最终结果（以并集为准） ===")
    print(f"  Precision={mU['precision']:.3f}  Recall={mU['recall']:.3f}  F1={mU['f1']:.3f}")
    print("\n=== 处理完成 ===")

# ==============================
# 入口
# ==============================

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
