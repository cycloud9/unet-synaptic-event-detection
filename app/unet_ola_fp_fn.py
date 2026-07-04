# -*- coding: utf-8 -*-
"""
U-Net FP提取器 - 概率团检测版
核心功能：加载数据 → U-Net推理 → 概率团检测 → FP识别 → 导出FP_events.csv
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import json
import zipfile
import matplotlib.pyplot as plt

try:
    from scipy.signal import resample_poly, find_peaks
    from scipy.ndimage import gaussian_filter1d
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ===============================
# 核心配置参数
# ===============================

# 模型路径
ENS2_MODEL_PATH = r"C:\Users\Michael\Desktop\lab1\unet_train\outputs\best_ens2_gaussian.keras"

# ENS² 输出修正参数 (自动检测设置)
ENS2_OUTPUTS_LOGIT = None      # 将由load_ens2_model_robust自动设置
ENS2_TEMPERATURE = None        # 从metrics.json读取
METRICS_JSON = r"C:\Users\Michael\Desktop\lab1\unet_train\outputs\metrics.json"

# ENS² OLA 参数 (统一10kHz)
ENS2_TARGET_FS = 10000.0       # 【关键】ENS²统一使用10kHz
ENS2_STRIDE_RATIO = 0.25       # stride = window * 0.25
ENS2_BATCH_SIZE = 256          # 批处理大小

# 概率团检测参数 (替代原来的复杂阈值)
BLOB_THR = 0.20             # 概率团阈值
BLOB_SMOOTH_MS = 0.60       # 高斯平滑窗口（ms）
BLOB_MIN_MS = 0.40          # 最小团长度（ms），太短多半是毛刺
MERGE_WITHIN_MS = 4.0       # 近邻合并阈值（ms）——避免"一峰裂两段"
MERGE_STRICT_MS = 1.0       # 如果同窗内已存在另一个峰，使用更严 1ms 再去重

# 谷对齐参数
VALLEY_LEFT_MS = 0.8           # 左搜索范围
VALLEY_RIGHT_MS = 1.2          # 右搜索范围

# ===============================
# 自定义层注册
# ===============================
@tf.keras.saving.register_keras_serializable()
class Match1DLike(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        x, skip = inputs
        tx = tf.shape(x)[1]
        ts = tf.shape(skip)[1]
        
        def crop_to_length(tensor, target_len):
            current_len = tf.shape(tensor)[1]
            start = tf.maximum(0, (current_len - target_len) // 2)
            return tensor[:, start:start+target_len, :]

        def pad_to_length(tensor, target_len):
            current_len = tf.shape(tensor)[1]
            pad_total = tf.maximum(0, target_len - current_len)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            paddings = tf.stack([[0, 0], [pad_left, pad_right], [0, 0]])
            return tf.pad(tensor, paddings)

        return tf.cond(
            tx > ts,
            lambda: crop_to_length(x, ts),
            lambda: pad_to_length(x, ts)
        )
    
    def get_config(self):
        return super().get_config()

# ===============================
# 模型加载工具
# ===============================

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

def build_clean_ens2_unet(input_len: int = 480, base: int = 32) -> tf.keras.Model:
    """重建与你训练时一致的双头 U-Net（无 Lambda，预设 Match1DLike 进行对齐）"""
    inp = tf.keras.Input(shape=(input_len, 1), name="input_trace")

    # ---- Encoder ----
    def conv_block(x, ch, i):
        x = tf.keras.layers.Conv1D(ch, 3, padding="same", name=f"enc_{i}_conv1")(x)
        x = tf.keras.layers.BatchNormalization(name=f"enc_{i}_bn1")(x)
        x = tf.keras.layers.ReLU(name=f"enc_{i}_relu1")(x)
        x = tf.keras.layers.Conv1D(ch, 3, padding="same", name=f"enc_{i}_conv2")(x)
        x = tf.keras.layers.BatchNormalization(name=f"enc_{i}_bn2")(x)
        x = tf.keras.layers.ReLU(name=f"enc_{i}_relu2")(x)
        x = tf.keras.layers.Dropout(0.0, name=f"enc_{i}_dropout")(x)
        return x

    e1 = conv_block(inp, base, 1)
    p1 = tf.keras.layers.MaxPooling1D(2, name="pool_1")(e1)

    e2 = conv_block(p1, base*2, 2)
    p2 = tf.keras.layers.MaxPooling1D(2, name="pool_2")(e2)

    e3 = conv_block(p2, base*4, 3)

    # ---- Bottleneck ----
    b  = tf.keras.layers.Conv1D(base*8, 3, padding="same", name="bottleneck_conv1")(e3)
    b  = tf.keras.layers.BatchNormalization(name="bottleneck_bn1")(b)
    b  = tf.keras.layers.ReLU(name="bottleneck_relu1")(b)
    b  = tf.keras.layers.Conv1D(base*8, 3, padding="same", name="bottleneck_conv2")(b)
    b  = tf.keras.layers.BatchNormalization(name="bottleneck_bn2")(b)
    b  = tf.keras.layers.ReLU(name="bottleneck_relu2")(b)
    b  = tf.keras.layers.Dropout(0.0, name="bottleneck_dropout")(b)

    # ---- Decoder ----
    d3 = tf.keras.layers.Conv1D(base*4, 3, padding="same", name="dec_3_conv1")(b)
    d3 = tf.keras.layers.BatchNormalization(name="dec_3_bn1")(d3)
    d3 = tf.keras.layers.ReLU(name="dec_3_relu1")(d3)
    d3 = tf.keras.layers.Conv1D(base*4, 3, padding="same", name="dec_3_conv2")(d3)
    d3 = tf.keras.layers.BatchNormalization(name="dec_3_bn2")(d3)
    d3 = tf.keras.layers.ReLU(name="dec_3_relu2")(d3)
    d3 = tf.keras.layers.Dropout(0.0, name="dec_3_dropout")(d3)

    u2 = tf.keras.layers.UpSampling1D(2, name="up_2")(d3)
    c2 = Match1DLike(name="crop_2")([u2, e2])
    cat2 = tf.keras.layers.Concatenate(name="cat_2")([c2, e2])

    d2 = tf.keras.layers.Conv1D(base*2, 3, padding="same", name="dec_2_conv1")(cat2)
    d2 = tf.keras.layers.BatchNormalization(name="dec_2_bn1")(d2)
    d2 = tf.keras.layers.ReLU(name="dec_2_relu1")(d2)
    d2 = tf.keras.layers.Conv1D(base*2, 3, padding="same", name="dec_2_conv2")(d2)
    d2 = tf.keras.layers.BatchNormalization(name="dec_2_bn2")(d2)
    d2 = tf.keras.layers.ReLU(name="dec_2_relu2")(d2)
    d2 = tf.keras.layers.Dropout(0.0, name="dec_2_dropout")(d2)

    u1 = tf.keras.layers.UpSampling1D(2, name="up_1")(d2)
    c1 = Match1DLike(name="crop_1")([u1, e1])
    cat1 = tf.keras.layers.Concatenate(name="cat_1")([c1, e1])

    d1 = tf.keras.layers.Conv1D(base, 3, padding="same", name="dec_1_conv1")(cat1)
    d1 = tf.keras.layers.BatchNormalization(name="dec_1_bn1")(d1)
    d1 = tf.keras.layers.ReLU(name="dec_1_relu1")(d1)
    d1 = tf.keras.layers.Conv1D(base, 3, padding="same", name="dec_1_conv2")(d1)
    d1 = tf.keras.layers.BatchNormalization(name="dec_1_bn2")(d1)
    d1 = tf.keras.layers.ReLU(name="dec_1_relu2")(d1)
    d1 = tf.keras.layers.Dropout(0.0, name="dec_1_dropout")(d1)

    prob = tf.keras.layers.Conv1D(1, 1, name="prob_head")(d1)  # 无激活（线性）
    rate = tf.keras.layers.Conv1D(1, 1, name="rate_head")(d1)

    return tf.keras.Model(inp, [prob, rate], name="epsc_ens2_unet")

def load_ens2_model_robust(path: str):
    """防栈ENS²模型加载：先尝试直接加载，失败则重建架构+按名加载权重"""
    custom_objects = {'Match1DLike': Match1DLike}
    model = None
    activation_lost = False
    
    try:
        print("尝试直接 load_model(..., safe_mode=False)")
        model = tf.keras.models.load_model(path, custom_objects=custom_objects,
                                          compile=False, safe_mode=False)
        print("直接加载成功")
    except Exception as e:
        print(f"直接加载失败：{e}")
        print("回退到重建+按名加载权重模式...")
        T = _read_input_len_from_keras(path, default_len=480)
        print(f"检测到输入长度: {T} 时间点")
        model = build_clean_ens2_unet(T, base=32)
        model.load_weights(path)
        activation_lost = True
        print("权重加载完成，使用重建的无Lambda架构")
    
    # 自动检测prob_head激活函数
    need_sigmoid = activation_lost  # 如果是重建分支，默认需要sigmoid
    
    if not activation_lost:
        try:
            prob_head = next(l for l in model.layers if l.name == "prob_head")
            act_name = getattr(prob_head.activation, "__name__", str(prob_head.activation))
            need_sigmoid = ("linear" in act_name.lower()) or ("none" in act_name.lower())
            print(f"检测到prob_head激活: {act_name}, need_sigmoid={need_sigmoid}")
        except Exception as e:
            print(f"激活检测失败，保守设置need_sigmoid=True: {e}")
            need_sigmoid = True
    
    # 全局设置输出类型标志
    global ENS2_OUTPUTS_LOGIT
    ENS2_OUTPUTS_LOGIT = need_sigmoid
    
    if need_sigmoid:
        print("[INFO] prob_head输出logit，推理时将应用sigmoid变换")
    else:
        print("[INFO] prob_head输出概率，推理时将应用概率域校准")
    
    return model

# ===============================
# 核心工具函数
# ===============================

def sigmoid(x):
    """稳定的sigmoid函数"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def hann_sqrt(W):
    """√Hann 权重：满足COLA条件的窗函数"""
    w = np.hanning(W)
    w = np.sqrt(np.maximum(w, 1e-8))
    return w

def resample_to_target_fs(trace: np.ndarray, fs_src: float, fs_target: float = 10000.0) -> Tuple[np.ndarray, float]:
    """重采样到目标采样率（统一10kHz）"""
    if abs(fs_src - fs_target) < 1e-6:
        return trace, fs_src
    
    if HAS_SCIPY:
        from fractions import Fraction
        ratio = fs_target / fs_src
        frac = Fraction(ratio).limit_denominator(1000)
        up, down = frac.numerator, frac.denominator
        resampled = resample_poly(trace, up, down)
        print(f"   重采样: {fs_src:.1f}Hz -> {fs_target:.1f}Hz (ratio={up}/{down})")
    else:
        t_src = np.arange(len(trace)) / fs_src
        n_target = int(len(trace) * fs_target / fs_src)
        t_target = np.arange(n_target) / fs_target
        resampled = np.interp(t_target, t_src, trace)
        print(f"   重采样: {fs_src:.1f}Hz -> {fs_target:.1f}Hz (线性插值)")
    
    return resampled.astype(np.float32), fs_target

# ===============================
# ENS² 逐点预测模块
# ===============================

def ens2_predict_overlap_add(model, trace, fs, invert=False, 
                             model_outputs_logit=None, temperature=1.0):
    """ENS² 逐点预测 + overlap-add拼接"""
    # 使用全局检测结果
    if model_outputs_logit is None:
        model_outputs_logit = ENS2_OUTPUTS_LOGIT
    
    # 自动推断窗口参数
    W = int(model.input_shape[1])
    stride = max(16, int(W * ENS2_STRIDE_RATIO))
    
    sig = (-trace if invert else trace).astype(np.float32)
    T = len(sig)

    # 反射填充，避免边界窗信息不足
    pad = W // 2
    sig_pad = np.pad(sig, (pad, pad), mode='reflect')
    T_pad = len(sig_pad)

    # 计算窗口起点
    starts = np.arange(0, T_pad - W + 1, stride, dtype=int)
    n_win = len(starts)
    print(f"   ENS² 推理: {n_win}个窗口, 步长={stride}, 窗长={W}")

    # 累加缓存（支持双头输出）
    sum_prob = np.zeros(T_pad, dtype=np.float32)
    sum_rate = np.zeros(T_pad, dtype=np.float32)
    sum_weights = np.zeros(T_pad, dtype=np.float32)
    w = hann_sqrt(W).astype(np.float32)

    # 预处理与训练对齐：去基线→翻转→窗口内min-max
    def _prepare_batch(ii):
        st = starts[ii]
        win = sig_pad[st:st+W].astype(np.float32)
        
        # 训练一致的预处理流程
        # 1. 去基线（前10%中位数）
        baseline_len = max(1, W // 10)
        baseline = float(np.median(win[:baseline_len]))
        
        # 2. 对齐基线（EPSC负向翻转已在sig阶段处理）
        win = win - baseline
        
        # 3. 窗口内Min-Max归一化
        win_min, win_max = np.min(win), np.max(win)
        if win_max > win_min:
            win = (win - win_min) / (win_max - win_min)
        else:
            win = np.zeros_like(win, dtype=np.float32)
            
        return win.reshape(W, 1)

    batch = []
    idxs = []
    
    for i in range(n_win):
        batch.append(_prepare_batch(i))
        idxs.append(i)
        
        if len(batch) == ENS2_BATCH_SIZE or i == n_win - 1:
            X = np.stack(batch, axis=0)

            try:
                y = model.predict(X, verbose=0)
            except Exception as e:
                print(f"模型推理失败，尝试转置输入维度: {e}")
                try:
                    y = model.predict(np.transpose(X, (0, 2, 1)), verbose=0)
                except Exception as e2:
                    print(f"转置后仍然失败: {e2}")
                    raise e2

            # 解析ENS²双头输出
            if isinstance(y, dict):
                prob_out = y.get('prob_head')
                rate_out = y.get('rate_head', np.zeros((len(X), W), np.float32))
            elif isinstance(y, (list, tuple)) and len(y) >= 2:
                prob_out, rate_out = y[0], y[1]
            else:
                prob_out = y
                rate_out = np.zeros((len(X), W), np.float32)

            # 规范化输出形状
            def normalize_output(out):
                out = np.squeeze(out)
                if out.ndim == 1:
                    out = np.tile(out[:, None], (1, W))
                elif out.ndim == 2 and out.shape[1] == 1:
                    out = np.tile(out, (1, W))
                elif out.ndim == 3 and out.shape[-1] == 1:
                    out = out[..., 0]
                return out.astype(np.float32)

            prob_batch = normalize_output(prob_out)
            rate_batch = normalize_output(rate_out)
            
            # 根据输出类型应用正确的校准
            if model_outputs_logit:
                # 标准logit→概率转换 + 温度缩放
                prob_batch = sigmoid(prob_batch / float(temperature))
            else:
                # 概率域温度校准（训练后calibrate_temperature的作用）
                if abs(float(temperature) - 1.0) > 1e-6:
                    eps = 1e-7
                    p = np.clip(prob_batch, eps, 1.0 - eps)
                    logit = np.log(p / (1.0 - p))
                    prob_batch = 1.0 / (1.0 + np.exp(-(logit / float(temperature))))
                    prob_batch = np.clip(prob_batch, 0.0, 1.0)

            # Overlap-add 拼接
            for k, i_win in enumerate(idxs):
                st = starts[i_win]
                if k < len(prob_batch):
                    sum_prob[st:st+W] += w * prob_batch[k]
                    sum_rate[st:st+W] += w * rate_batch[k]
                    sum_weights[st:st+W] += w

            batch.clear()
            idxs.clear()

    # 归一化 & 截去填充
    prob_global = sum_prob / np.maximum(sum_weights, 1e-8)
    rate_global = sum_rate / np.maximum(sum_weights, 1e-8)
    prob_global = prob_global[pad:pad+T]
    rate_global = rate_global[pad:pad+T]
    
    print(f"   拼接完成: prob范围=[{prob_global.min():.4f}, {prob_global.max():.4f}]")
    
    return prob_global, rate_global

# ===============================
# 概率团检测模块（替代复杂阈值逻辑）
# ===============================

def _gauss_sigma_samples(fs, sigma_ms):
    """计算高斯平滑的样本数sigma"""
    return max(0.0, float(sigma_ms) * 1e-3 * float(fs))

def _connected_segments(mask):
    """返回二值数组里各连通片的 [start, end) 索引列表。"""
    if mask.ndim != 1:
        mask = mask.ravel()
    n = len(mask)
    if n == 0:
        return []
    runs = []
    i = 0
    while i < n:
        if mask[i]:
            s = i
            while i < n and mask[i]:
                i += 1
            runs.append((s, i))  # [s, e)
        else:
            i += 1
    return runs

def _segment_center(prob, s, e, use_weighted=True):
    """返回 [s,e) 段的中心索引：加权质心（默认）或最大值索引。"""
    seg = prob[s:e]
    if len(seg) == 0:
        return s
    if use_weighted:
        w = np.clip(seg, 1e-6, 1.0)
        x = np.arange(len(seg), dtype=np.float64)
        c = int(round(s + (x * w).sum() / w.sum()))
        return c
    else:
        return int(s + np.argmax(seg))

def detect_centers_by_blobs(prob, fs,
                            thr_low=0.20,         # 低阈值：生长用
                            thr_high=None,        # 高阈值：选种子峰
                            smooth_ms=0.80,       # 稍强一点的平滑
                            min_len_ms=0.60,      # 至少 0.6 ms
                            min_prom=0.08,        # 概率峰显著性（prominence）
                            min_dist_ms=2.0,      # 种子峰最小间距
                            merge_within_ms=4.0,  # 4 ms 合并
                            strict_within_ms=1.0, # 1 ms 二次去重
                            trace=None):          # 可选：用于方向约束
    """
    高阈值找"可靠种子峰" → 从种子向两侧生长到低阈值形成 blob → 在 blob 内选中心。
    中心打分综合：概率 × 负向一阶导幅度（可选），避免把"向上/平坡"当 EPSC。
    
    参数:
    - prob: OLA后的一维概率序列
    - fs: 采样率
    - thr_low: 低阈值，用于blob生长
    - thr_high: 高阈值，用于种子峰选取（None则自动用95分位）
    - smooth_ms: 高斯平滑窗口(ms)
    - min_len_ms: 最小团长度(ms)
    - min_prom: 峰显著性阈值
    - min_dist_ms: 种子峰最小间距(ms)
    - merge_within_ms: 近邻合并阈值(ms)
    - strict_within_ms: 严格去重阈值(ms)
    - trace: 原始信号，用于方向约束
    
    返回:
    - centers: 事件中心的样本索引数组
    """
    p = np.asarray(prob, dtype=np.float32)
    
    # 1) 平滑概率
    if HAS_SCIPY and smooth_ms and smooth_ms > 0:
        sigma = max(0.5, float(smooth_ms) * 1e-3 * fs)
        p_s = gaussian_filter1d(p, sigma=sigma)
        print(f"   应用高斯平滑: sigma={sigma:.1f} samples")
    else:
        p_s = p

    # 2) 动态高阈值（百分位防台阶）：缺省用 95 分位与 0.45 取大
    valid = p_s[p_s >= thr_low]
    if valid.size >= int(0.01 * len(p_s)):
        ph = float(np.percentile(valid, 90))   # 比 95% 更稳一点
    else:
        ph = float(np.percentile(p_s, 95))

    thr_high = max(thr_low + 0.06, min(0.9, ph))  # 上限 0.9，且至少高于低阈值 0.06
    print(f"   高低阈值滞回: 低={thr_low:.2f}, 高={thr_high:.2f}, prominence>={min_prom:.2f}")

    # 3) 找"可靠种子峰"
    min_dist = max(1, int(round(min_dist_ms * 1e-3 * fs)))
    if HAS_SCIPY:
        peaks, props = find_peaks(p_s, height=thr_high, prominence=min_prom, distance=min_dist)
    else:
        # 简化版峰检测
        peaks = []
        for i in range(1, len(p_s) - 1):
            if (p_s[i] > p_s[i-1] and p_s[i] >= p_s[i+1] and 
                p_s[i] >= thr_high and 
                p_s[i] - max(p_s[i-1], p_s[i+1]) >= min_prom):
                peaks.append(i)
        peaks = np.array(peaks)

    print(f"   发现 {len(peaks)} 个种子峰（高阈值+显著性）")

    thr_low = float(thr_low)
    n = len(p_s)

    if len(peaks) == 0:
        # 兜底：直接用低阈值连通片作为 blob
        mask = p_s >= thr_low
        blobs = [(l, r) for (l, r) in _connected_segments(mask)
                if (r - l) >= int(round(min_len_ms * 1e-3 * fs))]
        print(f"   无种子 → 兜底连通片 {len(blobs)} 个")
    else:
        # 从每个种子向两侧生长到低阈值，得到 blob（原有逻辑）
        blobs = []
        for pk in peaks:
            # 左生长
            l = pk
            while l > 0 and p_s[l] >= thr_low:
                l -= 1
            # 右生长
            r = pk
            while r < n-1 and p_s[r] >= thr_low:
                r += 1
            # 最小长度过滤
            if (r - l) >= int(round(min_len_ms * 1e-3 * fs)):
                blobs.append((max(0, l+1), min(n, r)))
        print(f"   生长形成 {len(blobs)} 个有效 blob")

    if not blobs:
        return np.array([], dtype=int)


    # 5) 在每个 blob 内选中心：概率 × 方向因子（可选）
    # 方向因子：如果提供了 trace，就用 -d/dt 的正部分 (只奖励向下)
    score = p_s.copy()
    if trace is not None:
        tr = np.asarray(trace, dtype=np.float32)
        # 对 trace 做轻度平滑，防止噪声导数过大
        if HAS_SCIPY:
            tr_s = gaussian_filter1d(tr, sigma=max(0.5, 0.40e-3*fs))
        else:
            tr_s = tr
        dz = np.gradient(tr_s)  # 一阶导
        dir_gain = np.maximum(0.0, dz)  # 只保留"向上"
        # 归一化到 [0,1] 后作为增益
        if dir_gain.max() > 0:
            dir_gain = dir_gain / (dir_gain.max() + 1e-8)
            score = score * (0.6 + 0.4 * dir_gain)  # 方向最多再加 40% 权
            print(f"   应用方向约束: EPSC向下导数加权")

    centers = []
    for l, r in blobs:
        seg = score[l:r]
        if seg.size == 0:
            continue
        c = int(l + np.argmax(seg))  # 用打分的峰位置作为中心
        centers.append(c)

    if not centers:
        return np.array([], dtype=int)

    centers = np.sort(np.unique(np.asarray(centers, dtype=int)))

    # 6) 4 ms 合并：保留 score 更大的
    gap = int(round(merge_within_ms * 1e-3 * fs))
    kept = [centers[0]]
    for t in centers[1:]:
        if (t - kept[-1]) <= max(1, gap):
            if score[t] > score[kept[-1]]:
                kept[-1] = t
        else:
            kept.append(t)
    centers = np.array(kept, dtype=int)
    print(f"   {merge_within_ms:.1f}ms内合并: {len(centers)} 个中心")

    # 7) 1 ms 严格去重
    if strict_within_ms and len(centers) > 1:
        gap2 = int(round(strict_within_ms * 1e-3 * fs))
        kept = [centers[0]]
        for t in centers[1:]:
            if (t - kept[-1]) <= max(1, gap2):
                if score[t] > score[kept[-1]]:
                    kept[-1] = t
            else:
                kept.append(t)
        centers = np.array(kept, dtype=int)
        print(f"   {strict_within_ms:.1f}ms严格去重: {len(centers)} 个最终中心")

    return centers

# ===============================
# 后处理模块
# ===============================

def refine_extrema_alignment(trace, events, fs, left_ms=0.8, right_ms=1.2, mode="peak"):
    """在事件附近对齐到极值：mode='peak' 取最大值，'valley' 取最小值"""
    if len(events) == 0:
        return np.array([], dtype=int)

    L = int(left_ms * 1e-3 * fs)
    R = int(right_ms * 1e-3 * fs)
    n = len(trace)
    refined = []
    for center in events:
        s = max(0, center - L)
        e = min(n, center + R + 1)
        if e <= s:
            refined.append(center)
            continue
        seg = trace[s:e]
        if mode == "valley":
            rel = int(np.argmin(seg))
        else:  # 'peak'
            rel = int(np.argmax(seg))
        refined.append(s + rel)
    return np.array(refined, dtype=int)


# ===============================
# 数据加载函数
# ===============================

def load_trace_from_csv(csv_path: Path) -> Tuple[np.ndarray, float]:
    """从CSV加载trace数据，返回真实采样率"""
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = cols.get("time_s") or cols.get("time") or cols.get("time(s)") or df.columns[0]
    ycol = cols.get("current_pa") or cols.get("current") or cols.get("i_pa") or df.columns[1]
    
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(np.float64)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(np.float64)
    
    # 去除NaN
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    
    # 推断原始采样率
    dt = np.median(np.diff(t))
    fs_orig = 1.0 / dt
    
    return y, fs_orig

def load_events_from_csv(csv_path: Path) -> List[Dict]:
    """从CSV加载事件标注"""
    if not csv_path.exists():
        return []
    
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = cols.get("time (ms)") or cols.get("time_ms") or cols.get("time") or cols.get("time (s)") or df.columns[0]
    
    events = []
    time_values = []
    
    # 收集时间值用于单位判断
    for _, r in df.iterrows():
        try:
            tval = float(str(r[tcol]).replace(",", ""))
            time_values.append(tval)
        except Exception:
            continue
    
    if not time_values:
        return []
    
    # 单位识别：优先看列名，再看数值范围
    is_ms_unit = False
    col_name_lower = str(tcol).lower()
    if "ms" in col_name_lower:
        is_ms_unit = True
    elif "s)" in col_name_lower and "ms" not in col_name_lower:
        is_ms_unit = False
    else:
        max_time = max(time_values)
        is_ms_unit = max_time > 100
    
    # 处理事件
    for _, r in df.iterrows():
        try:
            tval = float(str(r[tcol]).replace(",", ""))
            if not is_ms_unit:
                tval *= 1000.0
            events.append({"time_ms": tval})
        except Exception:
            continue
    
    events.sort(key=lambda x: x["time_ms"])
    return events

# ===============================
# 事件评估函数
# ===============================

def eval_events(detected_idx, fs, gt_ms, tol_ms=1.2):
    """事件匹配评估"""
    pred_s = np.asarray(detected_idx, dtype=float) / float(fs)
    gt_s = np.asarray(gt_ms, dtype=float) / 1000.0
    pred_s.sort(); gt_s.sort()
    tol_s = float(tol_ms) / 1000.0

    # 贪心一一匹配
    tp = 0; i = j = 0
    used_pred = np.zeros(len(pred_s), dtype=bool)
    while i < len(gt_s) and j < len(pred_s):
        if abs(pred_s[j] - gt_s[i]) <= tol_s and not used_pred[j]:
            tp += 1
            used_pred[j] = True
            i += 1; j += 1
        elif pred_s[j] < gt_s[i] - tol_s:
            j += 1
        else:
            i += 1

    fp = len(pred_s) - int(used_pred.sum())
    fn = len(gt_s) - tp
    
    return tp, fp, fn

def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """计算分类指标"""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

# ===============================
# FP识别与导出
# ===============================

def identify_fp_events(detected_idx, fs, gt_ms, tol_ms=1.2):
    """识别假阳性事件，返回FP事件的样本索引"""
    pred_s = np.asarray(detected_idx, dtype=float) / float(fs)
    gt_s = np.asarray(gt_ms, dtype=float) / 1000.0
    pred_s_sorted_idx = np.argsort(pred_s)
    gt_s_sorted = np.sort(gt_s)
    pred_s_sorted = pred_s[pred_s_sorted_idx]
    tol_s = float(tol_ms) / 1000.0

    # 贪心匹配找到TP
    tp_mask = np.zeros(len(pred_s_sorted), dtype=bool)
    i = j = 0
    while i < len(gt_s_sorted) and j < len(pred_s_sorted):
        if abs(pred_s_sorted[j] - gt_s_sorted[i]) <= tol_s and not tp_mask[j]:
            tp_mask[j] = True
            i += 1; j += 1
        elif pred_s_sorted[j] < gt_s_sorted[i] - tol_s:
            j += 1
        else:
            i += 1

    # FP是未匹配的预测事件
    fp_mask_sorted = ~tp_mask
    fp_indices_sorted = pred_s_sorted_idx[fp_mask_sorted]
    fp_sample_indices = detected_idx[fp_indices_sorted]
    
    return fp_sample_indices

def save_fp_events_csv(target_id: str, trace: np.ndarray, fs: float, 
                       fp_sample_indices: np.ndarray, out_dir: str = "."):
    """保存FP事件到CSV文件"""
    if len(fp_sample_indices) == 0:
        print(f"   无FP事件，跳过CSV导出")
        return None
    
    # 准备数据
    fp_data = []
    for idx, sample_idx in enumerate(fp_sample_indices):
        sample_idx = int(sample_idx)
        time_ms = sample_idx / fs * 1000.0
        amplitude = float(trace[sample_idx]) if sample_idx < len(trace) else np.nan
        
        fp_data.append({
            "fp_index": idx + 1,
            "sample_index": sample_idx,
            "time_ms": time_ms,
            "amplitude_pa": amplitude,
            "target_id": target_id,
            "fs_hz": fs
        })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(fp_data)
    out_path = Path(out_dir) / f"{target_id}_FP_events.csv"
    df.to_csv(out_path, index=False)
    
    print(f"   FP事件已保存: {out_path} ({len(fp_data)}个FP)")
    return out_path

# ===============================
# FN识别与导出
# ===============================

def identify_fn_events(detected_idx, fs, gt_ms, tol_ms=1.2):
    """识别假阴性事件，返回未被检测到的GT事件时间(ms)"""
    pred_s = np.asarray(detected_idx, dtype=float) / float(fs)
    gt_s = np.asarray(gt_ms, dtype=float) / 1000.0
    pred_s_sorted = np.sort(pred_s)
    gt_s_sorted = np.sort(gt_s)
    tol_s = float(tol_ms) / 1000.0

    # 贪心匹配找到TP
    fn_mask = np.ones(len(gt_s_sorted), dtype=bool)  # 初始都是FN
    i = j = 0
    while i < len(gt_s_sorted) and j < len(pred_s_sorted):
        if abs(pred_s_sorted[j] - gt_s_sorted[i]) <= tol_s:
            fn_mask[i] = False  # 匹配成功，不是FN
            i += 1; j += 1
        elif pred_s_sorted[j] < gt_s_sorted[i] - tol_s:
            j += 1
        else:
            i += 1

    # 返回未匹配的GT事件时间(ms)
    fn_gt_times_ms = np.array(gt_ms)[fn_mask]
    return fn_gt_times_ms

def save_fn_events_csv(target_id: str, fn_gt_times_ms: np.ndarray, fs: float, out_dir: str = "."):
    """保存FN事件到CSV文件"""
    if len(fn_gt_times_ms) == 0:
        print(f"   无FN事件，跳过CSV导出")
        return None
    
    # 准备数据
    fn_data = []
    for idx, gt_time_ms in enumerate(fn_gt_times_ms):
        sample_idx = int(gt_time_ms * 1e-3 * fs)  # ms -> samples
        
        fn_data.append({
            "fn_index": idx + 1,
            "gt_time_ms": gt_time_ms,
            "gt_sample_index": sample_idx,
            "target_id": target_id,
            "fs_hz": fs
        })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(fn_data)
    out_path = Path(out_dir) / f"{target_id}_FN_events.csv"
    df.to_csv(out_path, index=False)
    
    print(f"   FN事件已保存: {out_path} ({len(fn_data)}个FN)")
    return out_path

# ===============================
# 可视化模块（保持原有双轴设计）
# ===============================

def plot_fp_gallery(trace, prob, fp_sample_indices, fs, target_id, 
                   window_ms=30.0, ncols=4, out_dir="."):
    """
    绘制FP事件画廊：每个FP前后30ms的窗口，叠加UNet概率
    """
    if len(fp_sample_indices) == 0:
        print("   无FP事件，跳过画廊绘制")
        return None
    
    # 计算窗口参数
    window_samples = int(window_ms * 1e-3 * fs)  # 30ms = 300 samples @ 10kHz
    half_window = window_samples // 2
    
    # 计算子图布局
    n_fps = len(fp_sample_indices)
    nrows = (n_fps + ncols - 1) // ncols  # 向上取整
    
    # 创建图形 - 调整尺寸为双y轴留出空间
    fig_width = ncols * 4.5  # 增加宽度
    fig_height = nrows * 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    fig.suptitle(f'FP Events Gallery - {target_id} ({n_fps} FPs)', fontsize=16, fontweight='bold')
    
    # 确保axes是2D数组
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # 绘制每个FP事件
    for i, fp_idx in enumerate(fp_sample_indices):
        row = i // ncols
        col = i % ncols
        ax1 = axes[row, col]  # 主坐标轴(信号)
        
        fp_idx = int(fp_idx)
        fp_time_ms = fp_idx / fs * 1000.0
        
        # 计算窗口范围
        start_idx = max(0, fp_idx - half_window)
        end_idx = min(len(trace), fp_idx + half_window)
        
        # 提取窗口数据
        window_trace = trace[start_idx:end_idx]
        window_prob = prob[start_idx:end_idx]
        
        # 创建时间轴(相对于FP位置的时间)
        n_samples = len(window_trace)
        relative_start = (start_idx - fp_idx) / fs * 1000.0  # 相对时间(ms)
        time_axis = np.linspace(relative_start, 
                               relative_start + (n_samples - 1) / fs * 1000.0, 
                               n_samples)
        
        # 绘制原始信号(主坐标轴)
        line1 = ax1.plot(time_axis, window_trace, 'k-', linewidth=0.8, alpha=0.7, label='Raw trace')
        
        # 标记FP位置
        fp_relative_time = 0.0  # FP在窗口中心
        fp_amplitude = window_trace[len(window_trace)//2] if len(window_trace) > 0 else 0
        ax1.axvline(x=fp_relative_time, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        fp_point = ax1.plot(fp_relative_time, fp_amplitude, 'ro', markersize=8, markerfacecolor='red', 
                           markeredgecolor='darkred', markeredgewidth=1.5, label='FP peak')
        
        # 创建右侧y轴用于概率
        ax2 = ax1.twinx()
        line2 = ax2.plot(time_axis, window_prob, 'r-', linewidth=1.2, alpha=0.8, label='UNet prob')
        
        # 设置坐标轴标签和范围
        ax1.set_xlabel('Relative Time (ms)', fontsize=8)
        ax1.set_ylabel('Amplitude (pA)', fontsize=8, color='black')
        ax2.set_ylabel('Probability', fontsize=8, color='red')
        
        # 设置概率轴范围和颜色
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor='red', labelsize=7)
        ax1.tick_params(axis='y', labelcolor='black', labelsize=8)
        ax1.tick_params(axis='x', labelsize=8)
        
        # 设置x轴范围
        ax1.set_xlim([-window_ms/2, window_ms/2])
        
        # 设置标题
        ax1.set_title(f'FP #{i+1}\nTime: {fp_time_ms:.1f} ms', fontsize=10, fontweight='bold')
        
        # 网格
        ax1.grid(True, alpha=0.3)
        
        # 只在第一个子图显示图例
        if i == 0:
            # 合并两个坐标轴的图例
            lines = line1 + fp_point + line2
            labels = ['Raw trace', 'FP peak', 'UNet prob']
            ax1.legend(lines, labels, fontsize=8, loc='upper left')
    
    # 隐藏多余的子图
    for i in range(n_fps, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # 为主标题留空间
    
    # 保存图像
    out_path = Path(out_dir) / f"{target_id}_FP_gallery.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"   FP画廊已保存: {out_path}")
    return out_path

def plot_fn_gallery(trace, prob, fn_gt_times_ms, fs, target_id, 
                   window_ms=30.0, ncols=4, out_dir="."):
    """
    绘制FN事件画廊：每个FN的GT位置前后30ms的窗口，叠加UNet概率
    """
    if len(fn_gt_times_ms) == 0:
        print("   无FN事件，跳过画廊绘制")
        return None
    
    # 计算窗口参数
    window_samples = int(window_ms * 1e-3 * fs)  # 30ms = 300 samples @ 10kHz
    half_window = window_samples // 2
    
    # 计算子图布局
    n_fns = len(fn_gt_times_ms)
    nrows = (n_fns + ncols - 1) // ncols  # 向上取整
    
    # 创建图形 - 调整尺寸为双y轴留出空间
    fig_width = ncols * 4.5  # 增加宽度
    fig_height = nrows * 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    fig.suptitle(f'FN Events Gallery - {target_id} ({n_fns} FNs)', fontsize=16, fontweight='bold')
    
    # 确保axes是2D数组
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # 绘制每个FN事件
    for i, fn_time_ms in enumerate(fn_gt_times_ms):
        row = i // ncols
        col = i % ncols
        ax1 = axes[row, col]  # 主坐标轴(信号)
        
        # 将GT时间(ms)转换为样本索引
        fn_idx = int(fn_time_ms * 1e-3 * fs)  # ms -> s -> samples
        
        # 计算窗口范围
        start_idx = max(0, fn_idx - half_window)
        end_idx = min(len(trace), fn_idx + half_window)
        
        # 提取窗口数据
        window_trace = trace[start_idx:end_idx]
        window_prob = prob[start_idx:end_idx]
        
        # 创建时间轴(相对于FN位置的时间)
        n_samples = len(window_trace)
        relative_start = (start_idx - fn_idx) / fs * 1000.0  # 相对时间(ms)
        time_axis = np.linspace(relative_start, 
                               relative_start + (n_samples - 1) / fs * 1000.0, 
                               n_samples)
        
        # 绘制原始信号(主坐标轴)
        line1 = ax1.plot(time_axis, window_trace, 'k-', linewidth=0.8, alpha=0.7, label='Raw trace')
        
        # 标记GT位置(FN应该被检测但未被检测的位置)
        gt_relative_time = 0.0  # GT在窗口中心
        # 找到GT位置对应的信号幅度
        center_idx = len(window_trace) // 2
        gt_amplitude = window_trace[center_idx] if len(window_trace) > 0 else 0
        
        ax1.axvline(x=gt_relative_time, color='orange', linestyle='--', alpha=0.8, linewidth=1.5)
        gt_point = ax1.plot(gt_relative_time, gt_amplitude, 'o', color='orange', markersize=8, 
                           markerfacecolor='orange', markeredgecolor='darkorange', 
                           markeredgewidth=1.5, label='GT (missed)')
        
        # 创建右侧y轴用于概率
        ax2 = ax1.twinx()
        line2 = ax2.plot(time_axis, window_prob, 'b-', linewidth=1.2, alpha=0.8, label='UNet prob')
        
        # 设置坐标轴标签和范围
        ax1.set_xlabel('Relative Time (ms)', fontsize=8)
        ax1.set_ylabel('Amplitude (pA)', fontsize=8, color='black')
        ax2.set_ylabel('Probability', fontsize=8, color='blue')
        
        # 设置概率轴范围和颜色
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='y', labelcolor='blue', labelsize=7)
        ax1.tick_params(axis='y', labelcolor='black', labelsize=8)
        ax1.tick_params(axis='x', labelsize=8)
        
        # 设置x轴范围
        ax1.set_xlim([-window_ms/2, window_ms/2])
        
        # 设置标题
        ax1.set_title(f'FN #{i+1}\nGT Time: {fn_time_ms:.1f} ms', fontsize=10, fontweight='bold')
        
        # 网格
        ax1.grid(True, alpha=0.3)
        
        # 只在第一个子图显示图例
        if i == 0:
            # 合并两个坐标轴的图例
            lines = line1 + gt_point + line2
            labels = ['Raw trace', 'GT (missed)', 'UNet prob']
            ax1.legend(lines, labels, fontsize=8, loc='upper left')
    
    # 隐藏多余的子图
    for i in range(n_fns, nrows * ncols):
        row = i // ncols
        col = i % ncols
        axes[row, col].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # 为主标题留空间
    
    # 保存图像
    out_path = Path(out_dir) / f"{target_id}_FN_gallery.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"   FN画廊已保存: {out_path}")
    return out_path

# ===============================
# 主推理函数（概率团版本）
# ===============================

def blob_based_inference(trace, fs, ens2_model):
    """
    基于概率团的推理主函数：ENS² 逐点预测 + 概率团中心检测 + 后处理
    """
    
    # 阶段1：ENS² 逐点预测
    print("阶段1: ENS² 逐点概率预测...")
    prob, rate = ens2_predict_overlap_add(ens2_model, trace, fs, 
                                        invert=False, 
                                        model_outputs_logit=ENS2_OUTPUTS_LOGIT,
                                        temperature=ENS2_TEMPERATURE)
    
    # 阶段2：概率团中心检测
    print("阶段2: 概率团中心检测...")
    detected_events = detect_centers_by_blobs(
        prob, fs,
        thr_low=0.20,
        thr_high=None,       # 动态 95 分位
        smooth_ms=0.80,
        min_len_ms=0.60,
        min_prom=0.08,
        min_dist_ms=2.0,
        merge_within_ms=4.0,
        strict_within_ms=1.0,
        trace=trace          # ✅ 方向约束需要原 trace
    )
    
    # 阶段3：后处理（保留谷对齐）
    print("阶段3: 后处理优化...")
    if len(detected_events) > 0:
        # 保留你现有的后处理对齐
        detected_events = refine_extrema_alignment(trace, detected_events, fs, 
                                                 VALLEY_LEFT_MS, VALLEY_RIGHT_MS,
                                           mode="peak")
    
    print(f"   最终结果: {len(detected_events)}事件")
    
    return detected_events, prob, rate

# ===============================
# 温度参数读取
# ===============================

def _resolve_temperature(metrics_json=None, default=1.0):
    """从metrics.json读取温度，失败则用默认值"""
    if metrics_json and Path(metrics_json).exists():
        try:
            with open(metrics_json, "r", encoding="utf-8") as f:
                m = json.load(f)
            t = float(m.get("temperature", default))
            print(f"[CONF] 从 {metrics_json} 读取温度: T={t:.3f}")
            return t
        except Exception as e:
           print(f"[WARN] 读取 metrics.json 失败: {e}")
    if ENS2_TEMPERATURE is not None:
        return float(ENS2_TEMPERATURE)
    print(f"[CONF] 使用默认温度: T={default:.3f}")
    return default

# ===============================
# 主函数
# ===============================

def main():
    # 配置
    DATA_DIR = r"C:\Users\Michael\Desktop\lab1\EPSCdataset"
    TARGET_ID = "18o24000"
    MATCH_TOL_MS = 4.0
    
    global ENS2_TEMPERATURE
    ENS2_TEMPERATURE = _resolve_temperature(METRICS_JSON)
    
    print("=== U-Net FP提取器 - 概率团检测版 ===")
    print("核心功能：加载数据 → U-Net推理 → 概率团检测 → FP识别 → 导出FP_events.csv")
    
    # 加载模型
    print("\n加载模型...")
    ens2_model = load_ens2_model_robust(ENS2_MODEL_PATH)
    print(f"ENS²模型加载成功: {ens2_model.count_params():,} 参数")
    print(f"自动检测输出类型: ENS2_OUTPUTS_LOGIT = {ENS2_OUTPUTS_LOGIT}")
    
    # 显示使用的检测参数
    print(f"\n概率团检测配置 (高低阈值滞回版):")
    print(f"  低阈值 (生长): 0.20")
    print(f"  高阈值 (种子): 自动95分位")
    print(f"  平滑窗口: 0.80ms")
    print(f"  最小团长: 0.60ms")
    print(f"  峰显著性: 0.08")
    print(f"  种子间距: 2.0ms")
    print(f"  合并容差: 4.0ms")
    print(f"  统一采样率: {ENS2_TARGET_FS}Hz")
    
    # 加载数据
    print(f"\n加载数据: {TARGET_ID}")
    trace_csv = Path(DATA_DIR) / f"{TARGET_ID}_trace.csv"
    events_csv = Path(DATA_DIR) / f"{TARGET_ID}_events.csv"
    
    if (not trace_csv.exists()) or (not events_csv.exists()):
        print("数据文件不存在:")
        print(f"   Trace: {trace_csv}")
        print(f"   Events: {events_csv}")
        return

    try:
        # 加载原始数据
        raw_trace, fs_original = load_trace_from_csv(trace_csv)
        events = load_events_from_csv(events_csv)
        
        print(f"原始数据概况:")
        print(f"   原始Trace: {len(raw_trace)/fs_original:.1f}s @ {fs_original:.1f}Hz")
        print(f"   GT事件: {len(events)} 个")
        
        # 统一重采样到10kHz（ENS²使用）
        print(f"\n重采样到统一采样率...")
        trace, fs = resample_to_target_fs(raw_trace, fs_original, ENS2_TARGET_FS)
        print(f"   重采样后: {len(trace)/fs:.1f}s @ {fs:.1f}Hz")
        
        # 测试范围限制（可选）
        test_duration_s = 300.0 if len(trace) > 30 * ENS2_TARGET_FS else len(trace) / fs
        test_samples = int(test_duration_s * fs)
        if len(trace) > test_samples:
            trace = trace[:test_samples]
            print(f"   限制到前{test_duration_s}s进行测试")
        
        # 筛选测试范围内的GT事件
        test_duration_ms = test_duration_s * 1000
        events_in_test = [e for e in events if e['time_ms'] <= test_duration_ms]
        true_events_ms = [e["time_ms"] for e in events_in_test]
        
        print(f"测试范围内GT事件: {len(true_events_ms)} 个")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 运行概率团推理
    print(f"\n开始概率团推理...")
    start_time = time.time()
    
    detected_events, prob, rate = blob_based_inference(trace, fs, ens2_model)
    
    total_time = time.time() - start_time
    print(f"\n推理完成，总耗时: {total_time:.2f}s")
    print(f"检测结果: {len(detected_events)}个事件")
    
    # 事件评估
    print(f"\n事件级评估:")
    tp, fp, fn = eval_events(detected_events, fs, true_events_ms, MATCH_TOL_MS)
    metrics = compute_metrics(tp, fp, fn)
    
    print(f"   匹配容差: {MATCH_TOL_MS} ms")
    print(f"   TP={tp}, FP={fp}, FN={fn}")
    print(f"   精确率: {metrics['precision']:.3f}")
    print(f"   召回率: {metrics['recall']:.3f}")
    print(f"   F1分数: {metrics['f1']:.3f}")
    
    # 识别并导出FP事件
    print(f"\n识别并导出FP事件...")
    fp_sample_indices = identify_fp_events(detected_events, fs, true_events_ms, MATCH_TOL_MS)
if __name__ == "__main__":
    main()