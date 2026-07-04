#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理 Brain organoids 下的 abf 文件：
1）把 abf 转成 xxx_trace.csv
2）为每个 abf 创建一个空的 xxx_events.csv（如果还不存在）
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyabf

# 改成你的根目录
ROOT = Path(r"D:\Yue Shu\EPSC&IPSC\Brain organoids")


def convert_abf_to_trace_csv(abf_path: Path) -> Path:
    """
    读取单个 abf，导出为 {stem}_trace.csv，包含两列：
    time_s, current_pA
    如果有多个 sweep，就简单按时间拼在一起。
    """
    abf = pyabf.ABF(str(abf_path))

    if abf.sweepCount > 1:
        # 多 sweep：全部拼成一个连续 trace
        ys = []
        for i in range(abf.sweepCount):
            abf.setSweep(i)
            ys.append(abf.sweepY.copy())
        current_pA = np.concatenate(ys)
        dt = 1.0 / abf.dataRate
        time_s = np.arange(current_pA.size) * dt
    else:
        # 单 sweep：直接用 sweepX / sweepY
        abf.setSweep(0)
        time_s = abf.sweepX.copy()      # 秒
        current_pA = abf.sweepY.copy()  # pA

    df = pd.DataFrame({
        "time_s": time_s,
        "current_pA": current_pA,
    })

    out_csv = abf_path.with_name(f"{abf_path.stem}_trace.csv")
    df.to_csv(out_csv, index=False)
    return out_csv


def ensure_empty_events_csv(abf_path: Path) -> Path:
    """
    为 abf 创建一个空的 {stem}_events.csv（只含表头），如果已存在则跳过。
    表头用 Time (ms)，后续 evt 导出的时间直接粘进去就行。
    """
    events_csv = abf_path.with_name(f"{abf_path.stem}_events.csv")
    if events_csv.exists():
        print(f"  已存在 events CSV，跳过：{events_csv.name}")
        return events_csv

    # 只要一个 Time (ms) 列就够了，后面 evt2asc.py 也是用这列
    empty_df = pd.DataFrame(columns=["Time (ms)"])
    empty_df.to_csv(events_csv, index=False)
    print(f"  已创建空 events CSV：{events_csv.name}")
    return events_csv


def main():
    print(f"遍历根目录：{ROOT}")
    abf_files = list(ROOT.rglob("*.abf"))
    print(f"共找到 {len(abf_files)} 个 abf 文件。\n")

    for abf_path in abf_files:
        print(f"处理 {abf_path}")
        # 1) abf -> trace.csv（如果已经有可以加一个存在就跳过的判断）
        trace_csv = abf_path.with_name(f"{abf_path.stem}_trace.csv")
        if trace_csv.exists():
            print(f"  已存在 trace CSV，跳过转换：{trace_csv.name}")
        else:
            trace_csv = convert_abf_to_trace_csv(abf_path)
            print(f"  已生成 trace CSV：{trace_csv.name}")

        # 2) 创建空 events.csv
        ensure_empty_events_csv(abf_path)

    print("\n全部完成 ✅")


if __name__ == "__main__":
    main()