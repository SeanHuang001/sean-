"""
价差诊断：画图、K 线连续性、回测开仓密度、平仓明细汇总。
运行：python diagnose_jump.py
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtest_xau_xaut_grid import load_and_merge

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "report"

XAU_PATH = DATA_DIR / "XAUUSDT_1m.csv"
XAUT_PATH = DATA_DIR / "XAUTUSDT_1m.csv"

# 图表区间（含首尾日）
PLOT_START = pd.Timestamp("2026-01-10", tz="UTC")
PLOT_END = pd.Timestamp("2026-01-21", tz="UTC")  # 不含 21 日 0 点 → 含 1 月 20 日全天

# 缺口检查区间（1 月 13–17 日，UTC 日历日）
GAP_DAY_START = date(2026, 1, 13)
GAP_DAY_END = date(2026, 1, 17)

# 成交明细开仓密度区间
TRADE_START = pd.Timestamp("2026-01-13", tz="UTC")
TRADE_END = pd.Timestamp("2026-01-18", tz="UTC")  # 不含 18 日 0 点 → 含 17 日全天

# 第 4 项：按 close_datetime 筛选 [2026-01-15 00:00 UTC, 2026-01-17 00:00 UTC)
CLOSE_WIN_START = pd.Timestamp("2026-01-15", tz="UTC")
CLOSE_WIN_END = pd.Timestamp("2026-01-17", tz="UTC")

GAP_THRESHOLD_MS = 120_000


def latest_trade_details_path() -> Path | None:
    if not REPORT_DIR.is_dir():
        return None
    subdirs = [p for p in REPORT_DIR.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    p = latest / "trade_details.csv"
    return p if p.is_file() else None


def _read_sorted_open_time_ms(path: Path) -> pd.Series:
    raw = pd.read_csv(path)
    if "open_time_ms" in raw.columns:
        ts_col, ts_unit = "open_time_ms", "ms"
    elif "timestamp" in raw.columns:
        ts_col, ts_unit = "timestamp", "s"
    elif "open_time" in raw.columns:
        ts_col, ts_unit = "open_time", "s"
    else:
        raise ValueError(f"找不到时间戳列：{path} {list(raw.columns)}")
    s = pd.to_numeric(raw[ts_col], errors="coerce")
    if ts_unit == "s":
        mx = s.max(skipna=True)
        if mx is not None and mx < 1e11:
            s = s * 1000.0
    s = s.dropna().astype("int64").sort_values().reset_index(drop=True)
    return s


def _dt_utc(ms: int) -> pd.Timestamp:
    return pd.to_datetime(ms, unit="ms", utc=True)


def _gap_in_calendar_window(ms: int) -> bool:
    d = _dt_utc(int(ms)).date()
    return GAP_DAY_START <= d <= GAP_DAY_END


def print_gaps_for_symbol(label: str, path: Path) -> None:
    if not path.is_file():
        print(f"[{label}] 文件不存在: {path}")
        return
    s = _read_sorted_open_time_ms(path)
    if len(s) < 2:
        print(f"[{label}] 数据行数不足，跳过缺口检查。")
        return
    diffs = s.diff().iloc[1:]
    bad_idx = diffs[diffs > GAP_THRESHOLD_MS].index
    any_gap = False
    for i in bad_idx:
        prev_ms = int(s.iloc[i - 1])
        next_ms = int(s.iloc[i])
        if not _gap_in_calendar_window(prev_ms):
            continue
        any_gap = True
        t0 = _dt_utc(prev_ms)
        t1 = _dt_utc(next_ms)
        delta_min = (next_ms - prev_ms) / 60_000.0
        t0s = t0.strftime("%Y-%m-%d %H:%M")
        t1s = t1.strftime("%Y-%m-%d %H:%M")
        print(f"[{label}] 缺口: {t0s} -> {t1s} (差{delta_min:.0f}分钟)")
    if not any_gap:
        print(f"[{label}] 在 {GAP_DAY_START}～{GAP_DAY_END} 内未发现相邻时间差 >2 分钟的缺口。")


def main() -> None:
    os.chdir(ROOT)

    # 1) 价差图
    if not XAU_PATH.is_file() or not XAUT_PATH.is_file():
        raise SystemExit(f"缺少数据文件: {XAU_PATH} 或 {XAUT_PATH}")

    merged = load_and_merge(str(XAU_PATH), str(XAUT_PATH))
    sub = merged[(merged["datetime"] >= PLOT_START) & (merged["datetime"] < PLOT_END)].copy()
    if sub.empty:
        print("警告: 筛选 2026-01-10～2026-01-20 后无数据，仍写出空图。")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sub["datetime"], sub["spread"], color="#2563eb", linewidth=0.8)
    ax.set_title("XAU - XAUT spread (2026-01-10 ~ 2026-01-20)")
    ax.set_xlabel("UTC")
    ax.set_ylabel("spread")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    out_png = ROOT / "diagnose_spread.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)

    # 2) 连续性
    print("=== 时间戳缺口（相邻两行间隔 > 2 分钟，且缺口起点落在 1/13～1/17 UTC 日历日内）===")
    print_gaps_for_symbol("XAUUSDT", XAU_PATH)
    print_gaps_for_symbol("XAUTUSDT", XAUT_PATH)

    # 3) / 4) 共用最新 trade_details.csv
    td = latest_trade_details_path()
    print("=== 开仓密度（open_datetime 按分钟，>=3 笔）===")
    tdf: pd.DataFrame | None = None
    if td is None:
        print(f"未找到最新回测的 trade_details.csv（检查 {REPORT_DIR}）。")
    else:
        print(f"使用: {td}")
        tdf = pd.read_csv(td, encoding="utf-8-sig")

    if tdf is not None:
        if tdf.empty:
            print("trade_details 为空，跳过第 3、4 项。")
        else:
            if "open_datetime" not in tdf.columns:
                print("CSV 无 open_datetime 列，跳过第 3 项。")
            else:
                odt = pd.to_datetime(tdf["open_datetime"], utc=True, errors="coerce")
                tdf_open = tdf.assign(_odt=odt).dropna(subset=["_odt"])
                mask3 = (tdf_open["_odt"] >= TRADE_START) & (tdf_open["_odt"] < TRADE_END)
                part3 = tdf_open.loc[mask3]
                part3 = part3.assign(_minute=part3["_odt"].dt.floor("min"))
                cnt = part3.groupby("_minute", sort=True).size()
                hi = cnt[cnt >= 3]
                if hi.empty:
                    print("该区间内无每分钟开仓 >= 3 笔的时间点。")
                else:
                    for minute_ts, n in hi.items():
                        ms = minute_ts.strftime("%Y-%m-%d %H:%M")
                        print(f"{ms} -> {int(n)}笔")

            # 4) 按 close_datetime 筛选并逐笔打印
            print("=== 平仓明细（close_datetime 在 2026-01-15 00:00～2026-01-17 00:00 UTC，左闭右开）===")
            need_cols = [
                "open_datetime",
                "close_datetime",
                "spread_side",
                "entry_spread",
                "take_profit_spread",
                "pnl",
            ]
            missing = [c for c in need_cols if c not in tdf.columns]
            if missing:
                print(f"缺少列: {missing}，跳过第 4 项。")
            else:
                cdt = pd.to_datetime(tdf["close_datetime"], utc=True, errors="coerce")
                m4 = (cdt >= CLOSE_WIN_START) & (cdt < CLOSE_WIN_END)
                part4 = tdf.loc[m4, need_cols].copy()
                if part4.empty:
                    print("该时间窗口内无成交记录。")
                else:
                    for _, row in part4.iterrows():
                        print(
                            f"open_datetime={row['open_datetime']} | close_datetime={row['close_datetime']} | "
                            f"spread_side={row['spread_side']} | entry_spread={row['entry_spread']} | "
                            f"take_profit_spread={row['take_profit_spread']} | pnl={row['pnl']}"
                        )
                    total_pnl = pd.to_numeric(part4["pnl"], errors="coerce").sum()
                    n_trades = len(part4)
                    print(f"--- 窗口内总 pnl: {total_pnl:.6f}，成交笔数: {n_trades}")

    print("诊断完成，请查看 diagnose_spread.png。")


if __name__ == "__main__":
    main()
