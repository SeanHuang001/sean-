"""
多交易对、多轮自动缩窄的网格参数搜索（并行）。
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
from datetime import datetime
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backtest_xau_xaut_grid import GridConfig, load_and_merge, run_backtest


def _resolve_leg_csv(data_dir: str, filename: str) -> str:
    """优先 data_dir 下文件，其次项目内 ./data/，再当前目录文件名。"""
    candidates = [
        os.path.join(data_dir, filename),
        os.path.join("data", filename),
        filename,
    ]
    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if os.path.isfile(c):
            return c
    return os.path.join(data_dir, filename)


def _fmt_table(df: pd.DataFrame) -> str:
    """终端/UTF-8 下中英文混排对齐：须按东亚字宽计算列宽（tabulate 默认不按双宽算）。"""
    if df.empty:
        return ""
    with pd.option_context(
        "display.unicode.east_asian_width",
        True,
        "display.colheader_justify",
        "right",
        "display.width",
        1000,
    ):
        return df.to_string(index=False, float_format="%.2f")


PAIRS: Dict[str, Dict[str, Any]] = {
    "XAU-XAUT": {
        "leg1": "XAUUSDT_1m.csv",
        "leg2": "XAUTUSDT_1m.csv",
        "maker_fee": 0.000042,
        "taker_fee": 0.000042,
        "slippage": 0.2,
    },
    "XAU-PAXG": {
        "leg1": "XAUUSDT_1m.csv",
        "leg2": "PAXGUSDT_1m.csv",
        "maker_fee": 0.000042,
        "taker_fee": 0.000042,
        "slippage": 0.2,
    },
    "XAUT-PAXG": {
        "leg1": "XAUTUSDT_1m.csv",
        "leg2": "PAXGUSDT_1m.csv",
        "maker_fee": 0.000042,
        "taker_fee": 0.000042,
        "slippage": 0.2,
    },
    "PAXG-XAUT": {
        "leg1": "PAXGUSDT_1m.csv",
        "leg2": "XAUTUSDT_1m.csv",
        "maker_fee": 0.000042,
        "taker_fee": 0.000042,
        "slippage": 0.2,
    },
    "COPPER-XCU": {
        "leg1": "COPPERUSDT_1m.csv",
        "leg2": "XCUUSDT_1m.csv",
        "maker_fee": 0.000042,
        "taker_fee": 0.000042,
        "slippage": 0.2,
    },
    "XBR-XTI": {
        "leg1": "XBRUSDT_1m.csv",
        "leg2": "XTIUSDT_1m.csv",
        "maker_fee": 0.000042,
        "taker_fee": 0.000042,
        "slippage": 0.2,
    },
}

_spreads: Optional[np.ndarray] = None
_leg1: Optional[np.ndarray] = None
_leg2: Optional[np.ndarray] = None
_dts: Optional[np.ndarray] = None
_MAX_LEVERAGE: float = 40.0


def _init_worker(
    spreads: np.ndarray,
    leg1: np.ndarray,
    leg2: np.ndarray,
    dts: np.ndarray,
    max_leverage: float,
) -> None:
    global _spreads, _leg1, _leg2, _dts, _MAX_LEVERAGE
    _spreads = spreads
    _leg1 = leg1
    _leg2 = leg2
    _dts = dts
    _MAX_LEVERAGE = float(max_leverage)


def _run_one(args_tuple: Tuple) -> Optional[Dict[str, Any]]:
    (
        step,
        width,
        qty,
        capital,
        center,
        maker_fee,
        taker_fee,
        slippage,
        leg1_price,
        stress_spread,
    ) = args_tuple

    lev = (width / step) * qty * leg1_price * 2.0 / capital
    if lev > _MAX_LEVERAGE:
        return None

    if float(step) <= 0:
        return None
    n_long = math.floor(float(width) / float(step))
    spread_drop = float(center) - float(stress_spread)  # 正值表示价差下跌
    if spread_drop > 0 and n_long > 0:
        # 第k张long仓开仓价差为 center - k*step，到stress_spread时亏损为：
        # (center - k*step) - stress_spread = spread_drop - k*step
        # 累加 k=1..n_long：n_long*spread_drop - step*n_long*(n_long+1)/2
        stress_loss = float(qty) * (
            n_long * spread_drop - float(step) * n_long * (n_long + 1) / 2.0
        )
        stress_loss = max(stress_loss, 0.0)  # 若参数导致结果为负则归零
    else:
        stress_loss = 0.0
    stress_drawdown_pct = stress_loss / float(capital) * 100.0
    if stress_drawdown_pct > 50.0:
        return None

    cfg = GridConfig(
        initial_capital=capital,
        grid_center=center,
        grid_step=float(step),
        grid_width=float(width),
        qty=float(qty),
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        taker_slippage_pt=slippage,
    )
    assert _spreads is not None and _leg1 is not None and _leg2 is not None and _dts is not None
    result, metrics, _, total_pnl, max_drawdown = run_backtest(
        None,
        cfg,
        spreads=_spreads,
        leg1_closes=_leg1,
        leg2_closes=_leg2,
        datetimes=_dts,
    )

    if metrics["trade_units_executed"] < 20:
        return None

    mdd_abs = abs(float(max_drawdown))
    calmar = metrics["annualized_return"] / mdd_abs if mdd_abs > 0 else 0.0

    return {
        "grid_step": float(step),
        "grid_width": float(width),
        "qty": int(qty),
        "grid_center": float(center),
        "近似杠杆": round(float(lev), 2),
        "total_pnl": round(float(total_pnl), 2),
        "年化收益率": round(float(metrics["annualized_return"]) * 100, 2),
        "总手续费": round(float(metrics["total_fee_paid"]), 2),
        "最大回撤百分比": round(float(max_drawdown) * 100, 2),
        "压力回撤%": round(float(stress_drawdown_pct), 2),
        "卡玛比率": round(float(calmar), 2),
        "交易次数": int(metrics["trade_units_executed"]),
        "回测起始时间": str(metrics.get("backtest_start_datetime", "")),
    }


def compute_stats(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    spread_mean = round(float(df["spread"].mean()), 4)
    spread_std = float(df["spread"].std())
    spread_range = float(df["spread"].max() - df["spread"].min())
    leg1_price = float(df["leg1_close"].median())
    return spread_mean, spread_std, spread_range, leg1_price


def compute_round1_ranges(
    spread_range: float,
    leg1_price: float,
    max_leverage: float,
    capital: float,
) -> Tuple[List[float], List[float], List[int], float, float, float, float, int]:
    step_min = round(spread_range * 0.005, 4)
    step_max = round(spread_range * 0.20, 4)
    if step_min < 0.01:
        step_min = 0.01
    round1_steps = [round(x, 4) for x in np.linspace(step_min, step_max, 12).tolist()]

    width_min = round(spread_range * 0.05, 4)
    width_max = round(spread_range * 1.0, 4)
    round1_widths = [round(x, 4) for x in np.linspace(width_min, width_max, 10).tolist()]

    median_width = round1_widths[len(round1_widths) // 2]
    median_step = round1_steps[len(round1_steps) // 2]
    typical_layers = median_width / median_step if median_step > 0 else 1.0
    qty_max_by_leverage = max_leverage * capital / (typical_layers * leg1_price * 2)
    qty_max = max(1, int(qty_max_by_leverage))
    qty_max = min(qty_max, 100)

    if qty_max <= 10:
        round1_qtys = list(range(1, qty_max + 1))
    else:
        round1_qtys = sorted(
            set(
                [1, 2, 3]
                + [int(x) for x in np.unique(np.geomspace(1, qty_max, 10).astype(int)).tolist()]
            )
        )

    return round1_steps, round1_widths, round1_qtys, step_min, step_max, width_min, width_max, qty_max


def narrow_range(
    top10_df: pd.DataFrame, col: str, n_points: int, min_step_size: float = 0.25
) -> List[float]:
    if top10_df.empty or col not in top10_df.columns:
        return []
    values = sorted(top10_df[col].unique().tolist())
    lo = float(min(values))
    hi = float(max(values))
    spread = hi - lo
    if spread < 1e-12:
        lo = max(lo * 0.95, min_step_size)
        hi = max(hi * 1.05, lo + min_step_size)
    else:
        lo = max(lo - spread * 0.2, min_step_size)
        hi = hi + spread * 0.2
    return [round(x, 4) for x in np.linspace(lo, hi, n_points).tolist()]


def narrow_range_int(top10_df: pd.DataFrame, col: str, n_points: int, min_v: int = 1) -> List[int]:
    if top10_df.empty or col not in top10_df.columns:
        return []
    vals = [int(x) for x in top10_df[col].unique().tolist()]
    lo = min(vals)
    hi = max(vals)
    spread = hi - lo
    if spread <= 0:
        lo = max(min_v, lo - 1)
        hi = hi + 1
    else:
        lo = max(min_v, int(lo - max(1, round(spread * 0.2))))
        hi = int(hi + max(1, round(spread * 0.2)))
    pts = np.linspace(lo, hi, n_points)
    out = sorted({max(min_v, int(round(x))) for x in pts})
    return out


def get_top10_by_calmar(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("卡玛比率", ascending=False).head(10).reset_index(drop=True)


def run_search_round(
    round_num: int,
    steps: Sequence[float],
    widths: Sequence[float],
    qtys: Sequence[int],
    grid_center: float,
    pair_cfg: Dict[str, Any],
    args: argparse.Namespace,
    leg1_price: float,
    stress_spread: float,
) -> List[Dict[str, Any]]:
    tasks = [
        (
            s,
            w,
            q,
            args.capital,
            grid_center,
            pair_cfg["maker_fee"],
            pair_cfg["taker_fee"],
            pair_cfg["slippage"],
            leg1_price,
            stress_spread,
        )
        for s, w, q in itertools.product(steps, widths, qtys)
    ]
    n_total = len(tasks)
    results: List[Dict[str, Any]] = []
    step_print = max(1, n_total // 20) if n_total >= 20 else 1

    with Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(
            _spreads,
            _leg1,
            _leg2,
            _dts,
            args.max_leverage,
        ),
    ) as pool:
        done = 0
        for r in pool.imap_unordered(_run_one, tasks, chunksize=max(1, n_total // (args.workers * 8))):
            done += 1
            if done % step_print == 0 or done == n_total:
                pct = 100.0 * done / n_total
                print(f"[第{round_num}轮] 已完成 {done}/{n_total} 组 ({pct:.1f}%)")
            if r is not None:
                results.append(r)

    return results


def _pair_safe_name(pair: str) -> str:
    return pair.replace("/", "-").replace("\\", "-")


def write_report(
    pair_name: str,
    leg1_file: str,
    leg2_file: str,
    all_results: List[Dict[str, Any]],
    grid_center: float,
    spread_mean: float,
    spread_std: float,
    spread_range: float,
    rounds: int,
    round_summaries: List[Dict[str, Any]],
    stress_spread: float,
) -> str:
    out_name = f"optimization_results_{_pair_safe_name(pair_name)}.txt"
    out_path = os.path.join(os.getcwd(), out_name)

    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# 网格搜索报告: {pair_name}")
    lines.append(f"# 运行时间: {now}")
    lines.append(f"# 数据: {leg1_file} + {leg2_file}")
    lines.append(f"# 指定 grid_center: {grid_center}")
    lines.append(f"# stress_spread = {stress_spread}")
    lines.append(f"# 价差统计: 均值={spread_mean}, 标准差={spread_std:.4f}, 极差={spread_range:.4f}")
    lines.append(f"# 搜索轮数: {rounds}")
    lines.append("# 回测起始点: 自动找 center 附近")
    lines.append("")

    for rs in round_summaries:
        lines.append("========================================")
        lines.append(f"第 {rs['round']} 轮 {rs['label']}")
        lines.append(
            f"参数范围: step=[{rs['step_lo']}-{rs['step_hi']}], "
            f"width=[{rs['width_lo']}-{rs['width_hi']}], qty=[{rs['qty_lo']}-{rs['qty_hi']}]"
        )
        lines.append(f"组合数: {rs['n_combos']}")
        lines.append(f"有效结果数: {rs['valid_n']}")
        lines.append("========================================")
        lines.append("")
        lines.append("Top10 按卡玛比率:")
        if rs["top10"] is not None and not rs["top10"].empty:
            lines.append(_fmt_table(rs["top10"]))
        else:
            lines.append("(无)")
        lines.append("")

    combined = pd.DataFrame(all_results)
    if not combined.empty:
        combined = combined.sort_values("卡玛比率", ascending=False).drop_duplicates(
            subset=["grid_step", "grid_width", "qty"], keep="first"
        )

    lines.append("========================================")
    lines.append("最终结果（所有轮次合并去重）")
    lines.append("========================================")
    lines.append("")

    cols_show = [
        "grid_step",
        "grid_width",
        "qty",
        "grid_center",
        "近似杠杆",
        "年化收益率",
        "最大回撤百分比",
        "压力回撤%",
        "卡玛比率",
        "total_pnl",
        "总手续费",
        "交易次数",
        "回测起始时间",
    ]

    def _block(title: str, df: pd.DataFrame) -> None:
        lines.append(title)
        if df.empty:
            lines.append("(无)")
        else:
            lines.append(_fmt_table(df[cols_show].head(50)))
        lines.append("")

    if combined.empty:
        _block("Top50 按卡玛比率降序:", combined)
    else:
        _block("Top50 按卡玛比率降序:", combined.sort_values("卡玛比率", ascending=False))
        _block("Top50 按年化收益率降序:", combined.sort_values("年化收益率", ascending=False))
        _block("Top50 按 total_pnl 降序:", combined.sort_values("total_pnl", ascending=False))
        sub = combined[combined["total_pnl"] > 0].copy()
        if not sub.empty:
            denom = sub["总手续费"].replace(0, np.nan)
            sub = sub.assign(_eff=sub["年化收益率"] / denom).sort_values("_eff", ascending=False)
            sub = sub.drop(columns=["_eff"])
        _block("Top50 按 年化收益率/总手续费 降序（筛选 total_pnl > 0）:", sub)

    text = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多交易对多轮网格参数搜索")
    parser.add_argument("--pair", required=True, choices=list(PAIRS.keys()), help="交易对名称")
    parser.add_argument("--center", required=True, type=float, help="网格中枢（手动指定）")
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=None,
        help="手动指定 step 列表；不指定则由价差自动推算",
    )
    parser.add_argument(
        "--widths",
        type=float,
        nargs="+",
        default=None,
        help="手动指定 width 列表；不指定则由价差自动推算",
    )
    parser.add_argument(
        "--qtys",
        type=float,
        nargs="+",
        default=None,
        help="手动指定 qty 列表；不指定则由杠杆等自动推算",
    )
    parser.add_argument("--rounds", type=int, default=3, help="搜索轮数，每轮缩窄范围（任一手动参数时固定为 1）")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--max_leverage", type=float, default=40.0)
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    parser.add_argument(
        "--data_dir",
        default="data",
        help="K 线 CSV 所在目录（默认项目下 data 文件夹）",
    )
    parser.add_argument(
        "--stress_spread",
        type=float,
        default=120.0,
        help="价差压力测试值（用于压力回撤%% 与筛选 <=50%%）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_name = args.pair
    pair_cfg = PAIRS[pair_name]

    leg1_path = _resolve_leg_csv(args.data_dir, pair_cfg["leg1"])
    leg2_path = _resolve_leg_csv(args.data_dir, pair_cfg["leg2"])

    df = load_and_merge(leg1_path, leg2_path)
    spread_mean, spread_std, spread_range, leg1_price = compute_stats(df)
    center = float(args.center)

    print(f"[AUTO] {pair_name}")
    print(f"  指定 center: {center}")
    print(f"  价差均值={spread_mean}, 标准差={spread_std:.4f}, 极差={spread_range:.4f}")
    print(f"  leg1 中位价={leg1_price:.2f}")
    print(f"  stress_spread={float(args.stress_spread)}")

    global _spreads, _leg1, _leg2, _dts
    _spreads = df["spread"].to_numpy(dtype=np.float64)
    _leg1 = df["leg1_close"].to_numpy(dtype=np.float64)
    _leg2 = df["leg2_close"].to_numpy(dtype=np.float64)
    _dts = df["open_time"].to_numpy(dtype=np.int64)

    (
        round1_steps,
        round1_widths,
        round1_qtys,
        step_min,
        step_max,
        width_min,
        width_max,
        qty_max,
    ) = compute_round1_ranges(spread_range, leg1_price, args.max_leverage, args.capital)

    manual_any = args.steps is not None or args.widths is not None or args.qtys is not None
    effective_rounds = 1 if manual_any else int(args.rounds)

    if manual_any and args.rounds != 1:
        print(f"  手动参数: 搜索轮数强制为 1（忽略 --rounds {args.rounds}）")

    if args.steps is not None:
        current_steps = [round(float(x), 4) for x in args.steps]
        print(f"  step: 手动 {current_steps} ({len(current_steps)} 个点)")
    else:
        current_steps = round1_steps
        print(f"  第一轮 step 范围: [{step_min}, {step_max}] ({len(round1_steps)} 个点)")

    if args.widths is not None:
        current_widths = [round(float(x), 4) for x in args.widths]
        print(f"  width: 手动 {current_widths} ({len(current_widths)} 个点)")
    else:
        current_widths = round1_widths
        print(f"  第一轮 width 范围: [{width_min}, {width_max}] ({len(round1_widths)} 个点)")

    if args.qtys is not None:
        current_qtys = [max(1, int(round(float(x)))) for x in args.qtys]
        print(f"  qty: 手动 {current_qtys} ({len(current_qtys)} 个点)")
    else:
        current_qtys = round1_qtys
        print(f"  第一轮 qty 范围: [1, {qty_max}] → {round1_qtys}")

    n1 = len(current_steps) * len(current_widths) * len(current_qtys)
    print(f"  第一轮总组合数: {n1}")

    all_results: List[Dict[str, Any]] = []
    round_summaries: List[Dict[str, Any]] = []

    for round_num in range(1, effective_rounds + 1):
        label = "粗搜索" if round_num == 1 else "细搜索"
        n_combos = len(current_steps) * len(current_widths) * len(current_qtys)
        print(
            f"\n[第{round_num}轮] 参数范围: step[{min(current_steps):.4f}-{max(current_steps):.4f}] "
            f"({len(current_steps)}点), width[{min(current_widths):.4f}-{max(current_widths):.4f}] "
            f"({len(current_widths)}点), qty {sorted(set(current_qtys))} ({len(set(current_qtys))}点)"
        )
        print(f"[第{round_num}轮] 组合数: {n_combos}")

        results = run_search_round(
            round_num,
            current_steps,
            current_widths,
            current_qtys,
            center,
            pair_cfg,
            args,
            leg1_price,
            float(args.stress_spread),
        )
        all_results.extend(results)
        valid_n = len(results)
        print(f"[第{round_num}轮] 完成，有效结果 {valid_n} 组")

        top10 = get_top10_by_calmar(results)
        print(f"[第{round_num}轮] Top10 卡玛比率:")
        if top10.empty:
            print("(无)")
        else:
            print(_fmt_table(top10))

        round_summaries.append(
            {
                "round": round_num,
                "label": label,
                "step_lo": min(current_steps) if current_steps else 0,
                "step_hi": max(current_steps) if current_steps else 0,
                "width_lo": min(current_widths) if current_widths else 0,
                "width_hi": max(current_widths) if current_widths else 0,
                "qty_lo": min(current_qtys) if current_qtys else 0,
                "qty_hi": max(current_qtys) if current_qtys else 0,
                "n_combos": n_combos,
                "valid_n": valid_n,
                "top10": top10,
            }
        )

        if round_num >= effective_rounds:
            break

        if top10.empty:
            print(f"[第{round_num}轮] 无有效结果，停止搜索")
            break

        current_steps = narrow_range(top10, "grid_step", 12, min_step_size=0.01)
        current_widths = narrow_range(top10, "grid_width", 10, min_step_size=0.1)
        current_qtys = narrow_range_int(top10, "qty", 8, min_v=1)
        if not current_steps or not current_widths or not current_qtys:
            print(f"[第{round_num}轮] 缩窄后范围为空，停止搜索")
            break

        nxt = round_num + 1
        print(f"\n[第{nxt}轮] 参数范围自动缩窄:")
        print(
            f"  step: [{min(current_steps):.4f}, {max(current_steps):.4f}] ({len(current_steps)} 个点)"
        )
        print(
            f"  width: [{min(current_widths):.4f}, {max(current_widths):.4f}] ({len(current_widths)} 个点)"
        )
        print(f"  qty: {current_qtys} ({len(current_qtys)} 个点)")
        print(
            f"  组合数: {len(current_steps) * len(current_widths) * len(current_qtys)}"
        )

    out_path = write_report(
        pair_name,
        pair_cfg["leg1"],
        pair_cfg["leg2"],
        all_results,
        center,
        spread_mean,
        spread_std,
        spread_range,
        effective_rounds,
        round_summaries,
        float(args.stress_spread),
    )
    print(f"\n报告已写入: {out_path}")


if __name__ == "__main__":
    main()
