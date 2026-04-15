import argparse
import html
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class GridConfig:
    initial_capital: float = 10000.0
    grid_center: float = 20.0
    grid_step: float = 5.0
    grid_width: float = 50.0
    qty: float = 1.0
    maker_fee: float = 0.000042
    taker_fee: float = 0.000042
    taker_slippage_pt: float = 0.2
    cb_trip: float = 0.0  # 熔断值，0 表示不启用。熔断线 = center ± cb_trip
    cb_reentry: float = 0.0  # 重进值，须 < cb_trip。重进线 = center ± cb_reentry

    @property
    def max_grid_units(self) -> int:
        # e.g. width=50, step=5 => 10 units each side
        return int(round(self.grid_width / self.grid_step))


@dataclass
class SpreadLot:
    open_datetime: str
    spread_side: str
    leg1_open_price: float
    leg1_open_direction: str
    leg2_open_price: float
    leg2_open_direction: str
    entry_fee: float
    leg1_qty: float
    leg2_qty: float


def grid_levels(cfg: GridConfig) -> List[float]:
    levels = []
    v = cfg.grid_center - cfg.grid_width
    vmax = cfg.grid_center + cfg.grid_width
    while v <= vmax + 1e-9:
        levels.append(round(v, 8))
        v += cfg.grid_step
    return levels


def load_and_merge(leg1_path: str, leg2_path: str) -> pd.DataFrame:
    def _read(path: str) -> pd.DataFrame:
        raw = pd.read_csv(path)
        # 自动识别时间戳列
        if "open_time_ms" in raw.columns:
            ts_col = "open_time_ms"
            ts_unit = "ms"
        elif "timestamp" in raw.columns:
            ts_col = "timestamp"
            ts_unit = "s"
        elif "open_time" in raw.columns:
            ts_col = "open_time"
            ts_unit = "s"
        else:
            raise ValueError(f"找不到时间戳列：{list(raw.columns)}")
        raw = raw[[ts_col, "close"]].copy()
        raw = raw.rename(columns={ts_col: "open_time"})
        if ts_unit == "s":
            # 兼容：有些“旧数据”虽然字段名是 timestamp/open_time，但实际已是毫秒级
            open_time_numeric = pd.to_numeric(raw["open_time"], errors="coerce")
            if open_time_numeric.max(skipna=True) < 1e11:
                raw["open_time"] = open_time_numeric * 1000
            else:
                raw["open_time"] = open_time_numeric
        return raw

    leg1 = _read(leg1_path).rename(columns={"close": "leg1_close"})
    leg2 = _read(leg2_path).rename(columns={"close": "leg2_close"})

    df = pd.merge(leg1, leg2, on="open_time", how="inner")
    df = df.sort_values("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("int64")
    df = df.dropna(subset=["open_time"])
    df["spread"] = df["leg1_close"] - df["leg2_close"]
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def _grid_key(x: float) -> float:
    """网格价位字典键统一为 8 位小数，避免浮点不一致导致同格多仓或快照失效。"""
    return round(float(x), 8)


def spread_to_target_units(spread: float, cfg: GridConfig) -> int:
    diff = spread - cfg.grid_center
    units = int(np.floor(diff / cfg.grid_step))
    if units > cfg.max_grid_units:
        units = cfg.max_grid_units
    if units < -cfg.max_grid_units:
        units = -cfg.max_grid_units
    return units


def run_backtest(
    df: pd.DataFrame | None,
    cfg: GridConfig,
    spreads: np.ndarray | None = None,
    leg1_closes: np.ndarray | None = None,
    leg2_closes: np.ndarray | None = None,
    datetimes: list | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame, float, float, List[Dict[str, Any]]]:
    cash = cfg.initial_capital
    leg1_qty = 0.0
    leg2_qty = 0.0
    current_units = 0
    total_taker_fee = 0.0
    total_maker_fee = 0.0
    trade_count = 0

    records: List[Dict] = []
    closed_arbs: List[Dict] = []
    positions: Dict[float, Dict] = {}
    grids = grid_levels(cfg)
    locked_long_levels: dict[float, float] = {}  # tp -> entry_spread
    locked_short_levels: dict[float, float] = {}  # tp -> entry_spread
    locked_reopen_long_levels: dict[float, float] = {}  # grid_level -> grid_level
    locked_reopen_short_levels: dict[float, float] = {}  # grid_level -> grid_level
    locked_tp_both: dict[float, float] = {}  # tp -> signed_entry_spread

    if df is not None:
        spreads = df["spread"].to_numpy(dtype=np.float64)
        leg1_closes = df["leg1_close"].to_numpy(dtype=np.float64)
        leg2_closes = df["leg2_close"].to_numpy(dtype=np.float64)
        open_times = df["open_time"].to_numpy()
        datetimes = df["datetime"].tolist()
    else:
        if spreads is None or leg1_closes is None or leg2_closes is None or datetimes is None:
            raise ValueError("When df is None, spreads/leg1_closes/leg2_closes/datetimes are required.")
        spreads = np.asarray(spreads, dtype=np.float64)
        leg1_closes = np.asarray(leg1_closes, dtype=np.float64)
        leg2_closes = np.asarray(leg2_closes, dtype=np.float64)
        dta = np.asarray(datetimes, dtype=np.int64)
        if len(dta) == len(spreads):
            open_times = dta
        else:
            open_times = np.arange(len(spreads), dtype=np.int64)

    half_step = cfg.grid_step / 2.0
    start_idx = 0
    for j in range(len(spreads)):
        if cfg.grid_center - half_step <= float(spreads[j]) <= cfg.grid_center + half_step:
            start_idx = j
            break
    else:
        start_idx = int(np.argmin(np.abs(spreads - float(cfg.grid_center))))

    prev_spread = float(spreads[start_idx])
    cb_tripped = False
    cb_trip_count = 0
    cb_events: List[Dict[str, Any]] = []

    use_int64_datetimes = df is None
    for i in range(start_idx, len(spreads)):
        spread = float(spreads[i])
        if use_int64_datetimes:
            dt = pd.Timestamp(int(datetimes[i]), unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
        else:
            dt = datetimes[i].strftime("%Y-%m-%d %H:%M:%S")
        leg1_px = float(leg1_closes[i])
        leg2_px = float(leg2_closes[i])

        # 解锁过期锁定
        for k in list(locked_long_levels.keys()):
            if spread <= locked_long_levels[k]:
                del locked_long_levels[k]
        for k in list(locked_short_levels.keys()):
            if spread >= locked_short_levels[k]:
                del locked_short_levels[k]
        for k in list(locked_reopen_long_levels.keys()):
            if spread > k + cfg.grid_step:
                del locked_reopen_long_levels[k]
        for k in list(locked_reopen_short_levels.keys()):
            if spread < k - cfg.grid_step:
                del locked_reopen_short_levels[k]
        for k in list(locked_tp_both.keys()):
            v = locked_tp_both[k]
            if v > 0 and spread <= v:
                del locked_tp_both[k]
            elif v < 0 and spread >= -v:
                del locked_tp_both[k]

        # --- 熔断机制 ---
        use_cb = cfg.cb_trip > 0 and cfg.cb_reentry > 0 and cfg.cb_reentry < cfg.cb_trip
        if use_cb:
            cb_upper_trip = cfg.grid_center + cfg.cb_trip
            cb_lower_trip = cfg.grid_center - cfg.cb_trip
            cb_upper_reentry = cfg.grid_center + cfg.cb_reentry
            cb_lower_reentry = cfg.grid_center - cfg.cb_reentry

            if not cb_tripped:
                if spread >= cb_upper_trip or spread <= cb_lower_trip:
                    cb_tripped = True
                    cb_trip_count += 1
                    positions_count_trip = len(positions)
                    forced_pnl_sum = 0.0
                    for g in list(positions.keys()):
                        pos = positions[g]
                        tp_spread = spread
                        actual_spread_move = tp_spread - pos["entry_spread"]
                        leg1_close_exec = pos["leg1_open_price"] + actual_spread_move / 2.0
                        leg2_close_exec = pos["leg2_open_price"] - actual_spread_move / 2.0
                        maker_fee = pos["leg1_qty"] * leg1_close_exec * cfg.maker_fee
                        taker_fee = pos["leg2_qty"] * leg2_close_exec * cfg.taker_fee
                        exit_fee = maker_fee + taker_fee
                        total_taker_fee += taker_fee
                        total_maker_fee += maker_fee
                        if pos["side"] == "LONG":
                            cash += pos["leg1_qty"] * leg1_close_exec
                            cash -= pos["leg2_qty"] * leg2_close_exec
                            cash -= exit_fee
                            leg1_qty -= pos["leg1_qty"]
                            leg2_qty += pos["leg2_qty"]
                            pnl = (
                                (leg1_close_exec - pos["leg1_open_price"]) * pos["leg1_qty"]
                                + (pos["leg2_open_price"] - leg2_close_exec) * pos["leg2_qty"]
                                - (pos["entry_fee"] + exit_fee)
                            )
                            leg1_close_direction = "SELL"
                            leg2_close_direction = "BUY"
                        else:
                            cash -= pos["leg1_qty"] * leg1_close_exec
                            cash += pos["leg2_qty"] * leg2_close_exec
                            cash -= exit_fee
                            leg1_qty += pos["leg1_qty"]
                            leg2_qty -= pos["leg2_qty"]
                            pnl = (
                                (pos["leg1_open_price"] - leg1_close_exec) * pos["leg1_qty"]
                                + (leg2_close_exec - pos["leg2_open_price"]) * pos["leg2_qty"]
                                - (pos["entry_fee"] + exit_fee)
                            )
                            leg1_close_direction = "BUY"
                            leg2_close_direction = "SELL"
                        forced_pnl_sum += float(pnl)
                        closed_arbs.append(
                            {
                                "open_datetime": pos["open_datetime"],
                                "close_datetime": dt,
                                "spread_side": "LONG_SPREAD" if pos["side"] == "LONG" else "SHORT_SPREAD",
                                "grid_level": pos["grid_level"],
                                "entry_spread": pos["entry_spread"],
                                "take_profit_spread": pos["take_profit_spread"],
                                "leg1_open_price": pos["leg1_open_price"],
                                "leg1_open_direction": pos["leg1_open_direction"],
                                "leg2_open_price": pos["leg2_open_price"],
                                "leg2_open_direction": pos["leg2_open_direction"],
                                "leg1_open_amount": pos["leg1_qty"] * pos["leg1_open_price"],
                                "leg2_open_amount": pos["leg2_qty"] * pos["leg2_open_price"],
                                "leg1_close_price": leg1_close_exec,
                                "leg1_close_direction": leg1_close_direction,
                                "leg2_close_price": leg2_close_exec,
                                "leg2_close_direction": leg2_close_direction,
                                "fee": pos["entry_fee"] + exit_fee,
                                "pnl": pnl,
                                "cb_forced": True,
                            }
                        )
                        trade_count += 1
                    cb_events.append(
                        {
                            "event": "TRIP",
                            "datetime": dt,
                            "spread": spread,
                            "positions_count": positions_count_trip,
                            "forced_pnl": forced_pnl_sum,
                            "cb_upper_trip": cb_upper_trip,
                            "cb_lower_trip": cb_lower_trip,
                        }
                    )
                    positions.clear()
                    locked_long_levels.clear()
                    locked_short_levels.clear()
                    locked_reopen_long_levels.clear()
                    locked_reopen_short_levels.clear()
                    locked_tp_both.clear()
                    prev_spread = spread
                    equity = cash + leg1_qty * leg1_px + leg2_qty * leg2_px
                    records.append(
                        {
                            "datetime": datetimes[i] if use_int64_datetimes else datetimes[i],
                            "open_time": int(open_times[i]),
                            "spread": float(spread),
                            "leg1_qty": float(leg1_qty),
                            "leg2_qty": float(leg2_qty),
                            "equity": float(equity),
                        }
                    )
                    continue

            else:
                if cb_lower_reentry <= spread <= cb_upper_reentry:
                    cb_tripped = False
                    grids_sorted = sorted(grids)
                    cur = spread
                    center = cfg.grid_center
                    eps = 1e-9
                    grids_to_rebuild: List[Tuple[float, str]] = []
                    if cur < center - eps:
                        cands = [g for g in grids_sorted if cur - eps <= g < center - eps]
                        grids_to_rebuild = [(g, "LONG") for g in sorted(cands, reverse=True)]
                    elif cur > center + eps:
                        cands = [g for g in grids_sorted if center + eps < g <= cur + eps]
                        grids_to_rebuild = [(g, "SHORT") for g in sorted(cands)]

                    for g_rb, side_rb in grids_to_rebuild:
                        gk_rb = round(float(g_rb), 8)
                        if gk_rb in positions:
                            continue
                        tp_rb = g_rb + cfg.grid_step if side_rb == "LONG" else g_rb - cfg.grid_step
                        leg1_open_exec = leg1_px
                        if side_rb == "LONG":
                            leg2_open_exec = leg2_px - cfg.taker_slippage_pt
                        else:
                            leg2_open_exec = leg2_px + cfg.taker_slippage_pt
                        maker_fee_o = cfg.qty * leg1_open_exec * cfg.maker_fee
                        taker_fee_o = cfg.qty * leg2_open_exec * cfg.taker_fee
                        entry_fee = maker_fee_o + taker_fee_o
                        total_taker_fee += taker_fee_o
                        total_maker_fee += maker_fee_o
                        if side_rb == "LONG":
                            cash -= cfg.qty * leg1_open_exec
                            cash += cfg.qty * leg2_open_exec
                            leg1_qty += cfg.qty
                            leg2_qty -= cfg.qty
                            leg1_dir, leg2_dir = "BUY", "SELL"
                        else:
                            cash += cfg.qty * leg1_open_exec
                            cash -= cfg.qty * leg2_open_exec
                            leg1_qty -= cfg.qty
                            leg2_qty += cfg.qty
                            leg1_dir, leg2_dir = "SELL", "BUY"
                        cash -= entry_fee
                        positions[gk_rb] = {
                            "grid_level": g_rb,
                            "side": side_rb,
                            "qty": cfg.qty,
                            "entry_spread": g_rb,
                            "take_profit_spread": tp_rb,
                            "open_datetime": dt,
                            "leg1_open_price": leg1_open_exec,
                            "leg1_open_direction": leg1_dir,
                            "leg2_open_price": leg2_open_exec,
                            "leg2_open_direction": leg2_dir,
                            "leg1_qty": cfg.qty,
                            "leg2_qty": cfg.qty,
                            "entry_fee": entry_fee,
                        }
                        trade_count += 1
                    cb_events.append(
                        {
                            "event": "REENTRY",
                            "datetime": dt,
                            "spread": spread,
                            "rebuilt_count": len(grids_to_rebuild),
                            "cb_upper_reentry": cb_upper_reentry,
                            "cb_lower_reentry": cb_lower_reentry,
                        }
                    )
                    prev_spread = spread
                    equity = cash + leg1_qty * leg1_px + leg2_qty * leg2_px
                    records.append(
                        {
                            "datetime": datetimes[i] if use_int64_datetimes else datetimes[i],
                            "open_time": int(open_times[i]),
                            "spread": float(spread),
                            "leg1_qty": float(leg1_qty),
                            "leg2_qty": float(leg2_qty),
                            "equity": float(equity),
                        }
                    )
                    continue
                else:
                    prev_spread = spread
                    equity = cash + leg1_qty * leg1_px + leg2_qty * leg2_px
                    records.append(
                        {
                            "datetime": datetimes[i] if use_int64_datetimes else datetimes[i],
                            "open_time": int(open_times[i]),
                            "spread": float(spread),
                            "leg1_qty": float(leg1_qty),
                            "leg2_qty": float(leg2_qty),
                            "equity": float(equity),
                        }
                    )
                    continue

        if i > start_idx:
            # 1) 先处理平仓：按每笔仓位各自止盈线精确匹配，不做批量平仓
            to_close = []
            seen_close = set()
            blocked_open_levels: set[float] = set()
            for g, pos in positions.items():
                if g in seen_close:
                    continue
                tp = float(pos["take_profit_spread"])
                if pos["side"] == "LONG":
                    # long仓止盈：价差上穿 tp 才触发
                    if prev_spread < tp <= spread:
                        to_close.append(g)
                        seen_close.add(g)
                else:
                    # short仓止盈：价差下穿 tp 才触发
                    if prev_spread > tp >= spread:
                        to_close.append(g)
                        seen_close.add(g)

            closed_this_bar = set()
            for g in sorted(to_close):
                pos = positions.pop(g)
                tp = float(pos["take_profit_spread"])
                blocked_open_levels.add(round(tp, 8))
                if pos["side"] == "LONG":
                    locked_reopen_short_levels[g] = g
                    locked_long_levels[tp] = pos["entry_spread"]
                    locked_tp_both[tp] = +pos["entry_spread"]
                    # 平仓执行价锁定在止盈价差，而非当前K线收盘价
                    tp_spread = pos["take_profit_spread"]
                    actual_spread_move = tp_spread - pos["entry_spread"]
                    leg1_close_exec = pos["leg1_open_price"] + actual_spread_move / 2
                    leg2_close_exec = pos["leg2_open_price"] - actual_spread_move / 2
                    maker_fee = pos["leg1_qty"] * leg1_close_exec * cfg.maker_fee
                    taker_fee = pos["leg2_qty"] * leg2_close_exec * cfg.taker_fee
                    exit_fee = maker_fee + taker_fee
                    total_taker_fee += taker_fee
                    total_maker_fee += maker_fee
                    cash += pos["leg1_qty"] * leg1_close_exec
                    cash -= pos["leg2_qty"] * leg2_close_exec
                    cash -= exit_fee
                    leg1_qty -= pos["leg1_qty"]
                    leg2_qty += pos["leg2_qty"]
                    pnl = (
                        (leg1_close_exec - pos["leg1_open_price"]) * pos["leg1_qty"]
                        + (pos["leg2_open_price"] - leg2_close_exec) * pos["leg2_qty"]
                        - (pos["entry_fee"] + exit_fee)
                    )
                    leg1_close_direction = "SELL"
                    leg2_close_direction = "BUY"
                else:
                    locked_reopen_long_levels[g] = g
                    locked_short_levels[tp] = pos["entry_spread"]
                    locked_tp_both[tp] = -pos["entry_spread"]
                    # 平仓执行价锁定在止盈价差，而非当前K线收盘价
                    tp_spread = pos["take_profit_spread"]
                    actual_spread_move = tp_spread - pos["entry_spread"]
                    leg1_close_exec = pos["leg1_open_price"] + actual_spread_move / 2
                    leg2_close_exec = pos["leg2_open_price"] - actual_spread_move / 2
                    maker_fee = pos["leg1_qty"] * leg1_close_exec * cfg.maker_fee
                    taker_fee = pos["leg2_qty"] * leg2_close_exec * cfg.taker_fee
                    exit_fee = maker_fee + taker_fee
                    total_taker_fee += taker_fee
                    total_maker_fee += maker_fee
                    cash -= pos["leg1_qty"] * leg1_close_exec
                    cash += pos["leg2_qty"] * leg2_close_exec
                    cash -= exit_fee
                    leg1_qty += pos["leg1_qty"]
                    leg2_qty -= pos["leg2_qty"]
                    pnl = (
                        (pos["leg1_open_price"] - leg1_close_exec) * pos["leg1_qty"]
                        + (leg2_close_exec - pos["leg2_open_price"]) * pos["leg2_qty"]
                        - (pos["entry_fee"] + exit_fee)
                    )
                    leg1_close_direction = "BUY"
                    leg2_close_direction = "SELL"

                assert g not in closed_this_bar, f"Duplicate close detected for grid level {g}"
                closed_this_bar.add(g)
                closed_arbs.append(
                    {
                        "open_datetime": pos["open_datetime"],
                        "close_datetime": dt,
                        "spread_side": "LONG_SPREAD" if pos["side"] == "LONG" else "SHORT_SPREAD",
                        "grid_level": pos["grid_level"],
                        "entry_spread": pos["entry_spread"],
                        "take_profit_spread": pos["take_profit_spread"],
                        "leg1_open_price": pos["leg1_open_price"],
                        "leg1_open_direction": pos["leg1_open_direction"],
                        "leg2_open_price": pos["leg2_open_price"],
                        "leg2_open_direction": pos["leg2_open_direction"],
                        "leg1_open_amount": pos["leg1_qty"] * pos["leg1_open_price"],
                        "leg2_open_amount": pos["leg2_qty"] * pos["leg2_open_price"],
                        "leg1_close_price": leg1_close_exec,
                        "leg1_close_direction": leg1_close_direction,
                        "leg2_close_price": leg2_close_exec,
                        "leg2_close_direction": leg2_close_direction,
                        "fee": pos["entry_fee"] + exit_fee,
                        "pnl": pnl,
                    }
                )
                trade_count += 1

            # 2) 再处理开仓：穿越触发；同一grid_level若已有持仓则跳过
            # 本根 K 线开仓判断一律基于快照，避免前面格的开仓改动 positions / locked_* 影响后面格
            positions_snapshot = set(positions.keys())
            locked_tp_both_snapshot = set(locked_tp_both.keys())
            locked_long_snapshot = set(locked_long_levels.keys())
            locked_short_snapshot = set(locked_short_levels.keys())
            locked_reopen_long_snapshot = set(locked_reopen_long_levels.keys())
            locked_reopen_short_snapshot = set(locked_reopen_short_levels.keys())
            # 本根 K 线内已对某网格价开过仓则不再开（防 grids 浮点重复或快照与 positions 键不一致）
            opened_this_bar: set[float] = set()

            for g in grids:
                down_cross = prev_spread > g >= spread
                up_cross = prev_spread < g <= spread
                gk = round(float(g), 8)

                if (
                    down_cross
                    and gk not in positions_snapshot
                    and gk not in opened_this_bar
                    and round(g, 8) not in blocked_open_levels
                    and g not in locked_tp_both_snapshot
                    and g not in locked_long_snapshot
                    and g not in locked_reopen_long_snapshot
                ):
                    # 向下穿越开多：买 leg1 卖 leg2
                    leg1_open_exec = leg1_px
                    leg2_open_exec = leg2_px - cfg.taker_slippage_pt
                    leg1_unit_qty = leg2_unit_qty = cfg.qty
                    maker_fee = leg1_unit_qty * leg1_open_exec * cfg.maker_fee
                    taker_fee = leg2_unit_qty * leg2_open_exec * cfg.taker_fee
                    entry_fee = maker_fee + taker_fee
                    total_taker_fee += taker_fee
                    total_maker_fee += maker_fee

                    cash -= leg1_unit_qty * leg1_open_exec
                    cash += leg2_unit_qty * leg2_open_exec
                    cash -= entry_fee
                    leg1_qty += leg1_unit_qty
                    leg2_qty -= leg2_unit_qty

                    positions[gk] = {
                        "grid_level": g,
                        "side": "LONG",
                        "qty": leg1_unit_qty,
                        "entry_spread": g,
                        "take_profit_spread": g + cfg.grid_step,
                        "open_datetime": dt,
                        "leg1_open_price": leg1_open_exec,
                        "leg1_open_direction": "BUY",
                        "leg2_open_price": leg2_open_exec,
                        "leg2_open_direction": "SELL",
                        "leg1_qty": leg1_unit_qty,
                        "leg2_qty": leg2_unit_qty,
                        "entry_fee": entry_fee,
                    }
                    opened_this_bar.add(gk)
                    trade_count += 1

                elif (
                    up_cross
                    and gk not in positions_snapshot
                    and gk not in opened_this_bar
                    and round(g, 8) not in blocked_open_levels
                    and g not in locked_tp_both_snapshot
                    and g not in locked_short_snapshot
                    and g not in locked_reopen_short_snapshot
                ):
                    # 向上穿越开空：卖 leg1 买 leg2
                    leg1_open_exec = leg1_px
                    leg2_open_exec = leg2_px + cfg.taker_slippage_pt
                    leg1_unit_qty = leg2_unit_qty = cfg.qty
                    maker_fee = leg1_unit_qty * leg1_open_exec * cfg.maker_fee
                    taker_fee = leg2_unit_qty * leg2_open_exec * cfg.taker_fee
                    entry_fee = maker_fee + taker_fee
                    total_taker_fee += taker_fee
                    total_maker_fee += maker_fee

                    cash += leg1_unit_qty * leg1_open_exec
                    cash -= leg2_unit_qty * leg2_open_exec
                    cash -= entry_fee
                    leg1_qty -= leg1_unit_qty
                    leg2_qty += leg2_unit_qty

                    positions[gk] = {
                        "grid_level": g,
                        "side": "SHORT",
                        "qty": leg1_unit_qty,
                        "entry_spread": g,
                        "take_profit_spread": g - cfg.grid_step,
                        "open_datetime": dt,
                        "leg1_open_price": leg1_open_exec,
                        "leg1_open_direction": "SELL",
                        "leg2_open_price": leg2_open_exec,
                        "leg2_open_direction": "BUY",
                        "leg1_qty": leg1_unit_qty,
                        "leg2_qty": leg2_unit_qty,
                        "entry_fee": entry_fee,
                    }
                    opened_this_bar.add(gk)
                    trade_count += 1

        current_units = len([p for p in positions.values() if p["side"] == "SHORT"]) - len(
            [p for p in positions.values() if p["side"] == "LONG"]
        )

        equity = cash + leg1_qty * float(leg1_closes[i]) + leg2_qty * float(leg2_closes[i])
        records.append(
            {
                "datetime": datetimes[i] if use_int64_datetimes else datetimes[i],
                "open_time": int(open_times[i]),
                "spread": float(spread),
                "leg1_qty": float(leg1_qty),
                "leg2_qty": float(leg2_qty),
                "equity": float(equity),
            }
        )
        prev_spread = spread

    result = pd.DataFrame(records)
    if use_int64_datetimes:
        result["datetime"] = pd.to_datetime(result["open_time"], unit="ms", utc=True)
    if not result.empty and os.environ.get("BACKTEST_DEBUG_EQUITY") == "1":
        print(
            "[权益调试用] 前10行 leg1_qty / leg2_qty / equity:\n"
            + result[["leg1_qty", "leg2_qty", "equity"]].head(10).to_string(index=False)
        )
    if result.empty:
        closed_df = pd.DataFrame(closed_arbs)
        metrics = {
            "initial_capital": cfg.initial_capital,
            "final_equity": cfg.initial_capital,
            "total_pnl": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "annualized_sharpe": 0.0,
            "trade_units_executed": 0,
            "total_taker_fee_paid": 0.0,
            "total_maker_fee_paid": 0.0,
            "total_fee_paid": 0.0,
            "grid_step": cfg.grid_step,
            "grid_width": cfg.grid_width,
            "max_grid_units_each_side": cfg.max_grid_units,
            "qty": cfg.qty,
            "closed_arb_count": 0,
            "closed_arb_win_rate": 0.0,
            "closed_arb_avg_pnl": 0.0,
            "closed_arb_total_pnl": 0.0,
            "backtest_start_idx": start_idx,
            "backtest_start_datetime": "",
            "backtest_start_spread": float(spreads[start_idx]) if len(spreads) else 0.0,
            "cb_trip_count": int(cb_trip_count),
        }
        return result, metrics, closed_df, 0.0, 0.0, cb_events

    equity_ret = result["equity"].pct_change().fillna(0.0)
    cummax_equity = result["equity"].cummax()
    result["drawdown"] = result["equity"] / cummax_equity - 1.0

    total_return = result["equity"].iloc[-1] / cfg.initial_capital - 1.0
    max_drawdown = result["drawdown"].min()
    if use_int64_datetimes:
        actual_start = pd.Timestamp(int(datetimes[start_idx]), unit="ms", tz="UTC")
        actual_end = pd.Timestamp(int(datetimes[-1]), unit="ms", tz="UTC")
    else:
        actual_start = datetimes[start_idx]
        actual_end = datetimes[-1]
    period_years = (actual_end - actual_start).total_seconds() / (365.0 * 24.0 * 3600.0)
    annualized_return = 0.0
    if period_years > 0:
        annualized_return = (result["equity"].iloc[-1] / cfg.initial_capital) ** (1.0 / period_years) - 1.0
    sharpe = 0.0
    if equity_ret.std() > 0:
        # minute bars annualized: 365*24*60
        sharpe = equity_ret.mean() / equity_ret.std() * np.sqrt(365 * 24 * 60)

    mdd_abs = abs(float(max_drawdown))
    calmar = float(annualized_return) / mdd_abs if mdd_abs > 0 else 0.0

    closed_df = pd.DataFrame(closed_arbs)
    if not closed_df.empty:
        win_rate = float((closed_df["pnl"] > 0).mean())
        avg_pnl = float(closed_df["pnl"].mean())
        total_realized_pnl = float(closed_df["pnl"].sum())
    else:
        win_rate = 0.0
        avg_pnl = 0.0
        total_realized_pnl = 0.0

    total_pnl = float(result["equity"].iloc[-1] - cfg.initial_capital)
    metrics = {
        "initial_capital": cfg.initial_capital,
        "final_equity": float(result["equity"].iloc[-1]),
        "total_pnl": total_pnl,
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": calmar,
        "annualized_sharpe": float(sharpe),
        "trade_units_executed": int(trade_count),
        "total_taker_fee_paid": float(total_taker_fee),
        "total_maker_fee_paid": float(total_maker_fee),
        "total_fee_paid": float(total_taker_fee + total_maker_fee),
        "grid_step": cfg.grid_step,
        "grid_width": cfg.grid_width,
        "max_grid_units_each_side": cfg.max_grid_units,
        "qty": cfg.qty,
        "closed_arb_count": int(len(closed_df)),
        "closed_arb_win_rate": win_rate,
        "closed_arb_avg_pnl": avg_pnl,
        "closed_arb_total_pnl": total_realized_pnl,
        "backtest_start_idx": int(start_idx),
        "backtest_start_datetime": str(actual_start),
        "backtest_start_spread": float(spreads[start_idx]),
        "cb_trip_count": int(cb_trip_count),
    }
    return result, metrics, closed_df, total_pnl, float(max_drawdown), cb_events


def save_equity_drawdown_plots(result: pd.DataFrame, report_dir: str) -> Tuple[str, str]:
    equity_png = os.path.join(report_dir, "equity_curve.png")
    dd_png = os.path.join(report_dir, "drawdown_curve.png")

    plt.figure(figsize=(12, 4.5))
    plt.plot(result["datetime"], result["equity"], color="#2563eb", linewidth=1.2)
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity (USDT)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(equity_png, dpi=130)
    plt.close()

    plt.figure(figsize=(12, 4.5))
    plt.plot(result["datetime"], result["drawdown"] * 100, color="#dc2626", linewidth=1.2)
    plt.title("Drawdown Curve")
    plt.xlabel("Time")
    plt.ylabel("Drawdown (%)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(dd_png, dpi=130)
    plt.close()
    return equity_png, dd_png


def _render_cb_section(cb_trip: float, cb_events: List[Dict[str, Any]]) -> str:
    if cb_trip == 0 or not cb_events:
        return (
            '<div class="section">'
            '<h2>熔断明细</h2>'
            '<p class="muted">未启用熔断机制</p>'
            "</div>"
        )
    trip_events = [e for e in cb_events if e.get("event") == "TRIP"]
    n_trip = len(trip_events)
    total_forced = sum(float(e.get("forced_pnl", 0.0)) for e in trip_events)

    rows: List[str] = []
    for e in cb_events:
        dt_s = html.escape(str(e.get("datetime", "")))
        sp = float(e.get("spread", 0.0))
        ev = e.get("event", "")
        if ev == "TRIP":
            evt_cell = '<span class="cb-trip">熔断触发</span>'
            u_t = float(e.get("cb_upper_trip", 0.0))
            lo_t = float(e.get("cb_lower_trip", 0.0))
            line_cell = f"+{u_t:.4f} / {lo_t:.4f}"
            ncell = str(int(e.get("positions_count", 0)))
            fp = float(e.get("forced_pnl", 0.0))
            pnl_cls = "pnl-pos" if fp >= 0 else "pnl-neg"
            pnl_cell = f'<span class="{pnl_cls}">{fp:,.2f}</span>'
        else:
            evt_cell = '<span class="cb-reentry">重进建仓</span>'
            u_r = float(e.get("cb_upper_reentry", 0.0))
            lo_r = float(e.get("cb_lower_reentry", 0.0))
            line_cell = f"+{u_r:.4f} / {lo_r:.4f}"
            ncell = str(int(e.get("rebuilt_count", 0)))
            pnl_cell = "-"
        rows.append(
            f"<tr><td>{dt_s}</td><td>{evt_cell}</td><td>{sp:.4f}</td>"
            f"<td>{html.escape(line_cell)}</td><td>{ncell}</td><td>{pnl_cell}</td></tr>"
        )
    rows_html = "\n".join(rows)
    summary = f"共触发熔断 {n_trip} 次，强平累计盈亏 {total_forced:,.2f} USDT。"
    return f"""
    <div class="section">
      <h2>熔断明细</h2>
      <div class="table-scroll">
        <table class="detail-table">
          <thead>
            <tr>
              <th>时间</th>
              <th>事件类型</th>
              <th>触发价差</th>
              <th>熔断线 / 重进线</th>
              <th>处理格数</th>
              <th>本次强平盈亏</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </div>
      <p class="muted" style="margin-top:12px">{html.escape(summary)}</p>
    </div>
    """


def build_html_report(
    metrics: Dict[str, float],
    pair_name: str,
    equity_png_name: str,
    dd_png_name: str,
    closed_df: pd.DataFrame,
    output_html_path: str,
    leg1_name: str = "leg1",
    leg2_name: str = "leg2",
    fee_note: str = "",
    cb_trip: float = 0.0,
    cb_events: List[Dict[str, Any]] | None = None,
) -> None:
    metric_name_map = {
        "initial_capital": "初始资金",
        "final_equity": "最终权益",
        "total_return": "总收益率",
        "annualized_return": "年化收益率",
        "max_drawdown": "最大回撤",
        "calmar_ratio": "卡玛比率",
        "annualized_sharpe": "年化夏普比率",
        "trade_units_executed": "成交笔数（开+平）",
        "total_taker_fee_paid": "吃单手续费总额",
        "total_maker_fee_paid": "挂单手续费总额",
        "total_fee_paid": "总手续费",
        "grid_step": "网格间距",
        "grid_width": "网格宽度",
        "max_grid_units_each_side": "每侧最大网格数",
        "qty": "每格下单数量",
        "closed_arb_count": "已完成套利笔数",
        "closed_arb_win_rate": "已完成套利胜率",
        "closed_arb_avg_pnl": "单笔平均盈利",
        "closed_arb_total_pnl": "已完成套利总盈利",
        "backtest_start_idx": "回测起始索引",
        "backtest_start_datetime": "回测起始时间",
        "backtest_start_spread": "回测起始价差",
        "cb_trip_count": "熔断触发次数",
    }

    metric_rows = []
    for k, v in metrics.items():
        label = metric_name_map.get(k, k)
        if isinstance(v, float):
            if "return" in k or "drawdown" in k or "win_rate" in k:
                metric_rows.append(f"<tr><td>{label}</td><td>{v:.2%}</td></tr>")
            else:
                metric_rows.append(f"<tr><td>{label}</td><td>{v:,.6f}</td></tr>")
        else:
            metric_rows.append(f"<tr><td>{label}</td><td>{v}</td></tr>")
    metric_table = "\n".join(metric_rows)

    if closed_df.empty:
        detail_table = "<p>暂无完整开平套利记录（可能样本内仍有未平仓）。</p>"
    else:
        show_df = closed_df.copy()
        # 兜底：若盈利为空则按现金流与手续费重算
        if "pnl" not in show_df.columns:
            show_df["pnl"] = np.nan
        if show_df["pnl"].isna().any():
            long_mask = show_df["spread_side"] == "LONG_SPREAD"
            show_df.loc[long_mask, "pnl"] = (
                (show_df.loc[long_mask, "leg1_close_price"] - show_df.loc[long_mask, "leg1_open_price"])
                / show_df.loc[long_mask, "leg1_open_price"]
                * show_df.loc[long_mask, "leg1_open_amount"]
                + (show_df.loc[long_mask, "leg2_open_price"] - show_df.loc[long_mask, "leg2_close_price"])
                / show_df.loc[long_mask, "leg2_open_price"]
                * show_df.loc[long_mask, "leg2_open_amount"]
                - show_df.loc[long_mask, "fee"]
            )
            show_df.loc[~long_mask, "pnl"] = (
                (show_df.loc[~long_mask, "leg1_open_price"] - show_df.loc[~long_mask, "leg1_close_price"])
                / show_df.loc[~long_mask, "leg1_open_price"]
                * show_df.loc[~long_mask, "leg1_open_amount"]
                + (show_df.loc[~long_mask, "leg2_close_price"] - show_df.loc[~long_mask, "leg2_open_price"])
                / show_df.loc[~long_mask, "leg2_open_price"]
                * show_df.loc[~long_mask, "leg2_open_amount"]
                - show_df.loc[~long_mask, "fee"]
            )

        c1o, c2o, c1c, c2c = (
            f"{leg1_name}开仓价格",
            f"{leg2_name}开仓价格",
            f"{leg1_name}平仓价格",
            f"{leg2_name}平仓价格",
        )
        c1od, c2od, c1cd, c2cd = (
            f"{leg1_name}开仓方向",
            f"{leg2_name}开仓方向",
            f"{leg1_name}平仓方向",
            f"{leg2_name}平仓方向",
        )
        show_df = show_df[
            [
                "open_datetime",
                "close_datetime",
                "spread_side",
                "leg1_open_price",
                "leg1_open_direction",
                "leg2_open_price",
                "leg2_open_direction",
                "leg1_close_price",
                "leg1_close_direction",
                "leg2_close_price",
                "leg2_close_direction",
                "fee",
                "pnl",
            ]
        ]
        show_df.columns = [
            "开仓时间",
            "平仓时间",
            "套利方向",
            c1o,
            c1od,
            c2o,
            c2od,
            c1c,
            c1cd,
            c2c,
            c2cd,
            "手续费",
            "盈利",
        ]
        for c in [c1o, c2o, c1c, c2c]:
            show_df[c] = show_df[c].map(lambda x: f"{x:.3f}")
        for c in ["手续费", "盈利"]:
            show_df[c] = show_df[c].map(lambda x: f"{x:,.4f}")
        detail_table = show_df.to_html(index=False, classes="detail-table", border=0, escape=False)

    cb_ev = cb_events if cb_events is not None else []
    cb_section = _render_cb_section(float(cb_trip), cb_ev)

    note_line = fee_note or "手续费与滑点以 GridConfig 为准；成交明细按一次完整套利（开仓+平仓）统计。"
    html = f"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{pair_name} 网格套利回测报告</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --primary: #2563eb;
      --danger: #dc2626;
      --border: #e5e7eb;
      --head: #f8fafc;
    }}
    body {{
      margin: 0;
      padding: 0;
      background: linear-gradient(180deg, #eef4ff 0%, var(--bg) 28%);
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
    }}
    .container {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    .hero {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px 22px;
      box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    }}
    h1 {{
      margin: 0;
      font-size: 26px;
      color: #111827;
    }}
    .note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
    .section {{
      margin-top: 20px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .metric-table, .detail-table {{
      border-collapse: collapse;
      width: 100%;
      overflow: hidden;
      border-radius: 10px;
    }}
    .metric-table th, .metric-table td, .detail-table th, .detail-table td {{
      border: 1px solid var(--border);
      padding: 9px 10px;
      font-size: 13px;
      white-space: nowrap;
    }}
    .metric-table th, .detail-table th {{
      background: var(--head);
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .metric-table tr:nth-child(even), .detail-table tr:nth-child(even) {{
      background: #fbfdff;
    }}
    .chart-wrap {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px;
      background: #fff;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 8px;
    }}
    .table-scroll {{
      max-height: 520px;
      overflow: auto;
      border-radius: 10px;
    }}
    .cb-trip {{ color: #dc2626; font-weight: 600; }}
    .cb-reentry {{ color: #16a34a; font-weight: 600; }}
    .pnl-pos {{ color: #16a34a; font-weight: 600; }}
    .pnl-neg {{ color: #dc2626; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1>{pair_name} 价差网格套利回测报告</h1>
      <p class="note">{note_line}</p>
    </div>

    <div class="section">
      <h2>关键指标</h2>
      <table class="metric-table">
        <thead><tr><th>指标</th><th>数值</th></tr></thead>
        <tbody>
          {metric_table}
        </tbody>
      </table>
    </div>

    <div class="section">
      <h2>权益曲线</h2>
      <div class="chart-wrap">
        <img src="{equity_png_name}" alt="equity curve" />
      </div>
    </div>

    <div class="section">
      <h2>最大回撤曲线</h2>
      <div class="chart-wrap">
        <img src="{dd_png_name}" alt="drawdown curve" />
      </div>
    </div>

    <div class="section">
      <h2>成交明细（按一次完整套利开平）</h2>
      <div class="table-scroll">
        {detail_table}
      </div>
    </div>
    {cb_section}
  </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)


def _infer_labels(leg1_path: str, leg2_path: str) -> Tuple[str, str, str]:
    leg1_label = os.path.splitext(os.path.basename(leg1_path))[0].split("_")[0]
    leg2_label = os.path.splitext(os.path.basename(leg2_path))[0].split("_")[0]
    return leg1_label, leg2_label, f"{leg1_label}-{leg2_label}"


def _resolve_input_csv(path: str) -> str:
    """若给定路径不存在，则尝试 ./data/<文件名> 或当前目录下的同名文件。"""
    if os.path.isfile(path):
        return path
    bn = os.path.basename(path.rstrip("/\\"))
    for cand in (os.path.join("data", bn), bn):
        if os.path.isfile(cand):
            return cand
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Spread grid backtest (generic leg1/leg2)")
    parser.add_argument(
        "--leg1",
        required=True,
        help="leg1 CSV 路径；若不存在则尝试 data/<文件名>",
    )
    parser.add_argument(
        "--leg2",
        required=True,
        help="leg2 CSV 路径；若不存在则尝试 data/<文件名>",
    )
    parser.add_argument("--pair_name", default="", help="交易对名称，默认从文件名推断")
    parser.add_argument("--center", type=float, default=None, help="网格中枢，默认用数据价差均值")
    parser.add_argument("--step", type=float, default=3.5)
    parser.add_argument("--width", type=float, default=20.0)
    parser.add_argument("--qty", type=float, default=1.0)
    parser.add_argument("--maker_fee", type=float, default=0.000042)
    parser.add_argument("--taker_fee", type=float, default=0.000042)
    parser.add_argument("--slippage", type=float, default=0.2)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--report_dir", default="report", help="Output report directory")
    parser.add_argument(
        "--cb_trip",
        type=float,
        default=0.0,
        help="熔断值，0=不启用，熔断线=center±cb_trip",
    )
    parser.add_argument(
        "--cb_reentry",
        type=float,
        default=0.0,
        help="重进值，重进线=center±cb_reentry，需小于 cb_trip",
    )
    args = parser.parse_args()

    args.leg1 = _resolve_input_csv(args.leg1)
    args.leg2 = _resolve_input_csv(args.leg2)

    leg1_label, leg2_label, default_pair = _infer_labels(args.leg1, args.leg2)
    pair_display = args.pair_name.strip() or default_pair

    df = load_and_merge(args.leg1, args.leg2)
    grid_center = float(args.center) if args.center is not None else float(df["spread"].mean())

    cfg = GridConfig(
        initial_capital=args.capital,
        grid_center=grid_center,
        grid_step=args.step,
        grid_width=args.width,
        qty=args.qty,
        maker_fee=args.maker_fee,
        taker_fee=args.taker_fee,
        taker_slippage_pt=args.slippage,
        cb_trip=args.cb_trip,
        cb_reentry=args.cb_reentry,
    )

    os.makedirs(args.report_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H-%M")
    run_report_dir = os.path.join(args.report_dir, f"{run_id}_{pair_display.replace('/', '-')}")
    os.makedirs(run_report_dir, exist_ok=True)

    result, metrics, closed_df, _, _, cb_events = run_backtest(df, cfg)

    curve_csv = os.path.join(run_report_dir, "equity_curve.csv")
    detail_csv = os.path.join(run_report_dir, "trade_details.csv")
    report_html = os.path.join(run_report_dir, "backtest_report.html")

    result.to_csv(curve_csv, index=False, encoding="utf-8-sig")
    closed_df.to_csv(detail_csv, index=False, encoding="utf-8-sig")
    equity_png, dd_png = save_equity_drawdown_plots(result, run_report_dir)
    fee_note = (
        f"maker_fee={cfg.maker_fee}, taker_fee={cfg.taker_fee}, "
        f"taker_slippage_pt={cfg.taker_slippage_pt}；{leg1_label}/{leg2_label} 价差网格。"
    )
    build_html_report(
        metrics=metrics,
        pair_name=pair_display,
        equity_png_name=os.path.basename(equity_png),
        dd_png_name=os.path.basename(dd_png),
        closed_df=closed_df,
        output_html_path=report_html,
        leg1_name=leg1_label,
        leg2_name=leg2_label,
        fee_note=fee_note,
        cb_trip=cfg.cb_trip,
        cb_events=cb_events,
    )

    print("=== Backtest Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print(f"report_html: {report_html}")
    print(f"equity_curve_csv: {curve_csv}")
    print(f"trade_details_csv: {detail_csv}")
    print(f"run_report_dir: {run_report_dir}")


if __name__ == "__main__":
    main()
