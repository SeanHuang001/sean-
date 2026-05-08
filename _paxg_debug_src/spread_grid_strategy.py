"""
价差网格策略：独立线程消费 tick，产生开平仓信号；on_fill 驱动持仓与锁定状态。
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

log = logging.getLogger(__name__)

_SIGNALS_JSONL_PATH = Path("signals.jsonl")
_signals_jsonl_lock = threading.Lock()

_STRATEGY_PAIR_DEBUG_PATH = Path("strategy_pair_debug.jsonl")
_strategy_pair_debug_lock = threading.Lock()
_strategy_pair_debug_last_mono: dict[str, float] = {}

_SIGNAL_WINDOW_SEC = 1.0
_SIGNAL_THROTTLE_COUNT = 2
_PENDING_TIMEOUT_SEC = 300.0

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class PairConfig:
    pair_name: str
    leg1: str
    leg2: str
    grid_center: float
    grid_step: float
    grid_width: float  # 半宽：网格区间为 [center - width, center + width]
    qty: float
    max_tick_age_ms: int = 2000
    enabled: bool = True
    cb_trip: float = 0.0  # 熔断偏移量，0=不启用。熔断线 = center ± cb_trip
    cb_reentry: float = 0.0  # 重进偏移量，须 < cb_trip。重进线 = center ± cb_reentry
    cb_consecutive: int = 1  # 连续 N 个 tick 超过熔断线才触发，默认 1（立即触发）


def _grid_levels(cfg: PairConfig) -> list[float]:
    # grid_width 为半宽：范围 [center - width, center + width]
    lo = cfg.grid_center - cfg.grid_width
    hi = cfg.grid_center + cfg.grid_width
    step = cfg.grid_step
    if step <= 0:
        return [round(cfg.grid_center, 10)]
    out: list[float] = []
    x = lo
    n = 0
    while x <= hi + 1e-12 and n < 10_000:
        out.append(round(x, 10))
        x += step
        n += 1
    return out


def _norm_grid(g: float) -> float:
    return round(float(g), 10)


def _leg_contracts_per_unit(leg_symbol: str) -> int:
    """每 1 单位 qty 对应的合约张数。XAU 1张=0.01盎司，XAUT/PAXG 1张=0.001盎司。
    为了两腿等量对冲（都是 qty*0.01 盎司），XAU 每单位下1张，XAUT/PAXG 每单位下10张。
    XCU 1张=1单位，COPPER 1张=0.1单位，1:10 对冲；XTI/XBR 两腿 contract_size 相同，1:1 对冲。"""
    if leg_symbol == "XAUUSDT":
        return 1
    if leg_symbol in ("XAUTUSDT", "PAXGUSDT"):
        return 10
    if leg_symbol == "XCUUSDT":
        return 1  # XCU 1张=1单位，COPPER 1张=0.1单位，1:10对冲
    if leg_symbol == "COPPERUSDT":
        return 10
    if leg_symbol in ("XTIUSDT", "XBRUSDT"):
        return 1  # 两腿contract_size相同，1:1对冲
    return 1


def _ts_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")


def _append_signals_jsonl(record: dict[str, Any]) -> None:
    try:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _signals_jsonl_lock:
            with open(_SIGNALS_JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        log.warning("signals.jsonl 写入失败: %s", e)


def _append_strategy_pair_debug(record: dict[str, Any]) -> None:
    try:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _strategy_pair_debug_lock:
            with open(_STRATEGY_PAIR_DEBUG_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        log.debug("strategy_pair_debug.jsonl 写入失败: %s", e)


def _throttled_pair_spreads_debug(
    ps: "PairState",
    entry_spread: Optional[float],
    exit_spread: Optional[float],
    event_ts_ms: Optional[int],
) -> None:
    """每秒最多写一行：诊断两腿时间戳是否不同步（猜想 A）。"""
    key = ps.config.pair_name
    now = time.monotonic()
    last = _strategy_pair_debug_last_mono.get(key, 0.0)
    if now - last < 1.0:
        return
    _strategy_pair_debug_last_mono[key] = now
    age_ms = abs(int(ps.leg1_ms_t) - int(ps.leg2_ms_t))
    _append_strategy_pair_debug(
        {
            "ts_ms": int(time.time() * 1000),
            "pair": ps.config.pair_name,
            "leg1_bid": ps.leg1_bid,
            "leg1_ask": ps.leg1_ask,
            "leg1_ms_t": ps.leg1_ms_t,
            "leg2_bid": ps.leg2_bid,
            "leg2_ask": ps.leg2_ask,
            "leg2_ms_t": ps.leg2_ms_t,
            "entry_spread": entry_spread,
            "exit_spread": exit_spread,
            "age_ms": age_ms,
            "event_ts_ms": event_ts_ms,
        }
    )


@dataclass
class CloseLockRollback:
    """平仓挂单时写入的锁定项，failed 时按此回滚。"""

    long_tp: Optional[float] = None
    short_tp: Optional[float] = None
    reopen_long_g: Optional[float] = None
    reopen_short_g: Optional[float] = None
    tp_both_tp: Optional[float] = None

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.long_tp is not None:
            d["long_tp"] = self.long_tp
        if self.short_tp is not None:
            d["short_tp"] = self.short_tp
        if self.reopen_long_g is not None:
            d["reopen_long_g"] = self.reopen_long_g
        if self.reopen_short_g is not None:
            d["reopen_short_g"] = self.reopen_short_g
        if self.tp_both_tp is not None:
            d["tp_both_tp"] = self.tp_both_tp
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> CloseLockRollback:
        # 兼容旧版误存为网格线 g 的 tp_both_g
        tp_both_tp = d.get("tp_both_tp", d.get("tp_both_g"))
        return CloseLockRollback(
            long_tp=d.get("long_tp"),
            short_tp=d.get("short_tp"),
            reopen_long_g=d.get("reopen_long_g"),
            reopen_short_g=d.get("reopen_short_g"),
            tp_both_tp=tp_both_tp,
        )


@dataclass
class PairState:
    config: PairConfig
    leg1_bid: Optional[float] = None
    leg1_ask: Optional[float] = None
    leg2_bid: Optional[float] = None
    leg2_ask: Optional[float] = None
    leg1_ms_t: int = 0
    leg2_ms_t: int = 0
    prev_entry: Optional[float] = None
    prev_exit: Optional[float] = None
    is_initialized: bool = False
    tick_count: int = 0
    positions: dict[float, dict[str, Any]] = field(default_factory=dict)
    pending_open_ts: dict[float, float] = field(default_factory=dict)
    pending_close_ts: dict[float, float] = field(default_factory=dict)
    signal_timestamps: list[float] = field(default_factory=list)
    locked_long_levels: dict[float, float] = field(default_factory=dict)
    locked_short_levels: dict[float, float] = field(default_factory=dict)
    locked_reopen_long_levels: dict[float, float] = field(default_factory=dict)
    locked_reopen_short_levels: dict[float, float] = field(default_factory=dict)
    locked_tp_both: dict[float, float] = field(default_factory=dict)
    close_rollbacks: dict[float, CloseLockRollback] = field(default_factory=dict)
    pending_open_meta: dict[float, dict[str, Any]] = field(default_factory=dict)
    cb_tripped: bool = False
    cb_consecutive_count: int = 0  # 不持久化，重启归零
    cb_trip_count: int = 0  # 持久化，用于监控


class SpreadGridStrategy:
    def __init__(
        self,
        configs: list[PairConfig],
        signal_q: queue.Queue,
        state_dir: Union[str, Path] = ".",
    ) -> None:
        self._signal_q = signal_q
        self._tick_q: queue.Queue = queue.Queue()
        self._state_dir = Path(state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._pairs: dict[str, PairState] = {}
        for c in configs:
            self._pairs[c.pair_name] = PairState(config=c)
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._saver: Optional[threading.Thread] = None
        self._pending_timeout_cr_logged: set[tuple[str, float, str]] = set()
        for name, ps in self._pairs.items():
            self._load_state(name, ps)

    # --- 持久化 ---

    def _state_path(self, pair_name: str) -> Path:
        safe = pair_name.replace("/", "_").replace("\\", "_")
        return self._state_dir / f"strategy_state_{safe}.json"

    def _float_key_dump(self, d: dict[float, Any]) -> dict[str, Any]:
        return {str(k): v for k, v in d.items()}

    def _float_key_load(self, d: dict[str, Any], inner: str = "dict") -> dict[float, Any]:
        out: dict[float, Any] = {}
        for k, v in d.items():
            out[float(k)] = v
        return out

    def _serialize_pair(self, ps: PairState) -> dict[str, Any]:
        cr: dict[str, Any] = {}
        for g, rb in ps.close_rollbacks.items():
            cr[str(g)] = rb.to_json()
        pom: dict[str, Any] = {str(g): deepcopy(m) for g, m in ps.pending_open_meta.items()}
        return {
            "is_initialized": ps.is_initialized,
            "positions": self._float_key_dump(ps.positions),
            "pending_open_ts": self._float_key_dump(ps.pending_open_ts),
            "pending_close_ts": self._float_key_dump(ps.pending_close_ts),
            "locked_long_levels": self._float_key_dump(ps.locked_long_levels),
            "locked_short_levels": self._float_key_dump(ps.locked_short_levels),
            "locked_reopen_long_levels": self._float_key_dump(ps.locked_reopen_long_levels),
            "locked_reopen_short_levels": self._float_key_dump(ps.locked_reopen_short_levels),
            "locked_tp_both": self._float_key_dump(ps.locked_tp_both),
            "close_rollbacks": cr,
            "pending_open_meta": pom,
            "cb_tripped": ps.cb_tripped,
            "cb_trip_count": ps.cb_trip_count,
        }

    def _deserialize_pair(self, ps: PairState, data: dict[str, Any]) -> None:
        ps.is_initialized = bool(data.get("is_initialized", False))
        # prev 不从文件恢复，启动后由前两个合法 tick 重新初始化
        ps.prev_entry = None
        ps.prev_exit = None
        ps.positions = self._float_key_load(data.get("positions") or {})
        ps.pending_open_ts = self._float_key_load(data.get("pending_open_ts") or {})
        if not ps.pending_open_ts and data.get("pending_opens"):
            for x in data["pending_opens"]:
                ps.pending_open_ts[float(x)] = 0.0
        ps.pending_close_ts = self._float_key_load(data.get("pending_close_ts") or {})
        if not ps.pending_close_ts and data.get("pending_closes"):
            for x in data["pending_closes"]:
                ps.pending_close_ts[float(x)] = 0.0
        ps.locked_long_levels = self._float_key_load(data.get("locked_long_levels") or {})
        ps.locked_short_levels = self._float_key_load(data.get("locked_short_levels") or {})
        ps.locked_reopen_long_levels = self._float_key_load(
            data.get("locked_reopen_long_levels") or {}
        )
        ps.locked_reopen_short_levels = self._float_key_load(
            data.get("locked_reopen_short_levels") or {}
        )
        ps.locked_tp_both = self._float_key_load(data.get("locked_tp_both") or {})
        ps.close_rollbacks.clear()
        for ks, vd in (data.get("close_rollbacks") or {}).items():
            ps.close_rollbacks[float(ks)] = CloseLockRollback.from_json(vd)
        ps.pending_open_meta.clear()
        for ks, vd in (data.get("pending_open_meta") or {}).items():
            ps.pending_open_meta[float(ks)] = vd
        ps.cb_tripped = bool(data.get("cb_tripped", False))
        ps.cb_trip_count = int(data.get("cb_trip_count", 0))
        ps.cb_consecutive_count = 0
        ps.tick_count = 0

    def _load_state(self, pair_name: str, ps: PairState) -> None:
        path = self._state_path(pair_name)
        if not path.is_file():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self._deserialize_pair(ps, data)
            self._clear_residual_pending_on_startup(pair_name, ps)
        except Exception as e:
            log.warning("加载状态失败 %s: %s", path, e)

    def _clear_residual_pending_on_startup(self, pair_name: str, ps: PairState) -> None:
        if not ps.pending_open_ts and not ps.pending_close_ts:
            return
        log.warning(
            "[WARN] pending 状态残留，已清空 pair=%s",
            pair_name,
        )
        for g in list(ps.pending_close_ts.keys()):
            rb = ps.close_rollbacks.pop(g, CloseLockRollback())
            self._rollback_close_locks(ps, rb)
            pos = ps.positions.get(g)
            if pos is not None:
                pos["closing"] = False
        ps.pending_close_ts.clear()
        ps.pending_open_ts.clear()
        ps.pending_open_meta.clear()
        self._pending_timeout_cr_logged = {
            k for k in self._pending_timeout_cr_logged if k[0] != pair_name
        }
        try:
            self._save_pair(pair_name, ps)
        except Exception as e:
            log.warning("清空 pending 后保存失败 %s: %s", pair_name, e)

    def _save_pair(self, pair_name: str, ps: PairState) -> None:
        path = self._state_path(pair_name)
        tmp = path.with_suffix(".tmp")
        blob = json.dumps(self._serialize_pair(ps), ensure_ascii=False, indent=2)
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(blob)
        tmp.replace(path)

    def _save_all_states(self) -> None:
        with self._lock:
            items = list(self._pairs.items())
        for name, ps in items:
            try:
                self._save_pair(name, ps)
            except Exception as e:
                log.warning("保存状态失败 %s: %s", name, e)

    # --- 锁定 ---

    def _apply_unlocks(self, ps: PairState, entry_spread: float, exit_spread: float) -> None:
        cfg = ps.config
        # 多/空仓止盈锁：tp 为止盈价；价差一旦达到 tp（与止盈触发同向）即解除，避免长期挡同格重开
        for tp, _v in list(ps.locked_long_levels.items()):
            if exit_spread >= tp:
                del ps.locked_long_levels[tp]
        for tp, _v in list(ps.locked_short_levels.items()):
            if entry_spread <= tp:
                del ps.locked_short_levels[tp]
        for k, _ in list(ps.locked_reopen_long_levels.items()):
            if exit_spread > k + cfg.grid_step:
                del ps.locked_reopen_long_levels[k]
        for k, _ in list(ps.locked_reopen_short_levels.items()):
            if entry_spread < k - cfg.grid_step:
                del ps.locked_reopen_short_levels[k]
        # key=止盈价 tp；v>0 多仓止盈记录用 exit 达 tp 解锁，v<0 空仓用 entry 达 tp 解锁
        for tp_key, v in list(ps.locked_tp_both.items()):
            if v > 0:
                if exit_spread >= tp_key:
                    del ps.locked_tp_both[tp_key]
            elif v < 0:
                if entry_spread <= tp_key:
                    del ps.locked_tp_both[tp_key]
            else:
                del ps.locked_tp_both[tp_key]

    def _rollback_close_locks(self, ps: PairState, rb: CloseLockRollback) -> None:
        if rb.long_tp is not None:
            ps.locked_long_levels.pop(rb.long_tp, None)
        if rb.short_tp is not None:
            ps.locked_short_levels.pop(rb.short_tp, None)
        if rb.reopen_long_g is not None:
            ps.locked_reopen_long_levels.pop(rb.reopen_long_g, None)
        if rb.reopen_short_g is not None:
            ps.locked_reopen_short_levels.pop(rb.reopen_short_g, None)
        if rb.tp_both_tp is not None:
            ps.locked_tp_both.pop(rb.tp_both_tp, None)

    def _install_close_locks_long(
        self, ps: PairState, grid_level: float, tp: float, entry_spread: float
    ) -> CloseLockRollback:
        g = _norm_grid(grid_level)
        tp = float(tp)
        es = float(entry_spread)
        ps.locked_long_levels[tp] = es
        ps.locked_reopen_short_levels[g] = g
        ps.locked_tp_both[tp] = es
        return CloseLockRollback(
            long_tp=tp, reopen_short_g=g, tp_both_tp=tp
        )

    def _install_close_locks_short(
        self, ps: PairState, grid_level: float, tp: float, entry_spread: float
    ) -> CloseLockRollback:
        g = _norm_grid(grid_level)
        tp = float(tp)
        es = float(entry_spread)
        ps.locked_short_levels[tp] = es
        ps.locked_reopen_long_levels[g] = g
        ps.locked_tp_both[tp] = -es
        return CloseLockRollback(
            short_tp=tp, reopen_long_g=g, tp_both_tp=tp
        )

    def _block_open_long(self, ps: PairState, g: float) -> bool:
        ng = _norm_grid(g)
        step = ps.config.grid_step if ps.config.grid_step > 0 else 0.0
        tp = _norm_grid(ng + step)
        if tp in ps.locked_long_levels:
            return True
        if ng in ps.locked_reopen_long_levels:
            return True
        if tp in ps.locked_tp_both:
            return True
        return False

    def _block_open_short(self, ps: PairState, g: float) -> bool:
        ng = _norm_grid(g)
        step = ps.config.grid_step if ps.config.grid_step > 0 else 0.0
        tp = _norm_grid(ng - step)
        if tp in ps.locked_short_levels:
            return True
        if ng in ps.locked_reopen_short_levels:
            return True
        if tp in ps.locked_tp_both:
            return True
        return False

    # --- 核心 tick ---

    def _pair_spreads(self, ps: PairState) -> Optional[tuple[float, float, int]]:
        c = ps.config
        if ps.leg1_bid is None or ps.leg1_ask is None:
            _throttled_pair_spreads_debug(ps, None, None, None)
            return None
        if ps.leg2_bid is None or ps.leg2_ask is None:
            _throttled_pair_spreads_debug(ps, None, None, None)
            return None
        entry_spread = ps.leg1_ask - ps.leg2_bid
        exit_spread = ps.leg1_bid - ps.leg2_ask
        ts_ms = min(ps.leg1_ms_t, ps.leg2_ms_t)
        _throttled_pair_spreads_debug(ps, entry_spread, exit_spread, ts_ms)
        age = abs(ps.leg1_ms_t - ps.leg2_ms_t)
        if age > c.max_tick_age_ms:
            return None
        return entry_spread, exit_spread, ts_ms

    def _update_leg_from_tick(self, ps: PairState, tick: dict[str, Any]) -> None:
        sym = tick["symbol"]
        c = ps.config
        if sym == c.leg1:
            ps.leg1_bid = float(tick["bid"])
            ps.leg1_ask = float(tick["ask"])
            ps.leg1_ms_t = int(tick["ms_t"])
        elif sym == c.leg2:
            ps.leg2_bid = float(tick["bid"])
            ps.leg2_ask = float(tick["ask"])
            ps.leg2_ms_t = int(tick["ms_t"])

    def _emit_open(
        self,
        ps: PairState,
        side: str,
        grid_level: float,
        entry_spread: float,
        ts_ms: int,
        mid_spread: float,
        out: list[dict[str, Any]],
    ) -> None:
        c = ps.config
        g = _norm_grid(grid_level)
        step = c.grid_step if c.grid_step > 0 else 0.0
        if side == "LONG":
            tp = g + step
        else:
            tp = g - step
        leg1_sz = int(c.qty) * _leg_contracts_per_unit(c.leg1)
        leg2_sz = int(c.qty) * _leg_contracts_per_unit(c.leg2)
        leg1_mid = (float(ps.leg1_bid) + float(ps.leg1_ask)) / 2.0
        leg2_mid = (float(ps.leg2_bid) + float(ps.leg2_ask)) / 2.0
        meta = {
            "side": side,
            "entry_spread": entry_spread,
            "take_profit_spread": tp,
            "qty": c.qty,
            "leg1_size": leg1_sz,
            "leg2_size": leg2_sz,
        }
        ps.pending_open_meta[g] = meta
        ps.pending_open_ts[g] = time.time()
        sig = {
            "type": "open",
            "pair": c.pair_name,
            "side": side,
            "leg1": c.leg1,
            "leg2": c.leg2,
            "qty": c.qty,
            "leg1_size": leg1_sz,
            "leg2_size": leg2_sz,
            "grid_level": g,
            "take_profit_spread": tp,
            "ts_ms": ts_ms,
            "leg1_mid": leg1_mid,
            "leg2_mid": leg2_mid,
        }
        out.append(sig)
        ps.signal_timestamps.append(time.monotonic())
        action = "空" if side == "SHORT" else "多"
        _append_signals_jsonl(
            {
                "ts_ms": ts_ms,
                "pair": c.pair_name,
                "mid_spread": float(g),
                "grid_level": float(g),
                "action": action,
            }
        )
        log.info(
            "[SIGNAL %s] %s OPEN  %s  grid=%s  entry_spread=%s  cfg_qty=%s  contracts %s=%s %s=%s",
            _ts_str(ts_ms),
            c.pair_name,
            side,
            g,
            entry_spread,
            c.qty,
            c.leg1,
            leg1_sz,
            c.leg2,
            leg2_sz,
        )

    def _emit_close(
        self,
        ps: PairState,
        grid_level: float,
        pos: dict[str, Any],
        ts_ms: int,
        exit_spread_mkt: float,
        out: list[dict[str, Any]],
    ) -> None:
        c = ps.config
        g = _norm_grid(grid_level)
        side = pos["side"]
        tp = float(pos["take_profit_spread"])
        entry_spread = float(pos["entry_spread"])
        leg1_mid = (float(ps.leg1_bid) + float(ps.leg1_ask)) / 2.0
        leg2_mid = (float(ps.leg2_bid) + float(ps.leg2_ask)) / 2.0
        leg1_sz = int(c.qty) * _leg_contracts_per_unit(c.leg1)
        leg2_sz = int(c.qty) * _leg_contracts_per_unit(c.leg2)
        sig = {
            "type": "close",
            "pair": c.pair_name,
            "side": side,
            "leg1": c.leg1,
            "leg2": c.leg2,
            "qty": pos["qty"],
            "leg1_size": leg1_sz,
            "leg2_size": leg2_sz,
            "grid_level": g,
            "entry_spread": entry_spread,
            "take_profit_spread": tp,
            "ts_ms": ts_ms,
            "exit_spread_market": float(exit_spread_mkt),
            "leg1_mid": leg1_mid,
            "leg2_mid": leg2_mid,
            "open_time": str(pos.get("open_time", "")),
            "open_fee": float(pos.get("open_fee", 0.0)),
        }
        out.append(sig)
        _append_signals_jsonl(
            {
                "ts_ms": ts_ms,
                "pair": c.pair_name,
                "mid_spread": float(tp),
                "grid_level": float(g),
                "action": "平",
            }
        )
        log.info(
            "[SIGNAL %s] %s CLOSE %s  grid=%s  tp=%s           cfg_qty=%s  contracts %s=%s %s=%s",
            _ts_str(ts_ms),
            c.pair_name,
            side,
            g,
            tp,
            c.qty,
            c.leg1,
            leg1_sz,
            c.leg2,
            leg2_sz,
        )

    def _process_pair_tick(self, ps: PairState, out: list[dict[str, Any]]) -> None:
        c = ps.config
        if not c.enabled:
            return
        got = self._pair_spreads(ps)
        if got is None:
            return
        entry_spread, exit_spread, ts_ms = got

        if ps.tick_count < 2:
            ps.prev_entry = entry_spread
            ps.prev_exit = exit_spread
            ps.tick_count += 1
            if ps.tick_count == 1:
                ps.is_initialized = True
            return

        pe, px = ps.prev_entry, ps.prev_exit
        if pe is None or px is None:
            ps.prev_entry = entry_spread
            ps.prev_exit = exit_spread
            return

        mid = (entry_spread + exit_spread) / 2.0

        self._apply_unlocks(ps, entry_spread, exit_spread)

        levels = _grid_levels(c)

        use_cb = (
            c.cb_trip > 0
            and c.cb_reentry > 0
            and c.cb_reentry < c.cb_trip
        )
        if not use_cb:
            ps.cb_consecutive_count = 0
        else:
            cb_upper_trip = c.grid_center + c.cb_trip
            cb_lower_trip = c.grid_center - c.cb_trip
            cb_upper_reentry = c.grid_center + c.cb_reentry
            cb_lower_reentry = c.grid_center - c.cb_reentry

            if not ps.cb_tripped:
                if exit_spread >= cb_upper_trip or entry_spread <= cb_lower_trip:
                    ps.cb_consecutive_count += 1
                else:
                    ps.cb_consecutive_count = 0

                if ps.cb_consecutive_count >= c.cb_consecutive:
                    ps.cb_tripped = True
                    ps.cb_trip_count += 1
                    ps.cb_consecutive_count = 0
                    log.warning(
                        "[CB_TRIP] pair=%s entry=%.4f exit=%.4f trip_count=%d positions=%d",
                        c.pair_name,
                        entry_spread,
                        exit_spread,
                        ps.cb_trip_count,
                        len(ps.positions),
                    )
                    for g_key in list(ps.positions.keys()):
                        g = _norm_grid(float(g_key))
                        pos = ps.positions.get(g)
                        if pos is None or pos.get("closing"):
                            continue
                        side = pos["side"]
                        tp = float(pos["take_profit_spread"])
                        es0 = float(pos["entry_spread"])
                        if side == "LONG":
                            rb = self._install_close_locks_long(ps, g, tp, es0)
                        else:
                            rb = self._install_close_locks_short(ps, g, tp, es0)
                        ps.close_rollbacks[g] = rb
                        ps.pending_close_ts[g] = time.time()
                        pos["closing"] = True
                        self._emit_close(ps, g, pos, ts_ms, exit_spread, out)
                    # 熔断强平后清空所有锁定，避免重进后格位被永久封锁
                    ps.locked_long_levels.clear()
                    ps.locked_short_levels.clear()
                    ps.locked_reopen_long_levels.clear()
                    ps.locked_reopen_short_levels.clear()
                    ps.locked_tp_both.clear()
                    ps.close_rollbacks.clear()
                    self._save_pair(c.pair_name, ps)
            else:
                if cb_lower_reentry <= mid <= cb_upper_reentry:
                    if not ps.positions and not ps.pending_close_ts:
                        ps.cb_tripped = False
                        ps.cb_consecutive_count = 0
                        log.info(
                            "[CB_REENTRY] pair=%s mid=%.4f 熔断解除，恢复正常开仓",
                            c.pair_name,
                            mid,
                        )
                        self._save_pair(c.pair_name, ps)

        # 先平仓
        for g in list(levels):
            g = _norm_grid(g)
            if g not in ps.positions:
                continue
            pos = ps.positions[g]
            if pos.get("closing"):
                continue
            side = pos["side"]
            tp = float(pos["take_profit_spread"])
            if side == "LONG":
                # 多头止盈：exit_spread 从下向上穿越 tp
                if px < tp <= exit_spread:
                    rb = self._install_close_locks_long(ps, g, tp, float(pos["entry_spread"]))
                    ps.close_rollbacks[g] = rb
                    ps.pending_close_ts[g] = time.time()
                    pos["closing"] = True
                    self._emit_close(ps, g, pos, ts_ms, exit_spread, out)
            else:
                # 空头止盈：entry_spread 从上向下穿越 tp
                if pe > tp >= entry_spread:
                    rb = self._install_close_locks_short(ps, g, tp, float(pos["entry_spread"]))
                    ps.close_rollbacks[g] = rb
                    ps.pending_close_ts[g] = time.time()
                    pos["closing"] = True
                    self._emit_close(ps, g, pos, ts_ms, exit_spread, out)

        # 再开仓（限流仅作用于开仓）
        now_m = time.monotonic()
        ps.signal_timestamps = [t for t in ps.signal_timestamps if t > now_m - _SIGNAL_WINDOW_SEC]
        throttled = len(ps.signal_timestamps) >= _SIGNAL_THROTTLE_COUNT

        def _is_slot_free(gn: float) -> bool:
            return (
                gn not in ps.positions
                and gn not in ps.pending_open_ts
                and gn not in ps.pending_close_ts
            )

        def _is_tp_of_existing_position(g: float) -> bool:
            for pos in ps.positions.values():
                if pos.get("closing"):
                    continue
                tp = float(pos.get("take_profit_spread", 0))
                if abs(tp - g) < 1e-9:
                    return True
            return False

        if throttled:
            log.warning(
                "[THROTTLE] pair=%s 最近%.1fs 内信号数>=%s，本 tick 跳过开仓",
                c.pair_name,
                _SIGNAL_WINDOW_SEC,
                _SIGNAL_THROTTLE_COUNT,
            )
        if not throttled and not ps.cb_tripped:
            # 候选：穿越范围 + g 相对 mid + g 相对 center（LONG 仅 center 下方，SHORT 仅 center 上方）
            long_candidate = None
            pe_val = ps.prev_entry if ps.prev_entry is not None else float("inf")
            for g in levels:
                g = _norm_grid(g)
                if (
                    g < c.grid_center
                    and g > mid
                    and entry_spread <= g < pe_val
                    and _is_slot_free(g)
                    and not self._block_open_long(ps, g)
                    and not _is_tp_of_existing_position(g)
                ):
                    long_candidate = g
                    break

            short_candidate = None
            px_val = ps.prev_exit if ps.prev_exit is not None else float("-inf")
            for g in reversed(levels):
                g = _norm_grid(g)
                if (
                    g > c.grid_center
                    and g < mid
                    and px_val < g <= exit_spread
                    and _is_slot_free(g)
                    and not self._block_open_short(ps, g)
                    and not _is_tp_of_existing_position(g)
                ):
                    short_candidate = g
                    break

            if long_candidate is not None:
                g = long_candidate
                if pe > g >= entry_spread:
                    self._emit_open(ps, "LONG", g, entry_spread, ts_ms, mid, out)

            if short_candidate is not None:
                g = short_candidate
                if px < g <= exit_spread:
                    self._emit_open(ps, "SHORT", g, exit_spread, ts_ms, mid, out)

        # prev 无论是否 throttled 都必须更新，否则 throttle 释放后会产生幽灵穿越
        ps.prev_entry = entry_spread
        ps.prev_exit = exit_spread

    def _process_tick(self, tick: dict[str, Any]) -> None:
        if tick.get("type") != "tick":
            return
        out: list[dict[str, Any]] = []
        with self._lock:
            for ps in self._pairs.values():
                self._update_leg_from_tick(ps, tick)
            for ps in self._pairs.values():
                self._process_pair_tick(ps, out)
        for sig in out:
            try:
                self._signal_q.put(sig, timeout=1.0)
            except queue.Full:
                log.critical(
                    "[CRITICAL] signal_q 满导致信号丢弃: type=%s pair=%s grid=%s side=%s",
                    sig.get("type"),
                    sig.get("pair"),
                    sig.get("grid_level"),
                    sig.get("side"),
                )
                if sig.get("type") == "close":
                    pair = str(sig.get("pair", ""))
                    try:
                        g = _norm_grid(float(sig.get("grid_level", 0)))
                    except (TypeError, ValueError):
                        continue
                    with self._lock:
                        ps = self._pairs.get(pair)
                        if ps is not None:
                            rb = ps.close_rollbacks.pop(g, CloseLockRollback())
                            self._rollback_close_locks(ps, rb)
                            ps.pending_close_ts.pop(g, None)
                            pos = ps.positions.get(g)
                            if pos is not None:
                                pos["closing"] = False
                            log.warning(
                                "[ROLLBACK] 已回滚 close 锁定 pair=%s grid=%s",
                                pair,
                                g,
                            )
        self._check_pending_timeouts()

    def _check_pending_timeouts(self) -> None:
        now = time.time()
        with self._lock:
            for pname, ps in self._pairs.items():
                for g, t0 in list(ps.pending_open_ts.items()):
                    if now - t0 > _PENDING_TIMEOUT_SEC:
                        key = (pname, g, "open")
                        if key not in self._pending_timeout_cr_logged:
                            self._pending_timeout_cr_logged.add(key)
                            meta = ps.pending_open_meta.get(g, {})
                            side = str(meta.get("side", "LONG"))
                            log.critical(
                                "[CRITICAL] pair=%s grid=%s pending=OPEN side=%s 超过 %.0fs 未得到回报，"
                                "保留 pending；请人工核对后可用 load_positions_from_exchange 等修正状态",
                                pname,
                                g,
                                side,
                                _PENDING_TIMEOUT_SEC,
                            )
                for g, t0 in list(ps.pending_close_ts.items()):
                    if now - t0 > _PENDING_TIMEOUT_SEC:
                        key = (pname, g, "close")
                        if key not in self._pending_timeout_cr_logged:
                            self._pending_timeout_cr_logged.add(key)
                            pos = ps.positions.get(g, {})
                            side = str(pos.get("side", "LONG"))
                            log.critical(
                                "[CRITICAL] pair=%s grid=%s pending=CLOSE side=%s 超过 %.0fs 未得到回报，"
                                "保留 pending；请人工核对后可用 load_positions_from_exchange 等修正状态",
                                pname,
                                g,
                                side,
                                _PENDING_TIMEOUT_SEC,
                            )

    def on_tick(self, tick: dict) -> None:
        self._tick_q.put(tick)

    def set_pending_open_audit(
        self, pair: str, grid_level: float, open_fee: float, open_time: str
    ) -> None:
        """成交写入 trades.jsonl 后由执行器调用，把 open_fee / open_time 写入 pending meta，on_fill 时进持仓。"""
        g = _norm_grid(grid_level)
        with self._lock:
            ps = self._pairs.get(pair)
            if ps is None or g not in ps.pending_open_meta:
                return
            ps.pending_open_meta[g]["open_fee"] = float(open_fee)
            ps.pending_open_meta[g]["open_time"] = str(open_time)

    def register_manual_open_pending(
        self,
        pair_name: str,
        side: str,
        grid_level: float,
        entry_spread: float,
    ) -> None:
        """初始建仓：不经信号队列注册 pending 开仓，供 OrderExecutor._execute_open 与 on_fill 对齐。"""
        g = _norm_grid(grid_level)
        with self._lock:
            ps = self._pairs.get(pair_name)
            if ps is None:
                return
            c = ps.config
            step = c.grid_step if c.grid_step > 0 else 0.0
            if side == "LONG":
                tp = g + step
            else:
                tp = g - step
            leg1_sz = int(c.qty) * _leg_contracts_per_unit(c.leg1)
            leg2_sz = int(c.qty) * _leg_contracts_per_unit(c.leg2)
            meta = {
                "side": side,
                "entry_spread": float(entry_spread),
                "take_profit_spread": float(tp),
                "qty": c.qty,
                "leg1_size": leg1_sz,
                "leg2_size": leg2_sz,
            }
            ps.pending_open_meta[g] = meta
            ps.pending_open_ts[g] = time.time()

    def bootstrap_pair_after_init(
        self,
        pair_name: str,
        entry_spread: float,
        exit_spread: float,
        quotes: Dict[str, Dict[str, Any]],
    ) -> None:
        """初始建仓完成且已回放行情前：同步两腿盘口与 prev 价差，避免首 tick 误判网格穿越。"""
        with self._lock:
            ps = self._pairs.get(pair_name)
            if ps is None:
                return
            c = ps.config
            for sym, leg in ((c.leg1, "leg1"), (c.leg2, "leg2")):
                q = quotes.get(sym)
                if not q:
                    continue
                bid = float(q["bid"])
                ask = float(q["ask"])
                ms_t = int(q["ms_t"])
                if leg == "leg1":
                    ps.leg1_bid = bid
                    ps.leg1_ask = ask
                    ps.leg1_ms_t = ms_t
                else:
                    ps.leg2_bid = bid
                    ps.leg2_ask = ask
                    ps.leg2_ms_t = ms_t
            ps.prev_entry = float(entry_spread)
            ps.prev_exit = float(exit_spread)
            ps.tick_count = 2
            ps.is_initialized = True

    def on_fill(
        self,
        pair: str,
        grid_level: float,
        signal_type: str,
        side: str,
        status: str,
    ) -> None:
        g = _norm_grid(grid_level)
        st = status.lower()
        if st == "partial":
            st = "failed"
        with self._lock:
            ps = self._pairs.get(pair)
            if ps is None:
                return
            ts_ms = int(time.time() * 1000)
            c = ps.config
            if signal_type == "open":
                if g not in ps.pending_open_ts:
                    return
                if st == "filled":
                    ps.pending_open_ts.pop(g, None)
                    self._pending_timeout_cr_logged.discard((pair, g, "open"))
                    meta = ps.pending_open_meta.pop(g, {})
                    ps.positions[g] = {
                        "side": meta.get("side", side),
                        "qty": meta.get("qty", c.qty),
                        "leg1_size": int(
                            meta.get(
                                "leg1_size",
                                int(meta.get("qty", c.qty))
                                * _leg_contracts_per_unit(c.leg1),
                            )
                        ),
                        "leg2_size": int(
                            meta.get(
                                "leg2_size",
                                int(meta.get("qty", c.qty))
                                * _leg_contracts_per_unit(c.leg2),
                            )
                        ),
                        "entry_spread": float(meta.get("entry_spread", 0)),
                        "take_profit_spread": float(meta.get("take_profit_spread", 0)),
                        "closing": False,
                        "open_time": str(
                            meta.get("open_time")
                            or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ),
                        "open_fee": float(meta.get("open_fee", 0.0)),
                    }
                    log.info(
                        "[FILL   %s] %s OPEN  %s  grid=%s  status=filled",
                        _ts_str(ts_ms),
                        pair,
                        side,
                        g,
                    )
                    self._save_pair(pair, ps)
                else:
                    ps.pending_open_ts.pop(g, None)
                    self._pending_timeout_cr_logged.discard((pair, g, "open"))
                    ps.pending_open_meta.pop(g, None)
                    log.info(
                        "[FILL   %s] %s OPEN  %s  grid=%s  status=failed",
                        _ts_str(ts_ms),
                        pair,
                        side,
                        g,
                    )
                    self._save_pair(pair, ps)
            elif signal_type == "close":
                if g not in ps.pending_close_ts:
                    return
                if st == "filled":
                    rb = ps.close_rollbacks.pop(g, CloseLockRollback())
                    ps.pending_close_ts.pop(g, None)
                    self._pending_timeout_cr_logged.discard((pair, g, "close"))
                    ps.positions.pop(g, None)
                    log.info(
                        "[FILL   %s] %s CLOSE %s  grid=%s  status=filled",
                        _ts_str(ts_ms),
                        pair,
                        side,
                        g,
                    )
                    self._save_pair(pair, ps)
                else:
                    rb = ps.close_rollbacks.pop(g, CloseLockRollback())
                    ps.pending_close_ts.pop(g, None)
                    self._pending_timeout_cr_logged.discard((pair, g, "close"))
                    if not ps.cb_tripped:
                        self._rollback_close_locks(ps, rb)
                    pos = ps.positions.get(g)
                    if pos is not None:
                        pos["closing"] = False
                    log.info(
                        "[FILL   %s] %s CLOSE %s  grid=%s  status=failed",
                        _ts_str(ts_ms),
                        pair,
                        side,
                        g,
                    )
                    self._save_pair(pair, ps)
            else:
                return

    def get_state_snapshot(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {}
            for name, ps in self._pairs.items():
                out[name] = {
                    "config": asdict(ps.config),
                    "is_initialized": ps.is_initialized,
                    "prev_entry": ps.prev_entry,
                    "prev_exit": ps.prev_exit,
                    "positions": {str(k): deepcopy(v) for k, v in ps.positions.items()},
                    "pending_open_ts": dict(ps.pending_open_ts),
                    "pending_close_ts": dict(ps.pending_close_ts),
                    "signal_timestamps_count": len(ps.signal_timestamps),
                    "locked_long_levels": dict(ps.locked_long_levels),
                    "locked_short_levels": dict(ps.locked_short_levels),
                    "locked_reopen_long_levels": dict(ps.locked_reopen_long_levels),
                    "locked_reopen_short_levels": dict(ps.locked_reopen_short_levels),
                    "locked_tp_both": dict(ps.locked_tp_both),
                }
            return out

    def load_positions_from_exchange(self, pair: str, positions: dict) -> None:
        """
        用交易所真实持仓覆盖本地 positions。
        positions: grid_level -> 仓位信息 dict（与内部 positions 结构一致，key 可为 float 或 str）。
        """
        with self._lock:
            ps = self._pairs.get(pair)
            if ps is None:
                return
            newp: dict[float, dict[str, Any]] = {}
            for k, v in positions.items():
                kk = float(k) if not isinstance(k, float) else k
                newp[_norm_grid(kk)] = deepcopy(v) if isinstance(v, dict) else v
            ps.positions = newp

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                tick = self._tick_q.get(timeout=0.25)
            except queue.Empty:
                continue
            self._process_tick(tick)

    def _saver_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(5.0)
            if self._stop.is_set():
                break
            self._save_all_states()

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, name="SpreadGridWorker", daemon=True)
        self._saver = threading.Thread(target=self._saver_loop, name="SpreadGridSaver", daemon=True)
        self._worker.start()
        self._saver.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        if self._saver is not None:
            self._saver.join(timeout=2.0)
        self._save_all_states()
