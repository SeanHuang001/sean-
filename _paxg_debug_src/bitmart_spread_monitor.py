# pip install websocket-client
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from typing import Optional

import websocket

log = logging.getLogger(__name__)

WS_URL = "wss://openapi-ws-v2.bitmart.com/api?protocol=1.1"
SUBSCRIBE_MSG = json.dumps(
    {
        "action": "subscribe",
        "args": [
            "futures/bookticker:XAUUSDT",
            "futures/bookticker:XAUTUSDT",
            "futures/bookticker:PAXGUSDT",
            "futures/bookticker:XTIUSDT",
            "futures/bookticker:XBRUSDT",
        ],
    }
)
SYMBOLS = (
    "XAUUSDT",
    "XAUTUSDT",
    "PAXGUSDT",
    "XTIUSDT",
    "XBRUSDT",
)
SILENCE_SEC = 5.0
_RAW_TICKS_PATH = Path("raw_ticks.jsonl")
_raw_ticks_lock = threading.Lock()
_RAW_TICK_SYMBOLS = frozenset({"PAXGUSDT", "XAUTUSDT"})
PING_INTERVAL = 15.0
RECONNECT_DELAY = 3.0
UI_POLL_MS = 50

BG = "#1e1e1e"
FG = "#d4d4d4"
BID_COLOR = "#3ecf8e"
ASK_COLOR = "#f14c4c"
POS_COLOR = "#3ecf8e"
NEG_COLOR = "#f14c4c"
FONT_MONO = ("Consolas", "Courier New")


class BitMartFeed(threading.Thread):
    def __init__(self, out_q: queue.Queue, stop: threading.Event):
        super().__init__(daemon=True)
        self._out_q = out_q
        self._stop = stop
        self._last_recv = [time.monotonic()]
        self._recv_lock = threading.Lock()
        self._gen = 0

    def _touch_recv(self) -> None:
        with self._recv_lock:
            self._last_recv[0] = time.monotonic()

    def _silence_seconds(self) -> float:
        with self._recv_lock:
            return time.monotonic() - self._last_recv[0]

    def _put_tick(self, symbol: str, bid: float, ask: float, ms_t: int) -> None:
        self._out_q.put({"type": "tick", "symbol": symbol, "bid": bid, "ask": ask, "ms_t": ms_t})

    def _on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        self._touch_recv()
        if message == "pong":
            return
        try:
            obj = json.loads(message)
        except json.JSONDecodeError:
            return
        data = obj.get("data")
        if not isinstance(data, dict):
            return
        sym = data.get("symbol")
        if sym not in SYMBOLS:
            return
        try:
            bid = float(data["best_bid_price"])
            ask = float(data["best_ask_price"])
        except (KeyError, TypeError, ValueError):
            return
        ms_t = data.get("ms_t")
        if ms_t is None:
            ms_t = int(time.time() * 1000)
        else:
            try:
                ms_t = int(ms_t)
            except (TypeError, ValueError):
                ms_t = int(time.time() * 1000)
        self._put_tick(sym, bid, ask, ms_t)
        if sym in _RAW_TICK_SYMBOLS:
            wall_ms = int(time.time() * 1000)
            rec = {
                "wall_ms": wall_ms,
                "symbol": sym,
                "bid": bid,
                "ask": ask,
                "server_ms_t": ms_t,
                "raw": message,
            }
            try:
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                with _raw_ticks_lock:
                    with open(_RAW_TICKS_PATH, "a", encoding="utf-8") as f:
                        f.write(line)
            except Exception as e:
                log.debug("raw_ticks.jsonl 写入失败: %s", e)

    def _on_open(self, ws: websocket.WebSocketApp) -> None:
        self._touch_recv()
        try:
            ws.send(SUBSCRIBE_MSG)
        except Exception:
            pass
        self._gen += 1
        gen = self._gen

        def pinger() -> None:
            while not self._stop.is_set() and gen == self._gen:
                time.sleep(PING_INTERVAL)
                if self._stop.is_set() or gen != self._gen:
                    break
                try:
                    ws.send("ping")
                except Exception:
                    break

        def watchdog() -> None:
            while not self._stop.is_set() and gen == self._gen:
                time.sleep(0.5)
                if self._stop.is_set() or gen != self._gen:
                    break
                if self._silence_seconds() > SILENCE_SEC:
                    try:
                        ws.close()
                    except Exception:
                        pass
                    break

        threading.Thread(target=pinger, daemon=True).start()
        threading.Thread(target=watchdog, daemon=True).start()

    def _on_error(self, ws: websocket.WebSocketApp, error) -> None:
        log.warning("[WS_ERROR] %s", error)

    def _on_close(self, ws: websocket.WebSocketApp, code, msg) -> None:
        self._gen += 1

    def run(self) -> None:
        reconnect_fails = 0
        while not self._stop.is_set():
            try:
                ws_app = websocket.WebSocketApp(
                    WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                ws_app.run_forever(ping_interval=0, ping_timeout=None)
                reconnect_fails = 0
            except Exception as e:
                log.warning("[WS_RECONNECT] exception: %s", e)
                reconnect_fails += 1
            if self._stop.is_set():
                break
            delay = min(RECONNECT_DELAY * (2 ** min(reconnect_fails, 5)), 60.0)
            time.sleep(delay)


def fmt_price(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def fmt_spread(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:+.4f}"


def fmt_time_ms(ms: Optional[int]) -> str:
    if ms is None:
        return "—"
    try:
        lt = time.localtime(ms / 1000.0)
        return time.strftime("%H:%M:%S", lt) + f".{ms % 1000:03d}"
    except (OSError, OverflowError, ValueError):
        return "—"


class SpreadMonitorApp:
    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._quotes: dict[str, dict] = {
            s: {"bid": None, "ask": None, "ms_t": None} for s in SYMBOLS
        }
        self._got_data = False

        self.root = tk.Tk()
        self.root.title("BitMart 合约盘口监控")
        self.root.configure(bg=BG)

        mono_large = tkfont.Font(family=FONT_MONO[0], size=20, weight="bold")
        mono_mid = tkfont.Font(family=FONT_MONO[0], size=13)
        mono_small = tkfont.Font(family=FONT_MONO[0], size=11)
        title_font = tkfont.Font(family=FONT_MONO[0], size=12, weight="bold")

        main = tk.Frame(self.root, bg=BG, padx=12, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        status_frame = tk.Frame(main, bg=BG)
        status_frame.pack(fill=tk.X, pady=(0, 8))
        self.status_lbl = tk.Label(
            status_frame,
            text="连接中...",
            fg="#e0c36a",
            bg=BG,
            font=mono_mid,
        )
        self.status_lbl.pack(anchor=tk.W)

        top = tk.Frame(main, bg=BG)
        top.pack(fill=tk.X, pady=(0, 16))

        self._bid_labels: dict[str, tk.Label] = {}
        self._ask_labels: dict[str, tk.Label] = {}
        self._time_labels: dict[str, tk.Label] = {}

        row1_syms = SYMBOLS[:3]
        row2_syms = SYMBOLS[3:]
        for col in range(4):
            top.columnconfigure(col, weight=1)

        def _add_price_card(parent: tk.Frame, row: int, col: int, sym: str) -> None:
            f = tk.Frame(parent, bg="#2d2d2d", padx=14, pady=12)
            f.grid(row=row, column=col, padx=6, pady=(0, 6), sticky=tk.NSEW)

            tk.Label(f, text=sym, fg=FG, bg="#2d2d2d", font=title_font).pack(anchor=tk.W)
            tk.Label(f, text="买一 (bid)", fg="#888", bg="#2d2d2d", font=mono_small).pack(anchor=tk.W, pady=(8, 0))
            bl = tk.Label(f, text="—", fg=BID_COLOR, bg="#2d2d2d", font=mono_large)
            bl.pack(anchor=tk.W)
            self._bid_labels[sym] = bl
            tk.Label(f, text="卖一 (ask)", fg="#888", bg="#2d2d2d", font=mono_small).pack(anchor=tk.W, pady=(6, 0))
            al = tk.Label(f, text="—", fg=ASK_COLOR, bg="#2d2d2d", font=mono_large)
            al.pack(anchor=tk.W)
            self._ask_labels[sym] = al
            tk.Label(f, text="最后更新", fg="#888", bg="#2d2d2d", font=mono_small).pack(anchor=tk.W, pady=(8, 0))
            tl = tk.Label(f, text="—", fg=FG, bg="#2d2d2d", font=mono_mid)
            tl.pack(anchor=tk.W)
            self._time_labels[sym] = tl

        for col, sym in enumerate(row1_syms):
            _add_price_card(top, 0, col, sym)
        for col, sym in enumerate(row2_syms):
            _add_price_card(top, 1, col, sym)

        spread_frame = tk.LabelFrame(
            main,
            text="价差组合",
            fg=FG,
            bg=BG,
            font=mono_small,
            padx=10,
            pady=10,
        )
        spread_frame.pack(fill=tk.BOTH, expand=True)

        spread_defs = [
            ("XAU买一 - XAUT卖一", "xau_bid_xaut_ask"),
            ("XAU卖一 - XAUT买一", "xau_ask_xaut_bid"),
            ("XAU买一 - PAXG卖一", "xau_bid_paxg_ask"),
            ("XAU卖一 - PAXG买一", "xau_ask_paxg_bid"),
            ("PAXG买一 - XAUT卖一", "paxg_bid_xaut_ask"),
            ("PAXG卖一 - XAUT买一", "paxg_ask_xaut_bid"),
            ("XCU买一 - COPPER卖一", "xcu_bid_copper_ask"),
            ("XCU卖一 - COPPER买一", "xcu_ask_copper_bid"),
            ("XTI买一 - XBR卖一", "xti_bid_xbr_ask"),
            ("XTI卖一 - XBR买一", "xti_ask_xbr_bid"),
        ]
        self._spread_labels: dict[str, tk.Label] = {}
        for name, key in spread_defs:
            row = tk.Frame(spread_frame, bg=BG)
            row.pack(fill=tk.X, pady=4)
            tk.Label(row, text=name, fg=FG, bg=BG, font=mono_mid, width=28, anchor=tk.W).pack(
                side=tk.LEFT
            )
            sl = tk.Label(row, text="—", fg=FG, bg=BG, font=mono_mid, anchor=tk.E)
            sl.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            self._spread_labels[key] = sl

        self._feed = BitMartFeed(self._q, self._stop)
        self._feed.start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll()

    def _drain_queue(self) -> None:
        try:
            while True:
                item = self._q.get_nowait()
                if item.get("type") != "tick":
                    continue
                sym = item["symbol"]
                if sym in self._quotes:
                    self._quotes[sym]["bid"] = item["bid"]
                    self._quotes[sym]["ask"] = item["ask"]
                    self._quotes[sym]["ms_t"] = item["ms_t"]
                    self._got_data = True
        except queue.Empty:
            pass

    def _compute_spreads(self) -> dict[str, Optional[float]]:
        x = self._quotes["XAUUSDT"]
        u = self._quotes["XAUTUSDT"]
        p = self._quotes["PAXGUSDT"]
        xcu = self._quotes["XCUUSDT"]
        copper = self._quotes["COPPERUSDT"]
        xti = self._quotes["XTIUSDT"]
        xbr = self._quotes["XBRUSDT"]

        def sub(a, b):
            if a is None or b is None:
                return None
            return a - b

        xcu_bid_copper_ask = sub(xcu["bid"], copper["ask"])
        xcu_ask_copper_bid = sub(xcu["ask"], copper["bid"])
        xti_bid_xbr_ask = sub(xti["bid"], xbr["ask"])
        xti_ask_xbr_bid = sub(xti["ask"], xbr["bid"])

        return {
            "xau_bid_xaut_ask": sub(x["bid"], u["ask"]),
            "xau_ask_xaut_bid": sub(x["ask"], u["bid"]),
            "xau_bid_paxg_ask": sub(x["bid"], p["ask"]),
            "xau_ask_paxg_bid": sub(x["ask"], p["bid"]),
            "paxg_bid_xaut_ask": sub(p["bid"], u["ask"]),
            "paxg_ask_xaut_bid": sub(p["ask"], u["bid"]),
            "xcu_bid_copper_ask": xcu_bid_copper_ask,
            "xcu_ask_copper_bid": xcu_ask_copper_bid,
            "xti_bid_xbr_ask": xti_bid_xbr_ask,
            "xti_ask_xbr_bid": xti_ask_xbr_bid,
        }

    def _refresh(self) -> None:
        if self._got_data:
            self.status_lbl.config(text="已连接", fg=BID_COLOR)

        for sym in SYMBOLS:
            q = self._quotes[sym]
            self._bid_labels[sym].config(text=fmt_price(q["bid"]))
            self._ask_labels[sym].config(text=fmt_price(q["ask"]))
            self._time_labels[sym].config(text=fmt_time_ms(q["ms_t"]))

        spreads = self._compute_spreads()
        for key, val in spreads.items():
            lbl = self._spread_labels[key]
            if val is None:
                lbl.config(text="—", fg=FG)
            else:
                c = POS_COLOR if val >= 0 else NEG_COLOR
                lbl.config(text=fmt_spread(val), fg=c)

    def _poll(self) -> None:
        self._drain_queue()
        self._refresh()
        self.root.after(UI_POLL_MS, self._poll)

    def _on_close(self) -> None:
        self._stop.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    SpreadMonitorApp().run()


if __name__ == "__main__":
    main()
