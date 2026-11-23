"""RegimeDip, stricter-regime variant — a long-only CBBTC dip/rebound bot.

The bot watches one-minute CBBTC bars and takes at most four full-size
long trades per window: up to two dip buys after a sharp but contained
pullback with an oversold RSI, and up to two rebound buys after a deeper
washout that shows a long lower wick and a strong bounce. Every trade
carries a stop and either a take-profit or a trailing exit, and a 60-bar
cooldown follows every exit. The regime filter only lets dip trades in
while the 360-bar trend is no worse than -50 bps, so the bot sits out
falling knives it would otherwise catch.

Calibration note: the dip thresholds are set for this market's calmer
one-minute bars (the dev window's sharpest 60-bar pullback from its high
is 184 bps and its deepest 60-bar net fall is 134 bps, so v1d's 150 bp
net-fall bar would never fire here): a pullback of at least 100 bps from
the 60-bar high, a 60-bar net fall of at least 60 bps, and RSI(14) below
40, with the reversal. The regime stays at the stricter -50 bps.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import makefun_bars

SYMBOL = "CBBTC"
BPS = 10_000
RSI_LEN = 14
MAX_TRADES = 4
MAX_DIP = 2
MAX_REBOUND = 2
COOLDOWN = 60
MIN_CLOSES = 62
LAST_ENTRY_SEQ = 1320
SLIP_BPS = 200
REGIME_BPS = -50
EQUITY_FLOOR_BPS = 98


class Trader:
    """All of the bot's state: price history, RSI, open trade and counters."""

    def __init__(self):
        self.started = False
        self.cash0 = 0
        self.closes = []
        self.highs = []
        self.lows = []
        self.rsi = []
        self._gain = 0.0
        self._loss = 0.0
        self.seen_t = set()
        self.trade = None
        self.pending = None
        self.dip_n = 0
        self.rb_n = 0
        self.rb_losses = 0
        self.entries = 0
        self.rest_until = -1

    def start(self, state):
        if not self.started:
            self.cash0 = state.cash
            self.started = True

    # -- history ---------------------------------------------------------

    def push_bar(self, bar):
        self.closes.append(int(bar["c"]))
        self.highs.append(int(bar["h"]))
        self.lows.append(int(bar["l"]))
        n = len(self.closes)
        if n < RSI_LEN + 1:
            self.rsi.append(None)
        elif n == RSI_LEN + 1:
            gains = 0.0
            losses = 0.0
            for i in range(1, RSI_LEN + 1):
                d = self.closes[i] - self.closes[i - 1]
                if d >= 0:
                    gains += d
                else:
                    losses -= d
            self._gain = gains / RSI_LEN
            self._loss = losses / RSI_LEN
            self.rsi.append(self._rsi_value())
        else:
            d = self.closes[-1] - self.closes[-2]
            g = float(d) if d > 0 else 0.0
            lo = float(-d) if d < 0 else 0.0
            self._gain = (self._gain * (RSI_LEN - 1) + g) / RSI_LEN
            self._loss = (self._loss * (RSI_LEN - 1) + lo) / RSI_LEN
            self.rsi.append(self._rsi_value())

    def _rsi_value(self):
        if self._loss == 0:
            return 100.0
        return 100.0 - 100.0 / (1.0 + self._gain / self._loss)

    # -- fall helpers ----------------------------------------------------

    def top_fall(self, span):
        """Fall in bps from the highest high of the last `span` bars to the
        previous close; 0 when there is not enough history."""
        if len(self.highs) <= span:
            return 0
        top = max(self.highs[-span - 1:-1])
        prev = self.closes[-2]
        return (top - prev) * BPS // top

    def close_fall(self, span):
        """Close-to-close fall in bps over `span` bars ending at the
        previous close; 0 when there is not enough history."""
        if len(self.closes) <= span + 1:
            return 0
        old = self.closes[-span - 2]
        prev = self.closes[-2]
        return (old - prev) * BPS // old

    def has_wick(self):
        """True when some bar in the last 20 has a range of at least 20 bps
        and closed in the top 40% of its own range."""
        n = len(self.closes)
        start = n - 20 if n >= 20 else 0
        for i in range(start, n):
            hi = self.highs[i]
            lo = self.lows[i]
            c = self.closes[i]
            if hi <= lo:
                continue
            if (hi - lo) * BPS // c < 20:
                continue
            if (c - lo) * 100 // (hi - lo) >= 60:
                return True
        return False

    # -- setups ----------------------------------------------------------

    def dip_setup(self):
        if self.dip_n >= MAX_DIP:
            return None
        if self.top_fall(60) < 100:
            return None
        if self.top_fall(720) >= 550:
            return None
        r = self.rsi
        if r[-1] is None or r[-2] is None or r[-1] >= 40:
            return None
        if self.closes[-1] <= self.closes[-2]:
            return None
        if self.close_fall(60) < 60:
            return None
        if len(self.closes) >= 361:
            regime = (self.closes[-1] - self.closes[-361]) * BPS // self.closes[-361]
            if regime < REGIME_BPS:
                return None
        cur = self.closes[-1]
        return {
            "kind": "dip",
            "stop": cur * (BPS - 60) // BPS,
            "take": cur * (BPS + 120) // BPS,
            "arm": 0,
            "trail": 0,
            "hold": 300,
        }

    def rebound_setup(self):
        if self.rb_n >= MAX_REBOUND or self.rb_losses >= 2:
            return None
        if self.close_fall(60) < 150 and self.close_fall(120) < 250:
            return None
        if self.top_fall(720) < 550:
            return None
        if not self.has_wick():
            return None
        cur = self.closes[-1]
        prev = self.closes[-2]
        if (cur - prev) * BPS // prev < 60:
            return None
        if cur <= self.highs[-2]:
            return None
        r = self.rsi
        if r[-1] is None or r[-2] is None or r[-1] <= r[-2]:
            return None
        low720 = min(self.lows[-721:-1])
        stop0 = low720 * (BPS - 20) // BPS
        lo_stop = cur * (BPS - 100) // BPS
        hi_stop = cur * (BPS - 50) // BPS
        stop = min(max(stop0, lo_stop), hi_stop)
        top720 = max(self.highs[-721:-1])
        if top720 - cur < 2 * (cur - stop):
            return None
        return {
            "kind": "rebound",
            "stop": stop,
            "take": None,
            "arm": 80,
            "trail": 80,
            "hold": 300,
        }

    # -- trade lifecycle -------------------------------------------------

    def finalize(self):
        t = self.trade
        if t is not None and t.get("exit_pending") and t["kind"] == "rebound":
            if t["exit_px"] > t["entry"]:
                self.rb_losses = 0
            else:
                self.rb_losses += 1
        self.trade = None

    def manage(self, state, bar):
        t = self.trade
        close = int(bar["c"])
        t["held"] += 1
        if close > t["peak"]:
            t["peak"] = close
        if not t["armed"] and close >= t["entry"] * (BPS + t["arm"]) // BPS:
            t["armed"] = True
        px = None
        if t["kind"] == "dip":
            if close >= t["take"]:
                px = close
            elif close <= t["stop"]:
                px = t["stop"]
            elif t["held"] >= t["hold"]:
                px = close
        else:
            if close <= t["stop"]:
                px = t["stop"]
            elif t["armed"] and close <= t["peak"] * (BPS - t["trail"]) // BPS:
                px = t["peak"] * (BPS - t["trail"]) // BPS
            elif t["held"] >= t["hold"]:
                px = close
        if px is None:
            return []
        qty = state.positions[SYMBOL].qty
        t["exit_px"] = px
        t["exit_pending"] = True
        self.rest_until = state.seq + COOLDOWN
        return [{"side": "sell", "symbol": SYMBOL, "qty": qty,
                 "max_slippage_bps": SLIP_BPS}]


TRADER = Trader()


def on_hello(state):
    TRADER.start(state)


def on_batch(state, batch):
    if not TRADER.started:
        TRADER.start(state)
    bar = None
    for b in batch.get("bars", []):
        if b.get("symbol") == SYMBOL:
            bar = b
            break
    if bar is None:
        return []
    t = batch.get("t")
    if t in TRADER.seen_t:
        return []
    TRADER.seen_t.add(t)
    TRADER.push_bar(bar)
    # A buy placed on the previous batch fills at this bar's open.
    if TRADER.pending is not None:
        pend = TRADER.pending
        TRADER.pending = None
        holding = state.positions.get(SYMBOL)
        if holding is not None:
            TRADER.trade = dict(pend)
            TRADER.trade["entry"] = int(bar["o"])
            TRADER.trade["held"] = 0
            TRADER.trade["peak"] = int(bar["o"])
            TRADER.trade["armed"] = False
            TRADER.trade["exit_pending"] = False
            TRADER.trade["exit_px"] = None
            if pend["kind"] == "dip":
                TRADER.dip_n += 1
            else:
                TRADER.rb_n += 1
            TRADER.entries += 1
    # An open trade: manage it, or clear it once the position is gone.
    if TRADER.trade is not None:
        holding = state.positions.get(SYMBOL)
        if holding is None:
            TRADER.finalize()
            return []
        return TRADER.manage(state, bar)
    # No trade open: guards, then rebound first, then dip.
    if state.warming_up:
        return []
    if len(TRADER.closes) < MIN_CLOSES:
        return []
    if state.seq > LAST_ENTRY_SEQ:
        return []
    if TRADER.entries >= MAX_TRADES:
        return []
    if state.seq < TRADER.rest_until:
        return []
    if state.equity < TRADER.cash0 * EQUITY_FLOOR_BPS // 100:
        return []
    setup = TRADER.rebound_setup() or TRADER.dip_setup()
    if setup is None:
        return []
    TRADER.pending = setup
    return [{"side": "buy", "symbol": SYMBOL,
             "notional_quote_units": state.cash,
             "max_slippage_bps": SLIP_BPS}]


if __name__ == "__main__":
    makefun_bars.run(on_batch, on_hello=on_hello)
