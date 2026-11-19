"""Data access and strategy pricing for the Strategy Analyzer (DuckDB on app_data/).

ADD OR CHANGE STRATEGIES IN THE `STRATEGIES` LIST BELOW - nothing else needs editing.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

STRIKE_GAP = 50          # default strike interval; each data set's own gap comes from its meta.json

# =============================================================================
# STRATEGIES
# -----------------------------------------------------------------------------
# Each leg = (quantity, side, distance)  or  (quantity, side, distance, expiry)
#   quantity : +buy / -sell, e.g. +1, -2, -3, +2
#   side     : "X"  = the side you click in the table (CE column or PE column)
#              "CE" / "PE" = always that side (for mixed strategies like a straddle)
#   distance : from the centre strike, in WING units (wing width is chosen in the sidebar,
#              50 by default): -1 = K - wing, 0 = K, +2 = K + 2 x wing, 0.5 = K + wing/2 ...
#   expiry   : "near" (default) = the expiry picked by Days to expiry (near week)
#              "w1" = next week (the 1st expiry after near), "w2" = far week (2nd after), "w3" = 3rd after
#              - all from the same data set;
#              "far" = a later expiry chosen by a rule (and optionally from another data set, e.g. monthly):
#                "next_month"   = the monthly expiry of the month after the near expiry's month
#                "next_monthly" = the first monthly expiry after the near expiry
#                "next_weekly"  = the very next expiry after the near expiry
#              (in the app you can also pick the far leg yourself: week N of a chosen month,
#               or N weeks after the near expiry)
#              (far legs need those option prices to be in your parquet files)
# mirror_pe  : True -> for the PE side the distances are flipped (+ becomes -), so a put
#              version is the mirror image of the call version (e.g. ratio legs go downwards).
# Price shown = sum(quantity x leg price): positive = net debit (you pay), negative = net credit.
# Optional "units": "points" -> distances are in points instead of wing units (used by custom strategies).
# You can also build strategies on screen ("Custom (build your own)") and save them - they are stored in
# custom_strategies.json next to the app, so you don't have to edit this file.
# =============================================================================
STRATEGIES = {
    "Butterfly 1-2-1": {
        "legs": [(+1, "X", -1), (-2, "X", 0), (+1, "X", +1)],
        "mirror_pe": True,
        "about": "Buy 1 lower, sell 2 centre, buy 1 upper",
    },
    "Ratio 1-3-2": {
        "legs": [(+1, "X", -1), (-3, "X", 0), (+2, "X", +1)],
        "mirror_pe": True,
        "about": "Buy 1 in-the-money side, sell 3 centre, buy 2 out-of-the-money side",
    },
    "Straddle": {
        "legs": [(+1, "CE", 0), (+1, "PE", 0)],
        "about": "Buy 1 CE + 1 PE at the same strike",
    },
    "Strangle": {
        "legs": [(+1, "CE", +1), (+1, "PE", -1)],
        "about": "Buy 1 CE one wing above + 1 PE one wing below the centre strike",
    },
    "Iron Butterfly": {
        "legs": [(-1, "CE", 0), (-1, "PE", 0), (+1, "CE", +1), (+1, "PE", -1)],
        "about": "Sell ATM straddle, buy wings (negative price = credit received)",
    },
    "Single Option": {
        "legs": [(+1, "X", 0)],
        "about": "Just the CE or PE at that strike",
    },
    "Calendar (sell near, buy next week)": {
        "legs": [(-1, "X", 0, "near"), (+1, "X", 0, "w1")],
        "about": "Sell the near-week option, buy the same strike in next week's expiry",
    },
    "Week calendar ATM±2 (buy near, sell next week)": {
        "legs": [(+1, "CE", +2, "near"), (-1, "CE", +2, "w1"), (+1, "PE", -2, "near"), (-1, "PE", -2, "w1")],
        "units": "steps",
        "about": "Buy near-week CE 2 strikes up + sell next-week CE same strike; buy near-week PE 2 strikes down "
                 "+ sell next-week PE same strike",
    },
    "Calendar CE/PE (near vs next month)": {
        "legs": [(-1, "CE", 0, "near"), (+1, "CE", 0, "far"), (+1, "PE", 0, "near"), (-1, "PE", 0, "far")],
        "far_expiry": "next_month",
        "about": "Sell near CE + buy next-month CE, buy near PE + sell next-month PE, all at the same strike",
    },
}


def offset_label(k):
    return "ATM" if k == 0 else (f"ATM+{k}" if k > 0 else f"ATM−{-k}")


def hhmm(tod):
    return f"{int(tod) // 60:02d}:{int(tod) % 60:02d}"


def is_two_sided(name):
    """True if the strategy has a CE and a PE version (uses side 'X')."""
    return any(leg[1] == "X" for leg in get_spec(name)["legs"])


def sides_of(name):
    return ["PE", "CE"] if is_two_sided(name) else ["ALL"]


def get_spec(name):
    """A strategy by name (from STRATEGIES) - or a spec dict itself (custom strategies built in the app)."""
    return name if isinstance(name, dict) else STRATEGIES[name]


TAG_LABELS = {"near": "Near week", "w1": "Next week", "w2": "Far week (+2)", "w3": "Week +3",
              "far": "Chosen far expiry"}


def leg_tags(name):
    """Expiry tags used by a strategy, in leg order: e.g. ['near', 'w1']."""
    out = []
    for leg in get_spec(name)["legs"]:
        t = leg[3] if len(leg) > 3 else "near"
        if t not in out:
            out.append(t)
    return out


def needs_far(name):
    """True if the strategy has legs on the rule-chosen far expiry ('far' - week/month rule, other data set)."""
    return "far" in leg_tags(name)


def multi_expiry(name):
    """True if the strategy uses more than the near expiry (next week, far week, chosen far ...)."""
    return any(t != "near" for t in leg_tags(name))


def tag_rule(tag):
    """Fixed rule for the week tags: w1 = 1st expiry after near, w2 = 2nd, w3 = 3rd."""
    return ("weeks_after", int(tag[1:])) if tag.startswith("w") and tag[1:].isdigit() else None


def resolve_legs(name, side, wing, gap=STRIKE_GAP):
    """[(qty, 'CE'/'PE', offset_points, 'near'/'far')] for one side of a strategy.
    Distances: "units": "steps" -> strike steps (x the index's strike gap); "points" -> points;
    otherwise wing units (x the wing width)."""
    spec = get_spec(name)
    units = spec.get("units", "wing")
    legs = []
    for leg in spec["legs"]:
        qty, s, mult = leg[:3]
        tag = leg[3] if len(leg) > 3 else "near"
        typ = side if s == "X" else s
        off = mult * gap if units == "steps" else (mult if units == "points" else mult * wing)
        if s == "X" and side == "PE" and spec.get("mirror_pe"):
            off = -off
        legs.append((qty, typ, off, tag))
    return legs


def leg_text(q, t, strike, tag, exps=None, show_exp=False):
    txt = f"{q:+d} {t} {strike:.0f}"
    if show_exp:
        e = (exps or {}).get(tag)
        txt += f" ({pd.Timestamp(e):%d %b})" if e is not None and pd.notna(e) else f" ({TAG_LABELS.get(tag, tag)})"
    return txt


def formula_text(legs, strike, exps=None):
    """'+1 CE 23950  −3 CE 24000  +2 CE 24050'  (with each leg's expiry when the strategy uses several)"""
    many = any(tag != "near" for *_, tag in legs)
    return "  ".join(leg_text(q, t, strike + off, tag, exps, many).replace("-", "−", 1)
                     for q, t, off, tag in legs)


def find_datasets(root):
    """Built data sets under the app data folder: [{id, underlying, contract, label, path, strike_gap, lot_size}]."""
    root = Path(root)
    out = []
    if not root.is_dir():
        return out
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "days.parquet").exists() and (d / "meta.json").exists():
            m = json.loads((d / "meta.json").read_text())
            out.append({"id": m.get("id", d.name), "underlying": m.get("underlying", "NIFTY"),
                        "contract": m.get("contract", "Weekly"), "label": m.get("label", d.name),
                        "path": str(d), "strike_gap": m.get("strike_gap", STRIKE_GAP),
                        "lot_size": m.get("lot_size", 75)})
    if not out and (root / "days.parquet").exists():          # data built by an older version (one data set)
        out.append({"id": "old_build", "underlying": "NIFTY", "contract": "Weekly", "label": "NIFTY (old build)",
                    "path": str(root), "strike_gap": STRIKE_GAP, "lot_size": 75, "legacy": True})
    return out


class Engine:
    def __init__(self, data_dir, vix_dir=None):
        self.dir = Path(data_dir)
        chain = self.dir / "chain"
        if not (self.dir / "days.parquet").exists() or not any(chain.glob("*.parquet")):
            raise FileNotFoundError(f"No app data in '{self.dir}'. Run build_data.py first.")
        self.con = duckdb.connect()
        self.con.execute(f"CREATE VIEW chain_raw AS SELECT * FROM read_parquet('{(chain / '*.parquet').as_posix()}', union_by_name=true)")
        cols = set(self.con.execute("SELECT * FROM chain_raw LIMIT 0").df().columns)
        self.has_iv = {"ce_iv", "pe_iv"} <= cols          # False for data built before IV was added
        iv = "" if self.has_iv else ", CAST(NULL AS FLOAT) AS ce_iv, CAST(NULL AS FLOAT) AS pe_iv"
        self.con.execute(f"CREATE VIEW chain AS SELECT *{iv} FROM chain_raw")
        self.days = pd.read_parquet(self.dir / "days.parquet")
        self.days["date"] = pd.to_datetime(self.days["date"])
        self.days["expiry"] = pd.to_datetime(self.days["expiry"])
        # week of month of every expiry: 1 = first expiry of that calendar month, ... ; is_last = last one
        ex = pd.DataFrame({"expiry": sorted(self.days["expiry"].unique())})
        ex["ym"] = ex["expiry"].dt.to_period("M")
        ex["week_no"] = ex.groupby("ym").cumcount() + 1
        ex["is_last_week"] = ex["week_no"] == ex.groupby("ym")["week_no"].transform("max")
        self.expiries = ex
        self.days = self.days.merge(ex[["expiry", "week_no", "is_last_week"]], on="expiry", how="left")
        self.con.register("days", self.days)
        meta = self.dir / "meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        self.gap = float(self.meta.get("strike_gap", STRIKE_GAP))
        self.underlying = self.meta.get("underlying", "NIFTY")
        self.contract = self.meta.get("contract", "Weekly")
        self.first_expiry = self.days["expiry"].min()
        self.last_expiry = self.days["expiry"].max()
        self._exp_by_date = {d: g.sort_values("expiry") for d, g in self.days.groupby("date")}

        # India VIX (optional, built from the vix folder by build_data.py)
        if vix_dir is None:                                  # shared VIX folder next to the data sets
            vix_dir = self.dir.parent / "vix" if (self.dir.parent / "vix").is_dir() else self.dir
        vix_dir = Path(vix_dir)
        vd, vm = vix_dir / "vix_daily.parquet", vix_dir / "vix_minute.parquet"
        self.vix_daily = pd.read_parquet(vd) if vd.exists() else None
        self.vix_minute = pd.read_parquet(vm) if vm.exists() else None
        if self.vix_daily is not None:
            self.vix_daily["date"] = pd.to_datetime(self.vix_daily["date"])
            self.vix_daily = self.vix_daily.set_index("date").sort_index()
        if self.vix_minute is not None:
            self.vix_minute["date"] = pd.to_datetime(self.vix_minute["date"])
        self.has_vix = self.vix_daily is not None and len(self.vix_daily) > 0
        self.has_vix_minute = self.vix_minute is not None and len(self.vix_minute) > 0

    # ---------------------------------------------------------------- far expiry (calendar strategies)
    def far_expiry_for(self, date, near, rule="next_month"):
        """The later expiry used by 'far' legs on this date, or None if it is not in the data.
        rule: "next_month" / "next_monthly" / "next_weekly", or
              ("week_month", month_offset, week)  week = 1..5 or "last"  e.g. ("week_month", 1, 2) = 2nd week of next month
              ("weeks_after", n)                   the n-th expiry after the near one"""
        g = self._exp_by_date.get(pd.Timestamp(date))
        if g is None:
            return None
        near = pd.Timestamp(near)
        later = g[g["expiry"] > near]
        if isinstance(rule, (tuple, list)) and rule[0] == "weeks_after":
            # n-th expiry after near among ALL expiries in the data (so missing chains are not skipped silently)
            allx = self.expiries["expiry"]
            nxt = allx[allx > near]
            if len(nxt) < int(rule[1]):
                return None
            target = nxt.iloc[int(rule[1]) - 1]
            return target if (later["expiry"] == target).any() else None
        if isinstance(rule, (tuple, list)) and rule[0] == "week_month":
            tm = (near.to_period("M") + int(rule[1]))
            ex = self.expiries[self.expiries["ym"] == tm]
            ex = ex[ex["is_last_week"]] if str(rule[2]) == "last" else ex[ex["week_no"] == int(rule[2])]
            if ex.empty:
                return None
            target = ex["expiry"].iloc[0]
            return target if target > near and (later["expiry"] == target).any() else None
        if rule == "next_weekly":
            c = later
        elif rule == "next_monthly":
            c = later[later["is_monthly"]]
        else:                                                   # next_month
            tm = near + pd.DateOffset(months=1)
            c = later[later["is_monthly"] & (later["expiry"].dt.year == tm.year) & (later["expiry"].dt.month == tm.month)]
        return c["expiry"].iloc[0] if len(c) else None

    def snapshot_far(self, snap, name, rule=None, far_engine=None):
        """Chain rows of the far expiry at the same date and minute as each near-expiry snapshot.
        rule: see far_expiry_for (default: the strategy's "far_expiry").
        far_engine: take the far leg from another data set (e.g. the monthly data); default = this one."""
        rule = rule or get_spec(name).get("far_expiry", "next_month")
        fe = far_engine or self
        pairs = snap[["date", "expiry", "tod"]].drop_duplicates()
        req = []
        for r in pairs.itertuples():
            far = fe.far_expiry_for(r.date, r.expiry, rule)
            if far is not None:
                req.append({"date": r.date, "near": r.expiry, "far": far, "tod": int(r.tod)})
        cols = ["date", "expiry", "far_expiry", "tod", "strike", "ce", "pe", "ce_iv", "pe_iv"]
        if not req:
            return pd.DataFrame(columns=cols)
        fe.con.register("far_req", pd.DataFrame(req))
        q = """
        WITH last AS (
            SELECT r.date, r.near, r.far, max(c.tod) AS tod
            FROM far_req r JOIN chain c ON c.date = r.date AND c.expiry = r.far AND c.tod <= r.tod
            GROUP BY r.date, r.near, r.far
        )
        SELECT l.date, l.near AS expiry, l.far AS far_expiry, c.tod, c.strike, c.ce, c.pe, c.ce_iv, c.pe_iv
        FROM last l JOIN chain c ON c.date = l.date AND c.expiry = l.far AND c.tod = l.tod
        """
        df = fe.con.execute(q).df()
        fe.con.unregister("far_req")
        for c in ("date", "expiry", "far_expiry"):
            df[c] = pd.to_datetime(df[c])
        return df[cols]

    def snapshot_tags(self, snap, name, far_rule=None, far_engine=None):
        """{tag: chain rows of that leg expiry} for every non-near expiry the strategy uses.
        w1/w2/w3 come from this data set (1st/2nd/3rd expiry after near); 'far' uses far_rule / far_engine."""
        out = {}
        for tag in leg_tags(name):
            if tag == "near":
                continue
            if tag == "far":
                out[tag] = self.snapshot_far(snap, name, far_rule, far_engine)
            else:
                out[tag] = self.snapshot_far(snap, name, tag_rule(tag), None)
        return out

    def expiries_on(self, date):
        """Expiries present in this data set's files on a date."""
        g = self._exp_by_date.get(pd.Timestamp(date))
        return [] if g is None else list(g["expiry"])

    def tag_expiry(self, tag, date, near, far_rule=None, far_engine=None, name=None):
        """(engine, expiry) for one leg expiry tag on a date, or (None, None) if not in the data."""
        if tag == "near":
            return self, pd.Timestamp(near)
        if tag == "far":
            fe = far_engine or self
            rule = far_rule or (get_spec(name).get("far_expiry", "next_month") if name is not None else "next_month")
            return fe, fe.far_expiry_for(date, near, rule)
        return self, self.far_expiry_for(date, near, tag_rule(tag))

    # ---------------------------------------------------------------- VIX
    def add_vix(self, df, tod):
        """Add VIX columns for each row's 'date': vix_open/high/low/close (daily), vix_at (1-minute, last
        minute <= tod) and vix = vix_at if available, else that day's close."""
        if df is None or df.empty:
            return df
        out = df.copy()
        for c in ("vix_open", "vix_high", "vix_low", "vix_close", "vix_at", "vix"):
            out[c] = np.nan
        if not self.has_vix:
            return out
        d = pd.to_datetime(out["date"])
        vd = self.vix_daily
        for c in ("open", "high", "low", "close"):
            out[f"vix_{c}"] = d.map(vd[c]).astype(float).values
        if self.has_vix_minute:
            m = self.vix_minute[self.vix_minute["tod"] <= int(tod)]
            at = m.sort_values("tod").groupby("date")["vix"].last()
            out["vix_at"] = d.map(at).astype(float).values
        out["vix"] = out["vix_at"].fillna(out["vix_close"])
        return out

    def vix_intraday(self, date):
        """1-minute VIX for one day (empty if no minute data)."""
        if not self.has_vix_minute:
            return pd.DataFrame()
        m = self.vix_minute[self.vix_minute["date"] == pd.Timestamp(date)]
        return pd.DataFrame({"time": [hhmm(t) for t in m["tod"]], "vix": m["vix"].astype(float).values})

    def _period(self, start, end):
        start = pd.Timestamp(start) if start is not None else self.first_expiry
        end = pd.Timestamp(end) if end is not None else self.last_expiry
        return start, end

    # ---------------------------------------------------------------- options
    def _week_mask(self, d, near_week):
        if near_week in (None, "all"):
            return d
        if near_week == "last":
            return d[d["is_last_week"]]
        return d[d["week_no"] == int(near_week)]

    def dte_options(self, contract, start=None, end=None, near_week="all"):
        """Days-to-expiry values with data, and how many expiries each has, for expiries in [start, end]."""
        start, end = self._period(start, end)
        d = self.days[(self.days["expiry"] >= start) & (self.days["expiry"] <= end)]
        if contract == "Monthly":                     # only monthly expiries of this data set
            d = d[d["is_monthly"]]
        d = self._week_mask(d, near_week)
        return d.groupby("dte")["expiry"].nunique().sort_index()

    # ---------------------------------------------------------------- snapshot
    def snapshot(self, dte, contract, tod, start=None, end=None, near_week="all"):
        """Chain rows at the last minute <= tod, on the day that is `dte` trading days before each expiry,
        for expiries between start and end (default: all)."""
        start, end = self._period(start, end)
        q = """
        WITH sel AS (
            SELECT date, expiry FROM days
            WHERE dte = ? AND (? <> 'Monthly' OR is_monthly) AND expiry BETWEEN ? AND ?
              AND (? = 'all' OR (? = 'last' AND is_last_week) OR CAST(week_no AS VARCHAR) = ?)
        ), last AS (
            SELECT c.date, c.expiry, max(c.tod) AS tod
            FROM chain c JOIN sel ON c.date = sel.date AND c.expiry = sel.expiry
            WHERE c.tod <= ?
            GROUP BY c.date, c.expiry
        )
        SELECT c.date, c.expiry, c.tod, c.strike, c.ce, c.pe, c.spot, c.ce_iv, c.pe_iv
        FROM chain c JOIN last l ON c.date = l.date AND c.expiry = l.expiry AND c.tod = l.tod
        """
        nw = str(near_week or "all")
        df = self.con.execute(q, [int(dte), contract, start.to_pydatetime(), end.to_pydatetime(),
                                  nw, nw, nw, int(tod)]).df()
        df["date"] = pd.to_datetime(df["date"])
        df["expiry"] = pd.to_datetime(df["expiry"])
        return df

    # ---------------------------------------------------------------- strategy prices
    @staticmethod
    def strategy_prices(snap, name, n_strikes=10, wing=50, gap=STRIKE_GAP, far_snap=None, step=None):
        """One row per (expiry, side, strike offset): every leg's strike and price, and the strategy price.
        far_snap: {tag: rows} from snapshot_tags (next week / far week / chosen far legs);
                  a single DataFrame is treated as the 'far' legs."""
        rows = []
        step = step or gap                       # distance between table rows / builder strike steps
        n_legs = len(get_spec(name)["legs"])
        if far_snap is None:
            far_snap = {}
        elif isinstance(far_snap, pd.DataFrame):
            far_snap = {"far": far_snap}
        tag_groups = {tag: ({k: g for k, g in df.groupby(["date", "expiry"])} if df is not None and len(df) else {})
                      for tag, df in far_snap.items()}
        extra = [t for t in leg_tags(name) if t != "near"]
        empty = pd.Series(dtype=float)
        for (date, expiry), g in snap.groupby(["date", "expiry"]):
            spot = g["spot"].dropna()
            if spot.empty:
                continue
            spot = float(spot.iloc[0])
            atm = round(spot / gap) * gap
            tod = g["tod"].iloc[0]
            g = g.drop_duplicates("strike").set_index("strike")
            px = {"near": {"CE": g["ce"], "PE": g["pe"]}}
            exps = {"near": expiry}
            for tag in extra:
                fg = tag_groups.get(tag, {}).get((date, expiry))
                if fg is not None and len(fg):
                    exps[tag] = fg["far_expiry"].iloc[0]
                    fg = fg.drop_duplicates("strike").set_index("strike")
                    px[tag] = {"CE": fg["ce"], "PE": fg["pe"]}
                else:
                    exps[tag] = pd.NaT
                    px[tag] = {"CE": empty, "PE": empty}
            atm_ce_iv = float(g["ce_iv"].get(atm, np.nan))
            atm_pe_iv = float(g["pe_iv"].get(atm, np.nan))
            atm_iv = float(np.nanmean([atm_ce_iv, atm_pe_iv])) if not (np.isnan(atm_ce_iv) and np.isnan(atm_pe_iv)) else np.nan
            for side in sides_of(name):
                legs = resolve_legs(name, side, wing, step)
                for k in range(-n_strikes, n_strikes + 1):
                    K = atm + k * step
                    row = {"date": date, "expiry": expiry, "time": hhmm(tod), "spot": spot, "atm": atm,
                           "offset": k, "strike": K, "side": side,
                           "formula": formula_text(legs, K, exps), "far_expiry": exps.get("far", pd.NaT),
                           "atm_ce_iv": atm_ce_iv, "atm_pe_iv": atm_pe_iv, "atm_iv": atm_iv}
                    for tag in extra:
                        row[f"exp_{tag}"] = exps[tag]
                    total, ok = 0.0, True
                    for i, (q, t, off, tag) in enumerate(legs, 1):
                        p = float(px.get(tag, px["near"])[t].get(K + off, np.nan))
                        row[f"leg{i}"] = leg_text(q, t, K + off, tag, exps, bool(extra))
                        row[f"leg{i}_price"] = p
                        if np.isnan(p):
                            ok = False
                        else:
                            total += q * p
                    row["price"] = total if ok else np.nan
                    rows.append(row)
        df = pd.DataFrame(rows)
        df.attrs["n_legs"] = n_legs
        return df

    # ---------------------------------------------------------------- intraday
    def _day_rows(self, date, expiries, strikes):
        q = f"""
        SELECT expiry, tod, strike, ce, pe, spot, ce_iv, pe_iv FROM chain
        WHERE date = ? AND expiry IN ({", ".join("?" * len(expiries))}) AND strike IN ({", ".join("?" * len(strikes))})
        ORDER BY tod
        """
        df = self.con.execute(q, [pd.Timestamp(date).date(), *[pd.Timestamp(e).date() for e in expiries], *strikes]).df()
        df["expiry"] = pd.to_datetime(df["expiry"])
        return df

    def intraday(self, date, expiry, name, side, strike, wing=50, atm=None, far_rule=None, far_engine=None,
                 step=None):
        """The strategy (fixed strikes and expiries) for every minute of one day, with ATM IV at `atm`."""
        legs = resolve_legs(name, side, wing, step or self.gap)
        expiry = pd.Timestamp(expiry)
        where = {}                                          # tag -> (engine, expiry)
        for tag in {tag for *_, tag in legs}:
            e_, x_ = self.tag_expiry(tag, date, expiry, far_rule, far_engine, name)
            if x_ is None:
                return pd.DataFrame()
            where[tag] = (e_, pd.Timestamp(x_))
        need = {}                                           # (engine id, expiry) -> [engine, strikes]
        for _, _, off, tag in legs:
            e_, x_ = where[tag]
            need.setdefault((id(e_), x_), [e_, set()])[1].add(float(strike + off))
        if atm is not None:
            need.setdefault((id(self), expiry), [self, set()])[1].add(float(atm))
        parts = [e_._day_rows(date, [x_], sorted(ks)) for (_, x_), (e_, ks) in need.items()]
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if df.empty:
            return df
        tbl = {col: df.pivot_table(index="tod", columns=["expiry", "strike"], values=col, aggfunc="last")
               for col in ("ce", "pe")}
        idx = tbl["ce"].index.union(tbl["pe"].index)
        total = pd.Series(0.0, index=idx)
        for q_, t, off, tag in legs:
            t_ = tbl[t.lower()].reindex(idx).ffill()
            key = (where[tag][1], float(strike + off))
            if key not in t_.columns:
                return pd.DataFrame()
            total = total + q_ * t_[key]
        near = df[df["expiry"] == expiry]
        spot = near.groupby("tod")["spot"].last().reindex(idx).ffill()
        out = pd.DataFrame({"time": [hhmm(t) for t in idx], "price": total.values, "spot": spot.values})
        if atm is not None:
            a = near[near["strike"] == float(atm)].groupby("tod")[["ce_iv", "pe_iv"]].last().reindex(idx).ffill()
            out["atm_ce_iv"] = a["ce_iv"].values
            out["atm_pe_iv"] = a["pe_iv"].values
            out["atm_iv"] = a[["ce_iv", "pe_iv"]].mean(axis=1).values
        return out.dropna(subset=["price"])

    # ---------------------------------------------------------------- ATM IV by day
    def atm_iv_by_day(self, contract, tod, start=None, end=None):
        """ATM IV on every trading day (nearest expiry; nearest monthly expiry for Monthly) at the last
        minute <= tod. ATM = spot rounded to the strike gap at that minute."""
        start, end = self._period(start, end)
        q = """
        WITH near AS (
            SELECT date, min(expiry) AS expiry, min(dte) AS dte FROM days
            WHERE (? <> 'Monthly' OR is_monthly) AND date BETWEEN ? AND ?
            GROUP BY date
        ), last AS (
            SELECT c.date, c.expiry, max(c.tod) AS tod
            FROM chain c JOIN near n ON c.date = n.date AND c.expiry = n.expiry
            WHERE c.tod <= ?
            GROUP BY c.date, c.expiry
        )
        SELECT c.date, c.expiry, n.dte, c.tod, c.strike, c.spot, c.ce_iv, c.pe_iv
        FROM chain c
        JOIN last l ON c.date = l.date AND c.expiry = l.expiry AND c.tod = l.tod
        JOIN near n ON c.date = n.date AND c.expiry = n.expiry
        """
        lo = (start - pd.Timedelta(days=45)).to_pydatetime()   # days before the first expiry in the period
        df = self.con.execute(q, [contract, lo, end.to_pydatetime(), int(tod)]).df()
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df["expiry"] = pd.to_datetime(df["expiry"])
        df = df[df["expiry"] >= start]
        rows = []
        for (date, expiry), g in df.groupby(["date", "expiry"]):
            spot = float(g["spot"].dropna().iloc[0]) if g["spot"].notna().any() else np.nan
            if np.isnan(spot):
                continue
            atm = round(spot / self.gap) * self.gap
            a = g[g["strike"] == atm]
            ce = float(a["ce_iv"].iloc[0]) if len(a) else np.nan
            pe = float(a["pe_iv"].iloc[0]) if len(a) else np.nan
            rows.append({"date": date, "expiry": expiry, "dte": int(g["dte"].iloc[0]), "time": hhmm(g["tod"].iloc[0]),
                         "spot": spot, "atm": atm, "ce_iv": ce, "pe_iv": pe,
                         "atm_iv": np.nanmean([ce, pe]) if not (np.isnan(ce) and np.isnan(pe)) else np.nan})
        return pd.DataFrame(rows)


# =============================================================================
# DESCRIPTIVE STATISTICS (pandas describe + a few extras)
# =============================================================================
def describe(df, columns):
    """{label: column} -> table with count, mean, std, min, 5%, 25%, median, 75%, 95%, max, skew."""
    data = pd.DataFrame({label: pd.to_numeric(df[col], errors="coerce") for label, col in columns.items()
                         if col in df.columns})
    if data.empty:
        return pd.DataFrame()
    d = data.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    d = d.rename(columns={"count": "Count", "mean": "Mean", "std": "Std dev", "min": "Min", "50%": "Median",
                          "max": "Max"})
    d["Skew"] = data.skew()
    d["Range"] = d["Max"] - d["Min"]
    d["Count"] = d["Count"].astype(int)
    return d[["Count", "Mean", "Median", "Std dev", "Min", "5%", "25%", "75%", "95%", "Max", "Range", "Skew"]]
