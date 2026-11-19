r"""
Step 1 - build the app's data from your raw 1-minute option-chain parquet files.

    python build_data.py                      build / update every data set in datasets.json
    python build_data.py --only nifty_monthly build just one data set
    python build_data.py --vix-download       also download daily India VIX (needs: pip install yfinance)
    python build_data.py --inspect            show what your raw files contain (expiries per file ...)

DATA SETS ARE KEPT SEPARATE. Each one is listed in datasets.json (next to this script) with its own
folder - e.g. NIFTY weekly, NIFTY monthly, later SENSEX / BANKNIFTY ... - and is built into its own
folder under app_data:

    app_data\nifty_weekly\    chain\<YYYY-MM-DD>.parquet, days.parquet, meta.json, manifest.json
    app_data\nifty_monthly\   (same, from the monthly folder only)
    app_data\vix\             India VIX (daily and/or 1-minute), shared by all data sets

Each data folder is laid out like  NIFTY_WEEKLY\2026\04\2026-04-01.parquet  (date in the file name).
To add an index later, add an entry to datasets.json (folder, symbol, strike_gap, lot_size) and run
this script again - nothing else changes. If a data set's folder is not found, the script asks for
the path in the terminal and saves it in datasets.json.

Run it again after adding new days: unchanged days are skipped, changed days are rebuilt
(--rebuild redoes everything).

Only a period:   python build_data.py --from 2025-11-01 --to 2026-04-30
Cut built data back to a period (deletes built days outside it, raw files untouched):
                 python build_data.py --from 2025-11-01 --to 2026-04-30 --clean
Start completely fresh: delete the app_data folder and run again.

India VIX (optional) - not in the option files: put NSE daily VIX CSVs and/or 1-minute VIX files
in the "vix" folder next to this script (any number of .csv / .xlsx), or use --vix-download.
"""
import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS (command-line options override the first two)
# =============================================================================
CONFIG_FILE = None                                       # set below: datasets.json next to this script
HERE = Path(__file__).resolve().parent
OUTPUT_FOLDER = str(HERE / "app_data")                    # where the app reads its data
CONFIG_FILE = HERE / "datasets.json"                      # your data sets (created on the first run)

# Template written to datasets.json on the first run. Edit datasets.json, not this.
DEFAULT_CONFIG = {
    "datasets": [
        {"id": "nifty_weekly", "underlying": "NIFTY", "contract": "Weekly", "label": "NIFTY Weekly",
         "folder": r"D:\NIFTY_WEEKLY", "symbol": "NIFTY", "strike_gap": 50, "lot_size": 75, "enabled": True},
        {"id": "nifty_monthly", "underlying": "NIFTY", "contract": "Monthly", "label": "NIFTY Monthly",
         "folder": r"D:\NIFTY_MONTHLY", "symbol": "NIFTY", "strike_gap": 50, "lot_size": 75, "enabled": True},
        {"id": "sensex_weekly", "underlying": "SENSEX", "contract": "Weekly", "label": "SENSEX Weekly",
         "folder": r"D:\SENSEX_WEEKLY", "symbol": "SENSEX", "strike_gap": 100, "lot_size": 20, "enabled": False},
        {"id": "sensex_monthly", "underlying": "SENSEX", "contract": "Monthly", "label": "SENSEX Monthly",
         "folder": r"D:\SENSEX_MONTHLY", "symbol": "SENSEX", "strike_gap": 100, "lot_size": 20, "enabled": False},
    ],
    "_help": ("One entry per data set. id = output folder name in app_data (letters, digits, _). "
              "underlying + contract = what the app shows (contract: Weekly or Monthly). folder = the folder with "
              "the daily parquet files. symbol = value in the 'symbol' column to keep. strike_gap = strike "
              "interval (NIFTY 50, SENSEX 100, BANKNIFTY 100). lot_size = default for the app's rupee value. "
              "enabled = false to skip. Optional: strike_window (points around spot to keep, default 20 x strike_gap)."),
}
VIX_FOLDER = str(HERE / "vix")                            # India VIX files (daily NSE CSV and/or 1-minute)

# (the symbol to keep and the strike gap are set per data set in datasets.json)
TIME_COL = "snapshot_time"          # minute timestamp column ('resp_time' is tried as a backup)
EXPIRY_COL = "auto"                 # "auto" = column with the weekly expiries; or "expiry" / "resp_expiry"
PRICE_SOURCE = "ltp"                # "ltp" or "mid" (bid-ask mid)
LTP_FALLBACK_TO_MID = True          # LTP 0 / missing -> use bid-ask mid
# (strikes kept around spot: strike_window per data set, default 20 x strike_gap)
MARKET_OPEN, MARKET_CLOSE = "09:15", "15:30"

READ_COLS = ["symbol", "trade_date", "expiry", "resp_expiry", "strike", "spot", "snapshot_time",
             "resp_time", "ce_ltp", "ce_bid", "ce_ask", "pe_ltp", "pe_bid", "pe_ask", "ce_iv", "pe_iv"]


# =============================================================================
# DATE / TIME PARSING (handles all common formats)
# =============================================================================
DATE_FORMATS = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%Y", "%d %b %Y",
                "%d%b%Y", "%d-%b-%y", "%d%b%y", "%d-%B-%Y", "%d %B %Y", "%Y%m%d", "%d.%m.%Y",
                "%d-%m-%Y %H:%M:%S", "%d-%b-%Y %H:%M:%S", "%Y/%m/%d"]


@lru_cache(maxsize=None)
def parse_one(text):
    text = text.strip()
    if not text or text.lower() in ("nan", "nat", "none"):
        return pd.NaT
    if text.isdigit() and len(text) >= 10:
        n = int(text)
        return pd.to_datetime(n, unit="ms" if n > 1e11 else "s").normalize()
    for fmt in DATE_FORMATS:
        try:
            return pd.to_datetime(text, format=fmt).normalize()
        except (ValueError, TypeError):
            pass
    try:
        return pd.to_datetime(text, dayfirst=not text[:4].isdigit()).normalize()
    except (ValueError, TypeError):
        return pd.NaT


def parse_dates(values):
    if pd.api.types.is_datetime64_any_dtype(values):
        v = values.dt.tz_localize(None) if values.dt.tz is not None else values
        return v.dt.normalize()
    txt = values.astype(str)
    mapping = {u: parse_one(u) for u in txt.unique()}
    return pd.to_datetime(txt.map(mapping))


@lru_cache(maxsize=None)
def clock_of(text):
    """Time of day (Timedelta) from text like 09:15, 09:15:00, 2026-04-01 09:15:00, ...+05:30."""
    text = text.strip()
    if not text or text.lower() in ("nan", "nat", "none"):
        return pd.NaT
    if "day" in text:
        try:
            return pd.to_timedelta(text)
        except ValueError:
            pass
    for kw in ({}, {"dayfirst": True}):
        try:
            ts = pd.to_datetime(text, **kw)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("Asia/Kolkata").tz_localize(None)
            return ts - ts.normalize()
        except (ValueError, TypeError):
            continue
    return pd.NaT


def to_datetime_on(values, trade_date):
    """Full timestamps on the trading date from any kind of time column."""
    v = values
    if pd.api.types.is_timedelta64_dtype(v):
        return trade_date + v
    if pd.api.types.is_datetime64_any_dtype(v):
        t = v.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None) if v.dt.tz is not None else v
        return trade_date + (t - t.dt.normalize())
    if pd.api.types.is_numeric_dtype(v):
        x = pd.to_numeric(v, errors="coerce")
        med = x.dropna().median()
        if med > 1e8:                                            # epoch seconds / ms
            t = pd.to_datetime(x, unit="ms" if med > 1e11 else "s", utc=True)
            t = t.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            return trade_date + (t - t.dt.normalize())
        hhmmss = x if med >= 10000 else x * 100                  # 91500 or 915
        secs = (hhmmss // 10000) * 3600 + ((hhmmss // 100) % 100) * 60 + hhmmss % 100
        return trade_date + pd.to_timedelta(secs, unit="s")
    txt = v.astype(str)
    mapping = {u: clock_of(u) for u in txt.unique()}
    return trade_date + pd.Series([mapping[x] for x in txt], index=v.index, dtype="timedelta64[ns]")


def date_from_name(path):
    for text in (path.stem, str(path)):
        m = re.search(r"(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])", text)
        if m:
            return pd.Timestamp(int(m[1]), int(m[2]), int(m[3]))
        m = re.search(r"(0[1-9]|[12]\d|3[01])[-_.]?(0[1-9]|1[0-2])[-_.]?(20\d{2})", text)
        if m:
            return pd.Timestamp(int(m[3]), int(m[2]), int(m[1]))
        m = re.search(r"(\d{1,2})[-_ ]?([A-Za-z]{3,9})[-_ ]?(20\d{2})", text)
        if m:
            try:
                return pd.to_datetime(f"{m[1]} {m[2][:3]} {m[3]}", format="%d %b %Y")
            except ValueError:
                pass
    return None


def is_monthly(expiry):
    return (expiry + pd.Timedelta(days=7)).month != expiry.month


# =============================================================================
# READING ONE DAY
# =============================================================================
def read_file(path):
    try:
        df = pd.read_parquet(path, columns=READ_COLS)
    except Exception:                       # some column missing -> read all, then check
        df = pd.read_parquet(path)
    missing = [c for c in ("strike", "spot", "ce_ltp", "pe_ltp") if c not in df.columns]
    if TIME_COL not in df.columns and "resp_time" not in df.columns:
        missing.append(TIME_COL)
    if missing:
        raise ValueError(f"missing columns {missing}; file has {list(df.columns)[:12]}...")
    return df


def num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def leg_price(df, p):
    bid, ask = num(df, f"{p}_bid"), num(df, f"{p}_ask")
    mid = ((bid + ask) / 2).where((bid > 0) & (ask > 0))
    ltp = num(df, f"{p}_ltp").where(lambda s: s > 0)
    if PRICE_SOURCE == "mid":
        return mid
    return ltp.fillna(mid) if LTP_FALLBACK_TO_MID else ltp


def at_time(day, hhmm):
    return day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:]))


OTHER_INDICES = {"NIFTY": "BANK|FIN|MID|NEXT", "SENSEX": "BANKEX|50"}


def process_day(path, file_date, expiry_col, symbol="NIFTY", strike_window=1000):
    df = read_file(path)

    if symbol and "symbol" in df.columns:
        symbol = symbol.upper()
        sym = df["symbol"].astype(str).str.upper().str.strip()
        if (sym == symbol).any():
            df = df[sym == symbol]
        elif sym.str.contains(symbol).any():
            excl = OTHER_INDICES.get(symbol)
            keep = sym.str.contains(symbol) & (~sym.str.contains(excl) if excl else True)
            df = df[keep]
        else:
            raise ValueError(f"symbol '{symbol}' not found (file has: {sorted(sym.unique())[:4]})")

    trade_date = file_date
    if "trade_date" in df.columns and df["trade_date"].notna().any():
        td = parse_dates(df["trade_date"].dropna().head(1)).iloc[0]
        if pd.notna(td):
            trade_date = td

    # time column: as-is or shifted UTC->IST, whichever puts more minutes in market hours
    lo, hi = at_time(trade_date, MARKET_OPEN), at_time(trade_date, MARKET_CLOSE) + pd.Timedelta(minutes=1)
    best, best_n = None, 0
    for col in [c for c in (TIME_COL, "resp_time") if c in df.columns]:
        try:
            dt = to_datetime_on(df[col], trade_date)
        except Exception:
            continue
        for shift in (pd.Timedelta(0), pd.Timedelta(hours=5, minutes=30)):
            cand = dt + shift
            n = cand[(cand >= lo) & (cand < hi)].dt.floor("min").nunique()
            if n > best_n * 1.2:
                best, best_n = cand, n
        if best_n and col == TIME_COL:
            break
    if best is None:
        raise ValueError(f"could not read times from '{TIME_COL}' (e.g. {df[TIME_COL].astype(str).head(2).tolist()})")

    exp = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for col in [expiry_col] + [c for c in ("expiry", "resp_expiry") if c != expiry_col]:
        if col in df.columns:
            exp = exp.fillna(parse_dates(df[col]))

    out = pd.DataFrame({
        "minute": best.dt.floor("min"), "expiry": exp, "strike": num(df, "strike"),
        "ce": leg_price(df, "ce"), "pe": leg_price(df, "pe"), "spot": num(df, "spot"),
        "ce_iv": num(df, "ce_iv").where(lambda v: v > 0), "pe_iv": num(df, "pe_iv").where(lambda v: v > 0),
    })
    # IV stored as a decimal (0.14) -> convert to percent (14.0)
    for c in ("ce_iv", "pe_iv"):
        med = out[c].median()
        if pd.notna(med) and med < 3:
            out[c] = out[c] * 100
    out = out[(out["minute"] >= lo) & (out["minute"] < hi)].dropna(subset=["minute", "expiry", "strike"])
    if out.empty:
        raise ValueError("no rows with time, expiry and strike inside market hours")

    spot = out[out["spot"] > 0].groupby("minute")["spot"].last()
    out["spot"] = out["minute"].map(spot)
    out = out[(out["strike"] - out["spot"]).abs() <= strike_window]

    out = (out.sort_values("minute")
              .groupby(["expiry", "strike", "minute"], as_index=False).last()
              .sort_values(["expiry", "strike", "minute"]))
    cols = ["ce", "pe", "ce_iv", "pe_iv"]
    out[cols] = out.groupby(["expiry", "strike"])[cols].ffill()   # carry last price / IV forward

    out["date"] = trade_date.date()
    out["tod"] = (out["minute"].dt.hour * 60 + out["minute"].dt.minute).astype("int16")   # minutes since midnight
    out["expiry"] = out["expiry"].dt.date
    out["strike"] = out["strike"].astype("float32")
    for c in ("ce", "pe", "spot", "ce_iv", "pe_iv"):
        out[c] = out[c].astype("float32")
    return out[["date", "tod", "expiry", "strike", "ce", "pe", "spot", "ce_iv", "pe_iv"]], trade_date


# =============================================================================
# MAIN
# =============================================================================
def detect_expiry_col(files):
    if EXPIRY_COL != "auto":
        return EXPIRY_COL
    pick = np.unique(np.linspace(0, len(files) - 1, min(15, len(files))).astype(int))
    found = {"expiry": set(), "resp_expiry": set()}
    for j in pick:
        try:
            df = read_file(files[j][0])
        except Exception:
            continue
        for c in found:
            if c in df.columns:
                found[c] |= set(parse_dates(df[c].dropna().drop_duplicates()).dropna())
    for c, s in found.items():
        weekly = sum(not is_monthly(pd.Timestamp(e)) for e in s)
        print(f"  {c:12s}: {len(s)} distinct expiries in sample ({weekly} weekly, {len(s) - weekly} monthly)")
    col = max(found, key=lambda c: (len(found[c]), c == "expiry"))
    print(f"  -> using expiry column '{col}'")
    return col


def build_days(out_dir):
    import duckdb
    chain_glob = (out_dir / "chain" / "*.parquet").as_posix()
    pairs = duckdb.sql(f"SELECT DISTINCT date, expiry FROM read_parquet('{chain_glob}')").df()
    pairs["date"] = pd.to_datetime(pairs["date"])
    pairs["expiry"] = pd.to_datetime(pairs["expiry"])
    trading = set(pairs["date"].unique())
    weekdays = pd.bdate_range(pairs["date"].min(), pairs["date"].max())
    holidays = [d.date() for d in weekdays if d not in trading]
    pairs["dte"] = np.busday_count(pairs["date"].values.astype("datetime64[D]"),
                                   pairs["expiry"].values.astype("datetime64[D]"),
                                   holidays=np.array(holidays, dtype="datetime64[D]"))
    pairs = pairs[pairs["dte"] >= 0].copy()
    pairs["is_monthly"] = pairs["expiry"].apply(is_monthly)
    pairs["date"] = pairs["date"].dt.date
    pairs["expiry"] = pairs["expiry"].dt.date
    pairs = pairs.sort_values(["expiry", "date"]).reset_index(drop=True)
    pairs.to_parquet(out_dir / "days.parquet", index=False)
    return pairs, holidays


# =============================================================================
# INDIA VIX (daily and/or 1-minute files)
# =============================================================================
def _norm(c):
    return re.sub(r"[^a-z0-9%]", "", str(c).lower())


def _find(cols, names):
    for n in names:
        if n in cols:
            return cols[n]
    return None


def read_vix_file(path):
    """One VIX file -> (daily rows, minute rows). Handles NSE CSVs and generic date/time/close files."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = df.dropna(how="all")
    cols = {_norm(c): c for c in df.columns}
    dcol = _find(cols, ["datetime", "timestamp", "date", "tradedate", "day", "time"])
    tcol = _find(cols, ["time"]) if dcol != cols.get("time") else None
    ccol = _find(cols, ["close", "vix", "indiavix", "ltp", "last", "price", "closeprice", "value"])
    if dcol is None or ccol is None:
        raise ValueError(f"needs a date and a close/VIX column; found {list(df.columns)}")
    numf = lambda c: pd.to_numeric(df[c].astype(str).str.replace(",", "").str.strip(), errors="coerce") if c else None
    close = numf(ccol)
    o, h, l = (numf(_find(cols, [n])) for n in ("open", "high", "low"))

    raw = df[dcol]
    if tcol is not None:                                           # separate date + time columns
        dates = parse_dates(raw)
        dt = pd.Series([d + clock_of(str(t)) if pd.notna(d) else pd.NaT for d, t in zip(dates, df[tcol])],
                       index=df.index, dtype="datetime64[ns]")
        minute = True
    elif pd.api.types.is_datetime64_any_dtype(raw):
        dt = raw.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None) if raw.dt.tz is not None else raw
        minute = (dt != dt.dt.normalize()).any()
    else:
        txt = raw.astype(str).str.strip()
        minute = txt.str.contains(":").any()
        if minute:
            first = txt.dropna().iloc[0]
            dt = pd.to_datetime(txt, errors="coerce", format="mixed", dayfirst=not first[:4].isdigit(), utc=True)
            if any(("+" in x[10:] or x.endswith("Z")) for x in txt.head(5)):
                dt = dt.dt.tz_convert("Asia/Kolkata")
            dt = dt.dt.tz_localize(None)
        else:
            dt = parse_dates(txt)

    if minute:
        m = pd.DataFrame({"dt": dt, "vix": close}).dropna()
        m = m[(m["dt"].dt.hour * 60 + m["dt"].dt.minute >= 9 * 60 + 15) & (m["dt"].dt.hour * 60 + m["dt"].dt.minute <= 15 * 60 + 30)]
        m["date"] = m["dt"].dt.normalize()
        m["tod"] = (m["dt"].dt.hour * 60 + m["dt"].dt.minute).astype("int16")
        m = m.sort_values("dt").groupby(["date", "tod"], as_index=False)["vix"].last()
        return None, m
    d = pd.DataFrame({"date": pd.to_datetime(dt).dt.normalize(), "close": close,
                      "open": o if o is not None else close, "high": h if h is not None else close,
                      "low": l if l is not None else close}).dropna(subset=["date", "close"])
    return d, None


def download_vix(vix_dir, start):
    """Daily India VIX from Yahoo Finance (^INDIAVIX) into vix/yahoo_indiavix.csv."""
    try:
        import yfinance as yf
    except ImportError:
        print("  --vix-download needs yfinance:  pip install yfinance")
        return
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    print(f"  downloading India VIX {start:%Y-%m-%d} -> today ...")
    d = yf.download("^INDIAVIX", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"),
                    progress=False, auto_adjust=False)
    if d is None or d.empty:
        print("  no VIX data returned (check your internet connection)")
        return
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d = d.reset_index()[["Date", "Open", "High", "Low", "Close"]]
    vix_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(vix_dir / "yahoo_indiavix.csv", index=False)
    print(f"  saved {len(d)} days to {vix_dir / 'yahoo_indiavix.csv'}")


def build_vix(vix_dir, out_dir):
    """All VIX files -> app_data/vix_daily.parquet (+ vix_minute.parquet if 1-minute data exists)."""
    files = sorted([p for p in vix_dir.rglob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls")]) if vix_dir.is_dir() else []
    if not files:
        print(f"VIX: no files in {vix_dir} (optional - see the top of this script)")
        return {}
    daily, minute = [], []
    for p in files:
        try:
            d, m = read_vix_file(p)
            if d is not None and len(d):
                daily.append(d.assign(src=p.name))
            if m is not None and len(m):
                minute.append(m)
        except Exception as e:
            print(f"  VIX file skipped - {p.name}: {type(e).__name__}: {e}")
    info = {}
    if minute:
        m = pd.concat(minute).sort_values(["date", "tod"]).drop_duplicates(["date", "tod"], keep="last")
        m["vix"] = m["vix"].astype("float32")
        m.to_parquet(out_dir / "vix_minute.parquet", index=False)
        from_minute = m.groupby("date")["vix"].agg(open="first", high="max", low="min", close="last").reset_index()
        daily.append(from_minute.assign(src="1-minute data"))
        info["vix_minute_days"] = int(m["date"].nunique())
    elif (out_dir / "vix_minute.parquet").exists():
        (out_dir / "vix_minute.parquet").unlink()
    if daily:
        d = pd.concat(daily)
        d["prio"] = (d["src"] == "1-minute data").astype(int)            # daily files win over derived values
        d = d.sort_values(["date", "prio"]).drop_duplicates("date", keep="first").drop(columns=["prio", "src"])
        d = d.sort_values("date").reset_index(drop=True)
        d.to_parquet(out_dir / "vix_daily.parquet", index=False)
        info.update({"vix_days": int(len(d)), "vix_first": str(d["date"].min().date()),
                     "vix_last": str(d["date"].max().date())})
    print(f"VIX: {info.get('vix_days', 0)} days ({info.get('vix_first', '-')} -> {info.get('vix_last', '-')})"
          + (f", 1-minute data on {info['vix_minute_days']} days" if "vix_minute_days" in info else ", daily only"))
    return info


def load_config():
    """datasets.json -> list of data sets. Creates it on the first run (asks for the NIFTY folders)."""
    if not CONFIG_FILE.exists():
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        print(f"Creating {CONFIG_FILE.name} - your list of data sets (edit it any time to add indices).")
        if sys.stdin.isatty():
            for ds in cfg["datasets"][:2]:
                typed = input(f"Folder for {ds['label']} (Enter = {ds['folder']}, '-' = none): ").strip().strip('"')
                if typed == "-":
                    ds["enabled"] = False
                elif typed:
                    ds["folder"] = typed
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    cfg = json.loads(CONFIG_FILE.read_text())
    out = []
    for ds in cfg.get("datasets", []):
        ds = dict(ds)
        ds.setdefault("contract", "Weekly")
        ds.setdefault("underlying", ds.get("symbol", "NIFTY"))
        ds.setdefault("symbol", ds["underlying"])
        ds.setdefault("label", f"{ds['underlying']} {ds['contract']}")
        ds.setdefault("strike_gap", 50)
        ds.setdefault("lot_size", 75)
        ds.setdefault("enabled", True)
        ds.setdefault("strike_window", 20 * ds["strike_gap"])
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", str(ds.get("id", ""))):
            sys.exit(f"datasets.json: every data set needs an 'id' made of letters, digits, _ or - (got {ds.get('id')!r})")
        out.append(ds)
    ids = [d["id"] for d in out]
    if len(ids) != len(set(ids)):
        sys.exit("datasets.json: two data sets have the same id")
    return out


def expiry_report(days):
    """Print the expiries found per month, marking the monthly ones."""
    ex = pd.to_datetime(pd.Series(days["expiry"].unique())).sort_values()
    print("  expiries (M = monthly = last expiry of its month):")
    for ym, g in ex.groupby(ex.dt.to_period("M")):
        print(f"    {ym}: " + ", ".join(f"{e:%d %b}{' (M)' if is_monthly(e) else ''}" for e in g))
    if not any(is_monthly(e) for e in ex):
        print("  (no monthly expiry in this data set)")


def save_folder(ds_id, folder):
    """Write a data set's folder into datasets.json (so it is not asked again)."""
    cfg = json.loads(CONFIG_FILE.read_text())
    for d in cfg.get("datasets", []):
        if d.get("id") == ds_id:
            d["folder"] = str(folder)
            d.pop("folders", None)
            d["enabled"] = True
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def ask_folder(ds):
    """Ask for a data set's folder in the terminal. Returns a Path, or None to skip."""
    if not sys.stdin.isatty():
        return None
    while True:
        typed = input(f"  Folder for {ds['label']} (paste the path, Enter = skip): ").strip().strip('"').strip("'")
        if not typed:
            print(f"  skipped (set \"enabled\": false for '{ds['id']}' in {CONFIG_FILE.name} to stop being asked)")
            return None
        p = Path(typed)
        if p.is_dir():
            save_folder(ds["id"], p.resolve())
            print(f"  saved in {CONFIG_FILE.name}")
            return p
        print("  That folder does not exist - copy it from the Explorer address bar and try again.")


def build_dataset(ds, out_root, args):
    """Build one data set from its own folder(s) into app_data/<id>. Returns its days table or None."""
    folders = ds.get("folders") or [ds.get("folder")]
    srcs = [Path(str(f).strip().strip('"')) for f in folders if f]
    missing = [str(p) for p in srcs if not p.is_dir()]
    srcs = [p for p in srcs if p.is_dir()]
    print(f"\n=== {ds['label']}  (id: {ds['id']})")
    if missing:
        print(f"  folder not found: {', '.join(missing)}")
        if not srcs:
            p = ask_folder(ds)
            if p is not None:
                srcs = [p]
            elif not sys.stdin.isatty():
                print(f"  -> set 'folder' for '{ds['id']}' in {CONFIG_FILE.name}, or run this script in a terminal "
                      f"(it will ask for the path)")
    if not srcs:
        print("  skipped")
        return None
    out_dir = out_root / ds["id"]
    (out_dir / "chain").mkdir(parents=True, exist_ok=True)

    by_date, no_date = {}, []
    for src in srcs:
        for p in sorted(src.rglob("*.parquet")):
            d = date_from_name(p)
            if d is None:
                no_date.append(p)
            else:
                by_date.setdefault(d, []).append(p)
    dates = sorted(by_date)
    months = sorted({f"{d:%Y-%m}" for d in dates})
    print(f"  source: {', '.join(str(p.resolve()) for p in srcs)}")
    print(f"  {sum(len(v) for v in by_date.values())} files, {len(dates)} trading days, months: "
          f"{', '.join(months) if months else 'none'}" + (f" ({len(no_date)} skipped: no date in name)" if no_date else ""))
    if not dates:
        print("  no parquet files with a date in the file name - skipped")
        return None

    d_from = pd.Timestamp(args.date_from) if args.date_from else None
    d_to = pd.Timestamp(args.date_to) if args.date_to else None
    if d_from is not None or d_to is not None:
        lo_d, hi_d = d_from or pd.Timestamp.min, d_to or pd.Timestamp.max
        dates = [d for d in dates if lo_d <= d <= hi_d]
        print(f"  period {args.date_from or 'start'} -> {args.date_to or 'end'}: {len(dates)} trading days")
        if args.clean:
            removed = 0
            for f in (out_dir / "chain").glob("*.parquet"):
                try:
                    d = pd.Timestamp(f.stem)
                except ValueError:
                    continue
                if not (lo_d <= d <= hi_d):
                    f.unlink()
                    removed += 1
            print(f"  --clean: removed {removed} built day(s) outside the period")

    if dates:
        expiry_col = detect_expiry_col([(p, d) for d in dates for p in by_date[d]])
    else:
        old_meta = out_dir / "meta.json"
        expiry_col = json.loads(old_meta.read_text()).get("expiry_column", "expiry") if old_meta.exists() else "expiry"

    man_path = out_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    done, failed, skipped = 0, [], 0
    for n, d in enumerate(dates, 1):
        key = f"{d:%Y-%m-%d}"
        paths = by_date[d]
        sig = sorted(str(p.resolve()) for p in paths)
        target = out_dir / "chain" / f"{key}.parquet"
        if target.exists() and not args.rebuild and manifest.get(key) == sig:
            skipped += 1
        else:
            parts, tdate = [], d
            for p in paths:
                try:
                    part, tdate = process_day(p, d, expiry_col, ds["symbol"], ds["strike_window"])
                    parts.append(part)
                except Exception as e:
                    failed.append((p, f"{type(e).__name__}: {e}"))
                    if len(failed) == 1:
                        print(f"  first error - {p.name}: {type(e).__name__}: {e}")
            if parts:
                day = (pd.concat(parts, ignore_index=True)
                         .drop_duplicates(["expiry", "strike", "tod"], keep="first")
                         .sort_values(["expiry", "strike", "tod"]))
                if f"{tdate:%Y-%m-%d}" != key and target.exists():
                    target.unlink()
                day.to_parquet(out_dir / "chain" / f"{tdate:%Y-%m-%d}.parquet", index=False)
                manifest[f"{tdate:%Y-%m-%d}"] = sig
                done += 1
        if n % 20 == 0 or n == len(dates):
            print(f"  {n}/{len(dates)} days  (built {done}, unchanged {skipped}, failed files {len(failed)})")
    man_path.write_text(json.dumps(manifest, indent=1))
    if failed:
        print(f"  {len(failed)} file(s) failed, e.g.:")
        for p, err in failed[:5]:
            print(f"    {p.name}: {err}")
    if not any((out_dir / "chain").glob("*.parquet")):
        print("  nothing was built for this data set")
        return None

    days, holidays = build_days(out_dir)
    per_day = days.groupby("date")["expiry"].nunique()
    print(f"  expiries per trading day in the files: usually {int(per_day.median())} "
          f"(min {int(per_day.min())}, max {int(per_day.max())})")
    if per_day.max() <= 1:
        print("  NOTE: each day's files hold ONE expiry only, so next-week / far-week legs (calendars) cannot be "
              "priced from this data set. Run  python build_data.py --inspect  to see what the files contain.")
    exp = pd.to_datetime(days["expiry"]).drop_duplicates()
    meta = {
        "id": ds["id"], "underlying": ds["underlying"].upper(), "contract": ds["contract"], "label": ds["label"],
        "symbol": ds["symbol"], "strike_gap": ds["strike_gap"], "lot_size": ds["lot_size"],
        "sources": [str(p.resolve()) for p in srcs], "expiry_column": expiry_col, "price": PRICE_SOURCE,
        "first_day": str(days["date"].min()), "last_day": str(days["date"].max()),
        "trading_days": int(days["date"].nunique()), "expiries": int(len(exp)),
        "monthly_expiries": int(sum(is_monthly(e) for e in exp)),
        "holidays_inferred": [str(h) for h in holidays],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    expiry_report(days)
    print(f"  done: {meta['trading_days']} trading days, {meta['expiries']} expiries "
          f"({meta['monthly_expiries']} monthly), {meta['first_day']} -> {meta['last_day']}  -> {out_dir}")
    return days


def inspect(datasets):
    """Show what the raw files contain: expiry columns, expiries per file and rows per expiry."""
    for ds in datasets:
        folders = [Path(str(f).strip().strip('"')) for f in (ds.get("folders") or [ds.get("folder")]) if f]
        files = sorted(p for f in folders if f.is_dir() for p in f.rglob("*.parquet"))
        print(f"\n=== {ds['label']}: {len(files)} files in {', '.join(map(str, folders))}")
        if not files:
            print("  no files (check the folder in datasets.json)")
            continue
        pick = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, min(4, len(files))).astype(int))]
        for p in pick:
            df = read_file(p)
            print(f"  {p.name}: {len(df):,} rows, columns with 'expir': {[c for c in df.columns if 'expir' in c.lower()]}")
            for c in ("expiry", "resp_expiry"):
                if c in df.columns:
                    vc = df[c].astype(str).value_counts()
                    print(f"    {c:12s}: {len(vc)} value(s): " + ", ".join(f"{k} ({v:,} rows)" for k, v in vc.head(6).items()))
            if "symbol" in df.columns:
                print(f"    symbol      : {sorted(df['symbol'].astype(str).unique())[:5]}")
        print("  -> Next-week / far-week legs need SEVERAL expiries in one day's file (in the expiry column).")


def main():
    ap = argparse.ArgumentParser(description="Build Strategy Analyzer data sets from raw option-chain parquet files")
    ap.add_argument("--only", nargs="+", default=None, help="build only these data set ids (see datasets.json)")
    ap.add_argument("--out", default=OUTPUT_FOLDER, help="app data folder (default app_data next to this script)")
    ap.add_argument("--rebuild", action="store_true", help="rebuild all days, not only new / changed ones")
    ap.add_argument("--from", dest="date_from", default=None, help="only days on/after this date (YYYY-MM-DD)")
    ap.add_argument("--to", dest="date_to", default=None, help="only days on/before this date (YYYY-MM-DD)")
    ap.add_argument("--vix", default=VIX_FOLDER, help="folder with India VIX files (default: vix next to this script)")
    ap.add_argument("--vix-download", action="store_true", help="download daily India VIX (needs yfinance)")
    ap.add_argument("--clean", action="store_true",
                    help="with --from/--to: delete already-built days outside that period")
    ap.add_argument("--inspect", action="store_true", help="only show what the raw files contain (no build)")
    args = ap.parse_args()
    if args.clean and not (args.date_from or args.date_to):
        sys.exit("--clean needs --from and/or --to (the period to keep).")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if (out_root / "days.parquet").exists():
        print(f"Note: {out_root} contains data from an older version (one mixed data set). It is no longer used - "
              f"you can delete {out_root / 'chain'}, days.parquet, meta.json and manifest.json there.")

    datasets = load_config()
    if args.only:
        unknown = set(args.only) - {d["id"] for d in datasets}
        if unknown:
            sys.exit(f"Unknown data set id(s): {', '.join(unknown)}. Known: {', '.join(d['id'] for d in datasets)}")
        datasets = [d for d in datasets if d["id"] in args.only]
    else:
        datasets = [d for d in datasets if d["enabled"]]
    if args.inspect:
        inspect(datasets)
        return
    print(f"Data sets to build: {', '.join(d['label'] for d in datasets) or 'none'}  (from {CONFIG_FILE.name})")

    built = {}
    for ds in datasets:
        days = build_dataset(ds, out_root, args)
        if days is not None:
            built[ds["label"]] = days

    # ---- India VIX: separate, shared by all data sets
    print("\n=== India VIX")
    vix_out = out_root / "vix"
    vix_out.mkdir(exist_ok=True)
    vix_dir = Path(args.vix)
    if args.vix_download:
        starts = [pd.Timestamp(d["date"].min()) for d in built.values()]
        start = (min(starts) if starts else pd.Timestamp.today() - pd.DateOffset(years=2)) - pd.Timedelta(days=10)
        download_vix(vix_dir, start)
    vix_info = build_vix(vix_dir, vix_out)
    (vix_out / "meta.json").write_text(json.dumps(vix_info, indent=2))

    print("\nSummary:")
    for label, days in built.items():
        print(f"  {label:20s} {days['date'].nunique():4d} days  {pd.Timestamp(days['date'].min()):%d %b %Y} -> "
              f"{pd.Timestamp(days['date'].max()):%d %b %Y}")
    if not built:
        print("  no data set was built - set the folders in datasets.json and run again")
    print(f"  India VIX            {vix_info.get('vix_days', 0)} days")
    print(f"\nData written to {out_root.resolve()}\nNext: streamlit run app.py")


if __name__ == "__main__":
    main()
