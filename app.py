"""
Step 2 - the Strategy Analyzer screen.

    streamlit run app.py

Pick a strategy, days to expiry, contract type and time of day; click any price in the table to see
how that strategy has priced across past expiries. Built-in strategies are in engine.py (STRATEGIES);
your own are built on screen ("Custom (build your own)") and saved in custom_strategies.json.
"""
import io
import json
import re
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (STRATEGIES, TAG_LABELS, Engine, describe, find_datasets, is_two_sided, leg_tags,
                    multi_expiry, needs_far, offset_label, sides_of)

SIDE_COLOR = {"CE": "#1D4ED8", "PE": "#C2410C", "ALL": "#6D28D9"}

st.set_page_config(page_title="Strategy Analyzer", layout="wide")
st.markdown("""
<style>
div[data-testid="stButton"] button { font-variant-numeric: tabular-nums; min-height: 38px; }
.strike-cell { text-align: center; line-height: 1.15; padding-top: 4px; }
.strike-cell b { font-size: 15px; }
.strike-cell span { font-size: 12px; color: #4B5563; }
.atm-row b { background: #FEF3C7; padding: 1px 8px; border-radius: 6px; }
.col-head { text-align: center; font-weight: 700; font-size: 14px; padding: 6px 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner=False)
def get_engine(folder):
    return Engine(folder)


@st.cache_data(show_spinner=False)
def get_prices(folder, dte, contract, tod, spec_json, n_strikes, wing, start, end, near_week="all", far_rule=None,
               far_folder=None, step=None):
    eng = get_engine(folder)
    far_eng = get_engine(far_folder) if far_folder else None     # far leg from another data set (e.g. monthly)
    name = json.loads(spec_json)
    snap = eng.snapshot(dte, contract, tod, start, end, near_week)
    far = eng.snapshot_tags(snap, name, far_rule, far_eng) if multi_expiry(name) else None  # next / far week legs
    p = Engine.strategy_prices(snap, name, n_strikes=n_strikes, wing=wing, gap=eng.gap, far_snap=far, step=step)
    return eng.add_vix(p, tod)


@st.cache_data(show_spinner=False)
def get_intraday(folder, date, expiry, spec_json, side, strike, wing, atm, far_rule=None, far_folder=None, step=None):
    far_eng = get_engine(far_folder) if far_folder else None
    return get_engine(folder).intraday(date, expiry, json.loads(spec_json), side, strike, wing, atm, far_rule, far_eng,
                                       step)


# ---------------------------------------------------------------- custom strategies (saved to a file)
CUSTOM_FILE = Path(__file__).with_name("custom_strategies.json")
CUSTOM_LABEL = "Custom (build your own)"
STRIKE_CHOICES = [offset_label(k) for k in range(10, -11, -1)]
BUILDER_COLS = ["Action", "Lots", "Type", "Strike", "Expiry"]
# builder "Expiry" choices -> leg expiry tag (old saved strategies used "Near" / "Far")
EXPIRY_CHOICES = {"Near week": "near", "Next week": "w1", "Far week (+2)": "w2", "Week +3": "w3",
                  "Chosen far (below)": "far"}
EXPIRY_OLD = {"Near": "near", "Far": "far"}


def load_custom():
    try:
        return json.loads(CUSTOM_FILE.read_text()) if CUSTOM_FILE.exists() else {}
    except Exception:
        return {}


def save_custom(d):
    CUSTOM_FILE.write_text(json.dumps(d, indent=2))


def strike_steps(label):
    label = str(label).replace("−", "-").strip()
    return 0 if label == "ATM" else int(label[3:])


def rows_to_spec(df):
    """Builder table -> strategy spec (distances in points, quantity = ±lots)."""
    legs, about = [], []
    for r in df.dropna(subset=BUILDER_COLS).itertuples(index=False):
        lots = int(r.Lots)
        if lots <= 0:
            continue
        qty = lots if r.Action == "Buy" else -lots
        tag = EXPIRY_CHOICES.get(r.Expiry, EXPIRY_OLD.get(r.Expiry, "near"))
        legs.append([qty, r.Type, strike_steps(r.Strike), tag])
        about.append(f"{r.Action} {lots} {r.Type} {r.Strike} {TAG_LABELS[tag].lower()}")
    return {"legs": legs, "units": "steps", "about": ", ".join(about), "rows": df[BUILDER_COLS].dropna().to_dict("records")}


def default_rows():
    return pd.DataFrame([{"Action": "Buy", "Lots": 1, "Type": "CE", "Strike": "ATM", "Expiry": "Near week"},
                         {"Action": "Buy", "Lots": 1, "Type": "PE", "Strike": "ATM", "Expiry": "Near week"}])


def edit_in_builder(name):
    rows = load_custom().get(name, {}).get("rows")
    df = pd.DataFrame(rows, columns=BUILDER_COLS) if rows else default_rows()
    df["Expiry"] = df["Expiry"].replace({"Near": "Near week", "Far": "Chosen far (below)"})
    st.session_state.builder_df = df
    st.session_state.builder_last = st.session_state.builder_df
    st.session_state.builder_ver = st.session_state.get("builder_ver", 0) + 1
    st.session_state.save_name = name.replace("Saved: ", "")
    st.session_state.strategy = CUSTOM_LABEL


def delete_custom(name):
    d = load_custom()
    d.pop(name, None)
    save_custom(d)
    st.session_state.strategy = list(STRATEGIES)[0]


@st.cache_data(show_spinner=False)
def get_atm_iv(folder, contract, tod, start, end):
    eng = get_engine(folder)
    d = eng.atm_iv_by_day(contract, tod, start, end)
    d = eng.add_vix(d, tod)
    if not d.empty:
        d["iv_minus_vix"] = d["atm_iv"] - d["vix"]
    return d


@st.cache_data(show_spinner=False)
def get_vix_day(folder, date):
    return get_engine(folder).vix_intraday(date)


# ---------------------------------------------------------------- data sets: underlying + contract
with st.sidebar:
    st.header("Settings")
data_root = str(Path(__file__).with_name("app_data"))      # where build_data.py writes the data sets

datasets = find_datasets(data_root)
if not datasets:
    st.error("No data found yet.")
    st.info("Build the data first:  python build_data.py   (it reads the folders listed in datasets.json) "
            "— then reload this page.")
    st.stop()

h1, h2, h3 = st.columns([2.2, 1, 1])
h1.title("Strategy Analyzer")
h1.caption("Option strategy prices from 1-minute option-chain history")
underlyings = sorted({d["underlying"] for d in datasets}, key=lambda u: (u != "NIFTY", u))
underlying = h2.selectbox("Underlying", underlyings, help="Indices you have built (datasets.json).")
ds_for_u = sorted([d for d in datasets if d["underlying"] == underlying],
                  key=lambda d: (d["contract"] != "Weekly", d["contract"]))
contract = h3.selectbox("Type of contract", [d["contract"] for d in ds_for_u],
                        help="Each type is its own data set (built from its own folder) - nothing is mixed.")
ds = next(d for d in ds_for_u if d["contract"] == contract)
data_dir = ds["path"]

try:
    eng = get_engine(data_dir)
except Exception as e:
    st.error(f"{e}")
    st.info("Build the data first:  python build_data.py  — then reload this page.")
    st.stop()
gap = int(eng.gap)
lot_size = int(ds.get("lot_size", 75))          # only for the ₹ values; set per index in datasets.json
if ds.get("legacy"):
    st.warning("This data was built by an older version (weekly and monthly in one set). Run the new "
               "build_data.py to build separate data sets, then delete the old chain folder in app_data.")

with st.sidebar:
    step = st.selectbox("Table strike step (points)", [gap, gap * 2, gap * 4], index=0, key=f"step_{underlying}",
                        help=f"Distance between the rows of the option-chain table (ATM+1, ATM+2 ...) and between "
                             f"the builder's strikes. {underlying} strikes are listed every {gap} points; choose "
                             f"{gap * 2} to step ATM+1 = ATM+{gap * 2} points.")
    wing = st.selectbox("Wing width (points)", [gap * i for i in (1, 2, 3, 4)], index=0, key=f"wing_{underlying}",
                        help=f"Distance between the legs of the built-in strategies (butterfly wings ...). "
                             f"{underlying} strike gap = {gap}.")
    n_strikes = st.slider("Strikes either side of ATM", 3, 10, 10)
    table_value = st.radio("Price shown in the table", ["Latest expiry", "Average of all expiries", "Median"],
                           help="The chart always shows every past expiry.")

if eng.meta:
    st.sidebar.caption(f"{ds['label']}: {eng.meta.get('first_day')} → {eng.meta.get('last_day')} · "
                       f"{eng.meta.get('trading_days')} days · {eng.meta.get('expiries')} expiries")
if eng.has_vix:
    vd = eng.vix_daily.index
    st.sidebar.caption(f"India VIX: {vd.min():%d %b %Y} → {vd.max():%d %b %Y} · "
                       + ("1-minute data" if eng.has_vix_minute else "daily data (VIX = that day's close)"))
else:
    st.sidebar.caption("India VIX: no data — add files to the vix folder or run "
                       "`python build_data.py --vix-download`")

# ---------------------------------------------------------------- period filter
with st.sidebar:
    st.subheader("Period")
    first_e, last_e = eng.first_expiry.date(), eng.last_expiry.date()
    preset = st.selectbox("Expiries to include",
                          ["All data", "Last 3 months", "Last 6 months", "Last 12 months", "Custom range"],
                          help="Only expiries in this period are used in the table, chart and downloads.")
    if preset == "Custom range":
        rng = st.date_input("Expiries from / to", value=(first_e, last_e),
                            min_value=first_e, max_value=last_e, format="DD-MM-YYYY")
        if not isinstance(rng, (tuple, list)) or len(rng) != 2:
            st.info("Pick an end date too.")
            st.stop()
        p_start, p_end = rng
    elif preset == "All data":
        p_start, p_end = first_e, last_e
    else:
        months = int(preset.split()[1])
        p_start = max(first_e, (pd.Timestamp(last_e) - pd.DateOffset(months=months) + pd.Timedelta(days=1)).date())
        p_end = last_e
    st.caption(f"Using expiries {p_start:%d %b %Y} → {p_end:%d %b %Y}")

# ---------------------------------------------------------------- selections
WEEK_OPTS = {"All weeks": "all", "1st week": "1", "2nd week": "2", "3rd week": "3", "4th week": "4",
             "5th week": "5", "Last week of month": "last"}
c0, c1, c3, c4 = st.columns([1.4, 1, 1, 1.2])
custom = load_custom()
options = list(STRATEGIES) + [f"Saved: {n}" for n in custom] + [CUSTOM_LABEL]
if st.session_state.get("strategy") not in options:
    st.session_state.strategy = options[0]
strategy = c0.selectbox("Strategy", options, key="strategy",
                        help="Built-in strategies, your saved ones, or build your own legs.")
near_week_lbl = c4.selectbox("Near expiry: week of month", list(WEEK_OPTS),
                             help="Use only expiries that are the 1st / 2nd / ... / last expiry of their month.")
near_week = WEEK_OPTS[near_week_lbl]

# ---- the strategy's legs: built-in, saved, or built here
if strategy == CUSTOM_LABEL:
    with st.container(border=True):
        st.markdown(f"**Build your strategy** — one row per leg (strike steps of {step} points). "
                    "Strike is relative to the centre strike of each "
                    "table row (ATM row = ATM). **Expiry**: *Near week* = the expiry picked by Days to expiry; "
                    "*Next week* / *Far week (+2)* / *Week +3* = the 1st / 2nd / 3rd expiry after it in the same data; "
                    "*Chosen far* = the far expiry you set below (week of a month, or another data set). "
                    "Add rows with the + at the bottom of the table; delete by selecting a row and pressing Delete.")
        ed_key = f"builder_{st.session_state.get('builder_ver', 0)}"
        if "builder_df" not in st.session_state:
            st.session_state.builder_df = default_rows()
        elif ed_key not in st.session_state and "builder_last" in st.session_state:
            st.session_state.builder_df = st.session_state.builder_last      # coming back: keep your legs
        if "Expiry" in st.session_state.builder_df:
            st.session_state.builder_df["Expiry"] = st.session_state.builder_df["Expiry"].replace(
                {"Near": "Near week", "Far": "Chosen far (below)"})
        edited = st.data_editor(
            st.session_state.builder_df, num_rows="dynamic", width="stretch", hide_index=True, key=ed_key,
            column_config={
                "Action": st.column_config.SelectboxColumn("Buy / Sell", options=["Buy", "Sell"], required=True, default="Buy"),
                "Lots": st.column_config.NumberColumn("Lots", min_value=1, max_value=100, step=1, required=True, default=1),
                "Type": st.column_config.SelectboxColumn("CE / PE", options=["CE", "PE"], required=True, default="CE"),
                "Strike": st.column_config.SelectboxColumn("Strike", options=STRIKE_CHOICES, required=True, default="ATM"),
                "Expiry": st.column_config.SelectboxColumn("Expiry", options=list(EXPIRY_CHOICES), required=True,
                                                           default="Near week"),
            })
        st.session_state.builder_last = edited
        spec = rows_to_spec(edited)
        if not spec["legs"]:
            st.warning("Add at least one complete leg.")
            st.stop()
        s1, s2 = st.columns([3, 1])
        save_as = s1.text_input("Save as (name)", key="save_name", placeholder="e.g. My 1-3-2 ratio")
        if s2.button("Save strategy", width="stretch", disabled=not save_as.strip()):
            name = re.sub(r"\s+", " ", save_as.strip())
            custom[name] = spec
            save_custom(custom)
            st.success(f"Saved — it is now in the Strategy list as 'Saved: {name}'.")
    strategy_name = "Custom"
elif strategy.startswith("Saved: "):
    spec = custom[strategy[len("Saved: "):]]
    strategy_name = strategy[len("Saved: "):]
    with st.expander(f"Legs of '{strategy_name}'", expanded=False):
        st.dataframe(pd.DataFrame(spec.get("rows", [])), hide_index=True, width="stretch")
        e1, e2 = st.columns(2)
        e1.button("Edit in builder", on_click=edit_in_builder, args=(strategy[len("Saved: "):],), width="stretch")
        e2.button("Delete this strategy", on_click=delete_custom, args=(strategy[len("Saved: "):],), width="stretch")
else:
    spec = STRATEGIES[strategy]
    strategy_name = strategy
spec_json = json.dumps(spec, sort_keys=True, default=str)

# ---- calendar strategies: you choose the far leg's expiry
far_rule, far_text, far_folder = None, "", None
if needs_far(spec):
    other = [d for d in ds_for_u if d["id"] != ds["id"]]
    if other:
        src_opts = {f"Same data ({contract})": None, **{f"{d['label']} data": d["path"] for d in other}}
        far_src = st.selectbox("Far leg data", list(src_opts),
                               help="Where the far legs' prices come from. The data sets stay separate; "
                                    "this only combines them for this strategy.")
        far_folder = src_opts[far_src]
    f0, f1, f2 = st.columns([1.4, 1, 1])
    how = f0.radio("Far leg expiry", ["Week of a month", "Weeks after near expiry"], horizontal=True,
                   help="Which later expiry the far legs (next-month legs) use.")
    if how == "Week of a month":
        MONTHS = {"Same month": 0, "Next month": 1, "2 months ahead": 2, "3 months ahead": 3}
        m_lbl = f1.selectbox("Far leg month", list(MONTHS), index=1)
        w_lbl = f2.selectbox("Far leg week", [k for k in WEEK_OPTS if k != "All weeks"], index=5)
        far_rule = ("week_month", MONTHS[m_lbl], WEEK_OPTS[w_lbl])
        far_text = f"{w_lbl.replace(' of month', '')} expiry of {m_lbl.lower()}"
    else:
        n = f1.number_input("Weeks after near expiry", min_value=1, max_value=12, value=1, step=1)
        far_rule = ("weeks_after", int(n))
        far_text = f"expiry {int(n)} week{'s' if n != 1 else ''} after the near expiry"

dte_avail = eng.dte_options("All", p_start, p_end, near_week)
max_dte = int(dte_avail.index.max()) if len(dte_avail) else 0
dte = c1.number_input("Days to expiry", min_value=0, max_value=max(max_dte, 0),
                      value=min(2, max_dte), step=1, help="Trading days before expiry; 0 = expiry day.")
tod_t = c3.time_input("Time of day", value=time(11, 30), step=60)
tod = tod_t.hour * 60 + tod_t.minute
n_exp = int(dte_avail.get(dte, 0))
dte_text = "expiry day" if dte == 0 else f"{dte} day{'s' if dte != 1 else ''} to expiry"
st.caption(f"**{strategy_name}**: {spec.get('about', '')}{'' if spec.get('units') == 'points' else f' · wing {wing}'} · "
           f"{contract} expiries{'' if near_week == 'all' else ' (' + near_week_lbl.lower() + ')'} {p_start:%d %b %Y} → {p_end:%d %b %Y} · {dte_text} · {tod_t:%H:%M} · {n_exp} expiries have data · "
           f"positive price = net debit, negative = net credit")

if not (time(9, 15) <= tod_t <= time(15, 30)):
    st.warning("Pick a time between 09:15 and 15:30.")
    st.stop()
if not len(dte_avail):
    st.warning(f"No {contract.lower()} expiries in {p_start:%d %b %Y} → {p_end:%d %b %Y}. "
               "Widen the period or choose the other contract type.")
    st.stop()

with st.spinner("Loading prices ..."):
    prices = get_prices(data_dir, int(dte), "All", tod, spec_json, n_strikes, wing, p_start, p_end,
                        near_week, far_rule, far_folder, step)

if prices.empty:
    st.warning(f"No {contract.lower()} expiries have data at {dte_text}. "
               f"Available days to expiry: {', '.join(map(str, dte_avail.index.tolist())) or 'none'}.")
    st.stop()

n_legs = len(spec["legs"])
strategy = strategy_name                      # display name from here on
if multi_expiry(spec):
    n_near = prices["expiry"].nunique()
    for tag in [t for t in leg_tags(spec) if t != "near"]:
        col = f"exp_{tag}"
        lbl = far_text if tag == "far" else TAG_LABELS[tag].lower()
        have = prices.dropna(subset=[col]) if col in prices else prices.iloc[0:0]
        if have.empty:
            last = prices.sort_values("date").iloc[-1]
            on_day = eng.expiries_on(last["date"])
            on_txt = ", ".join(f"{e:%d %b %Y}" for e in on_day) or "none"
            if tag != "far":
                n = int(tag[1:])
                after = [e for e in eng.expiries["expiry"] if e > last["expiry"]]
                want = f"{after[n - 1]:%d %b %Y}" if len(after) >= n else "a later expiry"
                why = (f"On {last['date']:%d %b %Y} (near expiry {last['expiry']:%d %b %Y}) your {ds['label']} data has "
                       f"only these expiries: **{on_txt}** - the {TAG_LABELS[tag].lower()} expiry ({want}) is not in "
                       f"that day's file. ")
            else:
                why = (f"On {last['date']:%d %b %Y} the far-leg data has no expiry matching **{far_text}**. ")
            st.warning(f"No prices for the **{TAG_LABELS[tag]}** legs. " + why +
                       "Calendar legs are priced from the SAME day's file, so each day's file must contain the "
                       "option chain of every expiry you use (e.g. this week AND next week). Run "
                       "`python build_data.py --inspect` to see which expiries your files hold."
                       + (" For a monthly far leg, set *Far leg data* to the Monthly data set." if tag == "far" else ""))
        else:
            st.caption(f"**{TAG_LABELS[tag]}** legs{' (' + lbl + ')' if tag == 'far' else ''}: e.g. near {have['expiry'].iloc[-1]:%d %b %Y} → "
                       f"{have[col].iloc[-1]:%d %b %Y} · {have['expiry'].nunique()} of {n_near} near expiries have data.")
sides = sides_of(spec)
two_sided = is_two_sided(spec)

# ---------------------------------------------------------------- IV filter (with the time of day)
IV_COLS = {"ATM IV (CE+PE average)": "atm_iv", "ATM CE IV": "atm_ce_iv", "ATM PE IV": "atm_pe_iv"}
iv_filter, iv_lo, iv_hi, iv_col = "Off", None, None, None
with st.container(border=True):
    v0, v1, v2, v3 = st.columns([1.4, 1, 1, 2.2])
    iv_filter = v0.selectbox("IV filter", ["Off"] + list(IV_COLS), key="iv_filter",
                             help="Keep only the expiries whose IV, at your time of day on the priced day, is inside "
                                  "the range. The table, chart, statistics and downloads all use the kept expiries.")
    if iv_filter != "Off":
        iv_col = IV_COLS[iv_filter]
        ivs = prices.drop_duplicates(["date", "expiry"])[iv_col].dropna()
        if ivs.empty:
            v3.warning("No IV in the data for these expiries - the IV filter is not applied.")
            iv_filter = "Off"
        else:
            lo_def, hi_def = float(np.floor(ivs.min())), float(np.ceil(ivs.max()))
            k_lo, k_hi = f"iv_lo_{iv_col}", f"iv_hi_{iv_col}"
            st.session_state.setdefault(k_lo, lo_def)
            st.session_state.setdefault(k_hi, hi_def)
            iv_lo = v1.number_input("IV from (%)", min_value=0.0, max_value=300.0, step=0.5, key=k_lo, format="%.1f")
            iv_hi = v2.number_input("IV to (%)", min_value=0.0, max_value=300.0, step=0.5, key=k_hi, format="%.1f")
            n_all = prices["expiry"].nunique()
            prices = prices[(prices[iv_col] >= iv_lo) & (prices[iv_col] <= iv_hi)]
            n_kept = prices["expiry"].nunique()
            v3.markdown(f"<div style='padding-top:30px;font-size:14px'>{iv_filter} {iv_lo:.1f}% – {iv_hi:.1f}% at "
                        f"{tod_t:%H:%M}: <b>{n_kept} of {n_all}</b> expiries kept "
                        f"<span style='color:#4B5563'>(data range {ivs.min():.1f}% – {ivs.max():.1f}%)</span></div>",
                        unsafe_allow_html=True)
            if prices.empty:
                st.warning("No expiries have their IV inside that range at this time of day. Widen the range.")
                st.stop()
    else:
        v1.markdown("<div style='padding-top:30px;font-size:14px;color:#4B5563'>Off - all expiries are used. "
                    "Pick an IV to keep only the days when IV was in a range at your time.</div>",
                    unsafe_allow_html=True)
iv_note = f" · {iv_filter} {iv_lo:.1f}–{iv_hi:.1f}%" if iv_filter != "Off" else ""

# ---------------------------------------------------------------- table values
latest_exp = prices["expiry"].max()
if table_value == "Latest expiry":
    tbl = prices[prices["expiry"] == latest_exp]
    table_note = (f"Prices for expiry {latest_exp:%d %b %Y} (priced {tbl['date'].iloc[0]:%d %b} "
                  f"{tbl['time'].iloc[0]}), ATM {tbl['atm'].iloc[0]:,.0f}")
else:
    agg = "mean" if table_value.startswith("Average") else "median"
    tbl = prices.groupby(["side", "offset"], as_index=False)["price"].agg(agg)
    table_note = f"{table_value.split(' of')[0]} across {prices['expiry'].nunique()} expiries{iv_note}"
val = {(r.side, r.offset): r.price for r in tbl.itertuples()}
strike_of = (prices[prices["expiry"] == latest_exp].drop_duplicates("offset")
             .set_index("offset")["strike"].to_dict())

# selection (side, offset) - reset if it doesn't fit the current strategy / strike range
sel = st.session_state.get("sel", (sides[-1], 0))
if sel[0] not in sides or abs(sel[1]) > n_strikes:
    sel = (sides[-1], 0)
st.session_state.sel = sel
sel_side, sel_off = sel


def pick(side, k):
    st.session_state.sel = (side, k)


def fmt(v):
    return "–" if pd.isna(v) else f"{v:,.2f}"


def side_name(side):
    return "" if side == "ALL" else f" {side}"


left, right = st.columns([1, 1.75], gap="large")

# ---------------------------------------------------------------- left: chain table
with left:
    if two_sided:
        hc = st.columns(3)
        hc[0].markdown(f"<div class='col-head' style='color:{SIDE_COLOR['PE']}'>{strategy} PE</div>",
                       unsafe_allow_html=True)
        hc[1].markdown("<div class='col-head'>Strike</div>", unsafe_allow_html=True)
        hc[2].markdown(f"<div class='col-head' style='color:{SIDE_COLOR['CE']}'>{strategy} CE</div>",
                       unsafe_allow_html=True)
    else:
        hc = st.columns([1, 1.4])
        hc[0].markdown("<div class='col-head'>Strike</div>", unsafe_allow_html=True)
        hc[1].markdown(f"<div class='col-head' style='color:{SIDE_COLOR['ALL']}'>{strategy}</div>",
                       unsafe_allow_html=True)

    with st.container(height=640):
        for k in range(n_strikes, -n_strikes - 1, -1):
            strike_txt = f"{strike_of[k]:,.0f}" if k in strike_of else ""
            strike_html = (f"<div class='strike-cell {'atm-row' if k == 0 else ''}'><b>{offset_label(k)}</b>"
                           f"<br><span>{strike_txt}</span></div>")
            if two_sided:
                rc = st.columns(3)
                cells = [(rc[0], "PE"), (rc[2], "CE")]
                rc[1].markdown(strike_html, unsafe_allow_html=True)
            else:
                rc = st.columns([1, 1.4])
                cells = [(rc[1], "ALL")]
                rc[0].markdown(strike_html, unsafe_allow_html=True)
            for col, side in cells:
                col.button(fmt(val.get((side, k))), key=f"{side}{k}", on_click=pick, args=(side, k),
                           type="primary" if (sel_side, sel_off) == (side, k) else "secondary",
                           width="stretch", help=f"{strategy}{side_name(side)} {offset_label(k)}")
    st.caption(f"{table_note}. Click any price to see its history.")

# ---------------------------------------------------------------- right: history, ATM IV, statistics
IV_COLOR = "#059669"
VIX_COLOR = "#DB2777"
VIX_NOTE = "VIX at your time of day (1-minute data)" if eng.has_vix_minute else "VIX = that day's close (daily data)"


def pct(v):
    return "–" if pd.isna(v) else f"{v:.2f}%"


def num_fmt(df):
    return df.style.format(lambda v: "–" if pd.isna(v) else (f"{v:,.0f}" if float(v).is_integer() and abs(v) > 999 else f"{v:,.2f}"))


with right:
    s = (prices[(prices["side"] == sel_side) & (prices["offset"] == sel_off)]
         .dropna(subset=["price"]).sort_values("expiry").reset_index(drop=True))
    color = SIDE_COLOR[sel_side]
    st.subheader(f"{strategy}{side_name(sel_side)} · {offset_label(sel_off)}")
    if not s.empty:
        st.caption(f"{s['formula'].iloc[-1]} (latest) · {dte_text} · {tod_t:%H:%M}{iv_note} · {contract} expiries · "
                   f"{len(s)} points · strikes from the ATM at that moment")
    if not eng.has_iv:
        st.info("IV is not in the built data. Run build_data.py again (delete app_data first) to add ATM IV.")

    tab_hist, tab_iv, tab_stats = st.tabs(["Price history", "ATM IV by day", "Statistics (describe)"])

    # ============================== tab 1: price history
    with tab_hist:
        if s.empty:
            st.info("No prices for this strategy at the selected days to expiry and time "
                    "(a leg's strike may be missing from the data).")
        else:
            m = st.columns(4)
            m[0].metric("Latest", f"{s['price'].iloc[-1]:,.2f}")
            m[1].metric("Average", f"{s['price'].mean():,.2f}")
            m[2].metric("Lowest", f"{s['price'].min():,.2f}")
            m[3].metric("Highest", f"{s['price'].max():,.2f}")
            m2 = st.columns(4)
            m2[0].metric("ATM IV (latest)", pct(s["atm_iv"].iloc[-1]))
            m2[1].metric("VIX (latest)", "–" if pd.isna(s["vix"].iloc[-1]) else f"{s['vix'].iloc[-1]:.2f}", help=VIX_NOTE)
            m2[2].metric(f"Latest ₹ (lot {lot_size})", f"₹{s['price'].iloc[-1] * lot_size:,.0f}")
            m2[3].metric("Average ₹", f"₹{s['price'].mean() * lot_size:,.0f}")

            st.caption(f"₹ at lot size {lot_size}: lowest ₹{s['price'].min() * lot_size:,.0f} · "
                       f"highest ₹{s['price'].max() * lot_size:,.0f}  (positive = you pay, negative = you receive)")
            t1, t2 = st.columns(2)
            show_iv = t1.toggle("Show ATM IV on the chart", value=eng.has_iv, disabled=not eng.has_iv)
            show_vix = t2.toggle("Show India VIX on the chart", value=eng.has_vix, disabled=not eng.has_vix)
            legs_txt = s.apply(lambda r: "<br>".join(f"{r[f'leg{i}']} @ {r[f'leg{i}_price']:,.2f}"
                                                     for i in range(1, n_legs + 1)), axis=1)
            iv_txt = s.apply(lambda r: "–" if pd.isna(r["atm_iv"]) else
                             f"{r['atm_iv']:.2f}% (CE {r['atm_ce_iv']:.2f} / PE {r['atm_pe_iv']:.2f})", axis=1)
            vix_txt = s.apply(lambda r: "–" if pd.isna(r["vix"]) else
                              f"{r['vix']:.2f} (day O {r['vix_open']:.2f} H {r['vix_high']:.2f} L {r['vix_low']:.2f} C {r['vix_close']:.2f})",
                              axis=1)
            custom = list(zip(s["date"].dt.strftime("%d %b %Y"), s["time"], s["spot"], legs_txt, iv_txt, vix_txt))
            fig = go.Figure(go.Scatter(
                x=s["expiry"], y=s["price"], mode="lines+markers", name=strategy,
                line=dict(color=color, width=2.5), marker=dict(size=8, color="white", line=dict(color=color, width=2)),
                customdata=custom,
                hovertemplate=("<b>Expiry %{x|%d %b %Y}</b><br>Priced %{customdata[0]} %{customdata[1]}"
                               "<br>Spot %{customdata[2]:,.2f}<br>ATM IV %{customdata[4]}<br>VIX %{customdata[5]}"
                               "<br>%{customdata[3]}"
                               f"<br><b>{strategy} %{{y:.2f}}</b><extra></extra>"),
            ))
            if show_iv and s["atm_iv"].notna().any():
                fig.add_trace(go.Scatter(x=s["expiry"], y=s["atm_iv"], mode="lines+markers", name="ATM IV (%)",
                                         yaxis="y2", line=dict(color=IV_COLOR, width=1.5, dash="dot"),
                                         marker=dict(size=5, color=IV_COLOR), hoverinfo="skip"))
            if show_vix and s["vix"].notna().any():
                fig.add_trace(go.Scatter(x=s["expiry"], y=s["vix"], mode="lines+markers", name="India VIX",
                                         yaxis="y2", line=dict(color=VIX_COLOR, width=1.5, dash="dash"),
                                         marker=dict(size=5, color=VIX_COLOR), hoverinfo="skip"))
            if iv_filter != "Off" and show_iv:
                fig.add_hrect(y0=iv_lo, y1=iv_hi, yref="y2", fillcolor=IV_COLOR, opacity=0.07, line_width=0)
            fig.add_hline(y=s["price"].mean(), line=dict(color="#9CA3AF", dash="dot", width=1),
                          annotation_text="average", annotation_position="top left")
            if (s["price"] < 0).any():
                fig.add_hline(y=0, line=dict(color="#111827", width=1))
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                              hoverlabel=dict(bgcolor="#111827", font_color="white"),
                              legend=dict(orientation="h", y=1.08, x=0),
                              xaxis=dict(title="Expiry (months)", tickformat="%b %Y", showgrid=False),
                              yaxis=dict(title="Strategy price (pts)", gridcolor="#E5E7EB"),
                              yaxis2=dict(title="ATM IV % / VIX", overlaying="y", side="right", showgrid=False))
            st.plotly_chart(fig, width="stretch")

            # ---- one expiry in detail: the strategy and ATM IV through that day
            with st.expander("See one expiry through the day (price, ATM IV, statistics)"):
                opts = {f"{r.expiry:%d %b %Y} (priced {r.date:%d %b})": r for r in s.itertuples()}
                choice = st.selectbox("Expiry", list(opts)[::-1])
                r = opts[choice]
                day = get_intraday(data_dir, r.date, r.expiry, spec_json, sel_side, float(r.strike), wing, float(r.atm),
                                   far_rule, far_folder, step)
                if day.empty:
                    st.info("No minute data for these strikes on that day.")
                else:
                    f2 = go.Figure(go.Scatter(x=day["time"], y=day["price"], mode="lines", name=strategy,
                                              line=dict(color=color, width=2), customdata=day[["spot"]].values,
                                              hovertemplate="%{x}  price %{y:.2f}  spot %{customdata[0]:,.2f}<extra></extra>"))
                    if "atm_iv" in day and day["atm_iv"].notna().any():
                        f2.add_trace(go.Scatter(x=day["time"], y=day["atm_iv"], mode="lines", yaxis="y2",
                                                name=f"ATM IV % ({r.atm:,.0f})", line=dict(color=IV_COLOR, width=1.5, dash="dot"),
                                                hovertemplate="%{x}  ATM IV %{y:.2f}%<extra></extra>"))
                    vday = get_vix_day(data_dir, r.date)
                    if not vday.empty:
                        day = day.merge(vday, on="time", how="left")
                        day["vix"] = day["vix"].ffill()
                        f2.add_trace(go.Scatter(x=day["time"], y=day["vix"], mode="lines", yaxis="y2", name="India VIX",
                                                line=dict(color=VIX_COLOR, width=1.5, dash="dash"),
                                                hovertemplate="%{x}  VIX %{y:.2f}<extra></extra>"))
                    f2.add_vline(x=r.time, line=dict(color="#9CA3AF", dash="dot"))
                    f2.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                                     legend=dict(orientation="h", y=1.12, x=0),
                                     title=dict(text=f"{r.date:%d %b %Y} · {r.formula} (strikes fixed)", font=dict(size=13)),
                                     xaxis=dict(nticks=10, showgrid=False), yaxis=dict(gridcolor="#E5E7EB"),
                                     yaxis2=dict(title="ATM IV % / VIX", overlaying="y", side="right",
                                                 showgrid=False))
                    st.plotly_chart(f2, width="stretch")
                    d1, d2, d3 = st.columns(3)
                    d1.metric("At open", f"{day['price'].iloc[0]:,.2f}", help=f"{day['time'].iloc[0]}")
                    d2.metric(f"At {r.time}", f"{r.price:,.2f}")
                    d3.metric("At close", f"{day['price'].iloc[-1]:,.2f}", help=f"{day['time'].iloc[-1]}")
                    e1, e2, e3, e4 = st.columns(4)
                    if "atm_iv" in day and day["atm_iv"].notna().any():
                        e1.metric("ATM IV open", f"{day['atm_iv'].dropna().iloc[0]:.2f}%")
                        e2.metric("ATM IV close", f"{day['atm_iv'].dropna().iloc[-1]:.2f}%")
                    if pd.notna(r.vix_close):
                        e3.metric("VIX open", f"{r.vix_open:.2f}")
                        e4.metric("VIX close", f"{r.vix_close:.2f}", help=f"High {r.vix_high:.2f} · Low {r.vix_low:.2f}")
                    st.markdown("**That day, minute by minute (describe)**")
                    st.dataframe(num_fmt(describe(day, {"Strategy price": "price", "ATM IV %": "atm_iv",
                                                        "ATM CE IV %": "atm_ce_iv", "ATM PE IV %": "atm_pe_iv",
                                                        "India VIX": "vix", "Spot": "spot"})), width="stretch")

            # ---- downloads
            leg_cols = [c for i in range(1, n_legs + 1) for c in (f"leg{i}", f"leg{i}_price")]
            exp_cols = [c for c in s.columns if c.startswith("exp_")]
            table = s[["expiry"] + exp_cols + [ "date", "time", "spot", "atm", "atm_ce_iv", "atm_pe_iv", "atm_iv", "vix",
                       "vix_open", "vix_high", "vix_low", "vix_close", "strike"]
                      + leg_cols + ["price"]].copy()
            table.columns = (["Expiry"] + [f"{TAG_LABELS[c[4:]]} expiry" for c in exp_cols] + [ "Priced on", "Time", "Spot", "ATM", "ATM CE IV %", "ATM PE IV %", "ATM IV %",
                              "VIX", "VIX open", "VIX high", "VIX low", "VIX close", "Centre strike"]
                             + [h for i in range(1, n_legs + 1) for h in (f"Leg {i}", f"Leg {i} price")]
                             + [strategy])
            table[f"₹ value (lot {lot_size})"] = table[strategy] * lot_size
            tag = f"{strategy.replace(' ', '_')}{side_name(sel_side).strip()}_{offset_label(sel_off)}_dte{dte}_{tod_t:%H%M}"
            b1, b2 = st.columns(2)
            b1.download_button("Download this history (CSV)", table.to_csv(index=False).encode(),
                               file_name=f"{tag}.csv", width="stretch")
            xl = io.BytesIO()
            with pd.ExcelWriter(xl, engine="openpyxl") as w:
                table.to_excel(w, sheet_name="This strategy", index=False)
                describe(s, {strategy: "price", "ATM IV %": "atm_iv", "India VIX": "vix",
                             "Spot": "spot"}).to_excel(w, sheet_name="Statistics")
                allv = prices.pivot_table(index="expiry", columns=["side", "offset"], values="price")
                allv.columns = [f"{'' if sd == 'ALL' else sd + ' '}{offset_label(o)}" for sd, o in allv.columns]
                allv.to_excel(w, sheet_name="All strikes")
            b2.download_button("Download all (Excel)", xl.getvalue(),
                               file_name=f"{strategy.replace(' ', '_')}_dte{dte}_{tod_t:%H%M}_{underlying.lower()}_{contract.lower()}.xlsx",
                               width="stretch")

    # ============================== tab 2: ATM IV by day
    with tab_iv:
        if not eng.has_iv:
            st.info("Rebuild the data to see ATM IV.")
        else:
            ivd = get_atm_iv(data_dir, "All", tod, p_start, p_end)
            if ivd.empty or ivd["atm_iv"].notna().sum() == 0:
                st.info("No ATM IV in this period.")
            else:
                st.caption(f"ATM IV on every trading day at {tod_t:%H:%M}, nearest {contract.lower()} expiry. "
                           "ATM = spot rounded to 50 at that minute; ATM IV = average of CE and PE IV at that strike.")
                iv_ok = ivd.dropna(subset=["atm_iv"])
                m = st.columns(4)
                m[0].metric("Latest", f"{iv_ok['atm_iv'].iloc[-1]:.2f}%", help=f"{iv_ok['date'].iloc[-1]:%d %b %Y}")
                m[1].metric("Average", f"{iv_ok['atm_iv'].mean():.2f}%")
                m[2].metric("Lowest", f"{iv_ok['atm_iv'].min():.2f}%", help=f"{iv_ok.loc[iv_ok['atm_iv'].idxmin(), 'date']:%d %b %Y}")
                m[3].metric("Highest", f"{iv_ok['atm_iv'].max():.2f}%", help=f"{iv_ok.loc[iv_ok['atm_iv'].idxmax(), 'date']:%d %b %Y}")
                if ivd["vix"].notna().any():
                    v_ok = ivd.dropna(subset=["vix"])
                    v = st.columns(4)
                    v[0].metric("VIX latest", f"{v_ok['vix'].iloc[-1]:.2f}", help=f"{v_ok['date'].iloc[-1]:%d %b %Y} · {VIX_NOTE}")
                    v[1].metric("VIX average", f"{v_ok['vix'].mean():.2f}")
                    sp = ivd["iv_minus_vix"].dropna()
                    v[2].metric("ATM IV − VIX (latest)", "–" if sp.empty else f"{sp.iloc[-1]:+.2f}",
                                help="Positive = ATM IV above VIX (weekly ATM options priced richer than VIX)")
                    v[3].metric("ATM IV − VIX (average)", "–" if sp.empty else f"{sp.mean():+.2f}")
                f3 = go.Figure()
                f3.add_trace(go.Scatter(x=ivd["date"], y=ivd["atm_iv"], mode="lines+markers", name="ATM IV",
                                        line=dict(color=IV_COLOR, width=2.5), marker=dict(size=5),
                                        customdata=list(zip(ivd["expiry"].dt.strftime("%d %b %Y"), ivd["dte"],
                                                            ivd["spot"], ivd["atm"], ivd["ce_iv"], ivd["pe_iv"],
                                                            ivd["vix"].map(lambda x: "–" if pd.isna(x) else f"{x:.2f}"))),
                                        hovertemplate=("<b>%{x|%d %b %Y}</b><br>ATM IV %{y:.2f}%"
                                                       "<br>CE %{customdata[4]:.2f}% · PE %{customdata[5]:.2f}%"
                                                       "<br>India VIX %{customdata[6]}"
                                                       "<br>ATM %{customdata[3]:,.0f} · spot %{customdata[2]:,.2f}"
                                                       "<br>Expiry %{customdata[0]} · %{customdata[1]} days to expiry"
                                                       "<extra></extra>")))
                f3.add_trace(go.Scatter(x=ivd["date"], y=ivd["ce_iv"], mode="lines", name="CE IV",
                                        line=dict(color=SIDE_COLOR["CE"], width=1, dash="dot"), hoverinfo="skip"))
                f3.add_trace(go.Scatter(x=ivd["date"], y=ivd["pe_iv"], mode="lines", name="PE IV",
                                        line=dict(color=SIDE_COLOR["PE"], width=1, dash="dot"), hoverinfo="skip"))
                if ivd["vix"].notna().any():
                    f3.add_trace(go.Scatter(x=ivd["date"], y=ivd["vix"], mode="lines+markers", name="India VIX",
                                            line=dict(color=VIX_COLOR, width=2, dash="dash"), marker=dict(size=4),
                                            hovertemplate="%{x|%d %b %Y}  VIX %{y:.2f}<extra></extra>"))
                if iv_filter != "Off":
                    f3.add_hrect(y0=iv_lo, y1=iv_hi, fillcolor=IV_COLOR, opacity=0.08, line_width=0,
                                 annotation_text=f"IV filter {iv_lo:.1f}–{iv_hi:.1f}%", annotation_position="top left")
                    in_band = ivd[iv_col.replace("atm_", "") if iv_col != "atm_iv" else "atm_iv"].between(iv_lo, iv_hi)
                    st.caption(f"{int(in_band.sum())} of {int(ivd['atm_iv'].notna().sum())} trading days had "
                               f"{iv_filter} inside {iv_lo:.1f}–{iv_hi:.1f}% at {tod_t:%H:%M} (shaded band).")
                f3.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                                 hoverlabel=dict(bgcolor="#111827", font_color="white"),
                                 legend=dict(orientation="h", y=1.08, x=0),
                                 xaxis=dict(title="Trading day", showgrid=False),
                                 yaxis=dict(title="IV (%) / VIX", gridcolor="#E5E7EB"))
                st.plotly_chart(f3, width="stretch")

                pick_day = st.date_input("Look up one day", value=iv_ok["date"].iloc[-1].date(),
                                         min_value=ivd["date"].min().date(), max_value=ivd["date"].max().date(),
                                         format="DD-MM-YYYY")
                one = ivd[ivd["date"] == pd.Timestamp(pick_day)]
                if one.empty:
                    st.info("No data on that day (holiday or missing file).")
                else:
                    o = one.iloc[0]
                    st.success(f"**{o.date:%d %b %Y} at {o.time}**: ATM {o.atm:,.0f} · ATM IV **{o.atm_iv:.2f}%** "
                               f"(CE {o.ce_iv:.2f}% / PE {o.pe_iv:.2f}%) · spot {o.spot:,.2f} · "
                               f"expiry {o.expiry:%d %b %Y} ({'expiry day' if o.dte == 0 else f'{o.dte} day' + ('s' if o.dte != 1 else '') + ' to expiry'})")
                    if pd.notna(o.vix_close):
                        st.info(f"**India VIX on {o.date:%d %b %Y}**: open {o.vix_open:.2f} · high {o.vix_high:.2f} · "
                                f"low {o.vix_low:.2f} · close {o.vix_close:.2f}"
                                + (f" · at {o.time}: {o.vix_at:.2f}" if pd.notna(o.vix_at) else "")
                                + (f" · ATM IV − VIX: {o.iv_minus_vix:+.2f}" if pd.notna(o.iv_minus_vix) else ""))
                    elif eng.has_vix:
                        st.info("No India VIX for that day in your VIX files.")

                st.markdown("**ATM IV statistics for the period (describe)**")
                st.dataframe(num_fmt(describe(ivd, {"ATM IV %": "atm_iv", "ATM CE IV %": "ce_iv",
                                                    "ATM PE IV %": "pe_iv", "India VIX": "vix",
                                                    "ATM IV − VIX": "iv_minus_vix", "Spot": "spot"})), width="stretch")
                with st.expander("By month"):
                    bm = (ivd.assign(Month=ivd["date"].dt.strftime("%Y-%m")).groupby("Month")
                          .agg(Days=("atm_iv", "count"), **{"ATM IV mean": ("atm_iv", "mean"),
                               "ATM IV min": ("atm_iv", "min"), "ATM IV max": ("atm_iv", "max"),
                               "VIX mean": ("vix", "mean"), "VIX min": ("vix", "min"), "VIX max": ("vix", "max"),
                               "ATM IV − VIX mean": ("iv_minus_vix", "mean")}))
                    st.dataframe(num_fmt(bm), width="stretch")
                with st.expander("Every day (table)"):
                    tv = ivd[["date", "expiry", "dte", "time", "spot", "atm", "ce_iv", "pe_iv", "atm_iv",
                              "vix", "vix_open", "vix_high", "vix_low", "vix_close", "iv_minus_vix"]].copy()
                    tv.columns = ["Date", "Expiry", "Days to expiry", "Time", "Spot", "ATM", "CE IV %", "PE IV %", "ATM IV %",
                                  "VIX", "VIX open", "VIX high", "VIX low", "VIX close", "ATM IV − VIX"]
                    st.dataframe(tv.sort_values("Date", ascending=False), width="stretch", hide_index=True)
                    st.download_button("Download ATM IV by day (CSV)", tv.to_csv(index=False).encode(),
                                       file_name=f"atm_iv_{underlying.lower()}_{contract.lower()}_{tod_t:%H%M}.csv")

    # ============================== tab 3: statistics of the selected strategy
    with tab_stats:
        if s.empty:
            st.info("No data for this selection.")
        else:
            st.caption(f"pandas describe() over {len(s)} expiries: {strategy}{side_name(sel_side)} "
                       f"{offset_label(sel_off)} · {dte_text} · {tod_t:%H:%M}{iv_note} · {p_start:%d %b %Y} → {p_end:%d %b %Y}")
            cols = {f"{strategy} price": "price", "ATM IV %": "atm_iv", "India VIX": "vix", "Spot": "spot"}
            cols.update({f"Leg {i} price ({s[f'leg{i}'].iloc[-1].split(' ', 1)[0]} {s[f'leg{i}'].iloc[-1].split()[1]})":
                         f"leg{i}_price" for i in range(1, n_legs + 1)})
            st.dataframe(num_fmt(describe(s, cols)), width="stretch")

            k = st.columns(5)
            k[0].metric("Debit expiries", f"{(s['price'] > 0).mean() * 100:.0f}%")
            k[1].metric("Credit expiries", f"{(s['price'] < 0).mean() * 100:.0f}%")
            corr = s[["price", "atm_iv"]].corr().iloc[0, 1] if s["atm_iv"].notna().sum() > 2 else float("nan")
            k[2].metric("Corr. with ATM IV", "–" if pd.isna(corr) else f"{corr:+.2f}",
                        help="+1 = moves with IV, −1 = moves against IV, 0 = no relation")
            cv = s[["price", "vix"]].corr().iloc[0, 1] if s["vix"].notna().sum() > 2 else float("nan")
            k[3].metric("Corr. with VIX", "–" if pd.isna(cv) else f"{cv:+.2f}", help=VIX_NOTE)
            k[4].metric("Change", f"{s['price'].iloc[-1] - s['price'].iloc[0]:+,.2f}")

            st.markdown("**By month of expiry**")
            bm = (s.assign(Month=s["expiry"].dt.strftime("%Y-%m")).groupby("Month")
                  .agg(Expiries=("price", "count"), Mean=("price", "mean"), Median=("price", "median"),
                       Min=("price", "min"), Max=("price", "max"),
                       **{"Mean ATM IV %": ("atm_iv", "mean"), "Mean VIX": ("vix", "mean")}))
            st.dataframe(num_fmt(bm), width="stretch")