# Nifty Strategy Analyzer

A local app (runs in your browser) that shows option-strategy prices - butterfly, 1-3-2 ratio,
straddle, strangle, iron butterfly, single option, or any strategy you add - by days to expiry
and time of day, built from your 1-minute option-chain parquet files.

## Files

| File | What it is |
|---|---|
| `datasets.json` | **Your data sets**: one entry per index + contract type (folder, symbol, strike gap, lot size) |
| `build_data.py` | Step 1: builds every data set in `datasets.json` into its own folder in `app_data\` |
| `engine.py` | Prices strategies. **Strategies are defined at the top (`STRATEGIES`)** |
| `app.py` | The screen (Streamlit) |
| `build_data.bat` / `run_app.bat` | Double-click shortcuts for step 1 / opening the app |
| `requirements.txt` | Python libraries |
| `vix\` | Put India VIX files here (optional, see below) |

## 1. Install (once)

```
pip install -r requirements.txt
```

## 2. Build the data (once, then again after adding new days)

### Your data sets are kept separate

Each data set has its own folder of daily files and is built into its own folder - nothing is mixed:

| Data set (datasets.json) | Your folder | Built into |
|---|---|---|
| NIFTY Weekly | `D:\NIFTY_WEEKLY\2026\04\2026-04-01.parquet ...` | `app_data\nifty_weekly\` |
| NIFTY Monthly | `D:\NIFTY_MONTHLY\2026\04\2026-04-01.parquet ...` | `app_data\nifty_monthly\` |
| SENSEX Weekly / Monthly (later) | `D:\SENSEX_WEEKLY\...` | `app_data\sensex_weekly\` ... |
| India VIX | `vix\` folder or `--vix-download` | `app_data\vix\` (shared) |

Open **`datasets.json`** and set each `"folder"` to your real path (use `\\` in JSON paths, e.g.
`"D:\\NIFTY_MONTHLY"`). Data sets with `"enabled": false` are skipped - the SENSEX entries are ready
for later. Then run:

```
python build_data.py                         # all enabled data sets + VIX
python build_data.py --only nifty_monthly    # just one data set
python build_data.py --vix-download          # also download daily India VIX
```

or press Run on `build_data.py`, or double-click `build_data.bat`. For each data set it prints the
files and months found, the expiries (monthly ones marked **(M)**) and any file that failed.
Unchanged days are skipped next time; a day is rebuilt when its files change. `--rebuild` redoes all.
If a data set's folder is not found (e.g. the first time), the script **asks for the path in the
terminal** and saves it in `datasets.json` - press Enter to skip a data set. (In VS Code use the normal
Run button / terminal; the "Code Runner" output panel cannot take typed input.)

To see what your raw files contain - which expiries each day's file holds - run:

```
python build_data.py --inspect
```

After building, it also prints how many expiries each day's files hold. Next-week / far-week legs
(calendars) need **several expiries in one day's file**; if each file holds only one expiry, those legs
cannot be priced from it.

### Adding another index later (SENSEX, BANKNIFTY ...)

Add or enable an entry in `datasets.json`:

```json
{"id": "sensex_weekly", "underlying": "SENSEX", "contract": "Weekly", "label": "SENSEX Weekly",
 "folder": "D:\\SENSEX_WEEKLY", "symbol": "SENSEX", "strike_gap": 100, "lot_size": 20, "enabled": true}
```

- `id`: folder name in `app_data` (letters, digits, `_`)
- `underlying` / `contract`: what the app's *Underlying* and *Type of contract* lists show
- `symbol`: the value in your files' `symbol` column
- `strike_gap`: NIFTY 50, SENSEX 100, BANKNIFTY 100 ... (strategies, ATM and the builder follow it)
- `lot_size`: default for the rupee value (only used for the ₹ values shown)

Run `python build_data.py` (or `--only sensex_weekly`). No code changes needed.

## 3. Open the app

```
streamlit run app.py
```

or double-click `run_app.bat`. It opens at http://localhost:8501. (`app.py` cannot be started with
the editor's Run button - it needs `streamlit run`.)

## Using it

- **Strategy**: pick from the list (see below to add your own).
- **Days to expiry**: trading days before expiry (0 = expiry day). Holidays are taken from the
  weekdays missing in your data.
- **Underlying** and **Type of contract** (top right): pick the data set - e.g. NIFTY + Weekly uses only
  the NIFTY weekly data, NIFTY + Monthly only the monthly data. Each has its own strike gap and lot size.
- **Time of day**: the minute to price at (the last minute at or before it is used).
- **IV filter** (box under the selections): *Off*, *ATM IV (CE+PE average)*, *ATM CE IV* or *ATM PE IV*,
  with **IV from / to (%)**. Only the expiries whose IV - at your time of day on the priced day - is
  inside the range are kept, so you see the strategy's price for "this time of day AND this IV".
  The table, chart, statistics and downloads all use the kept expiries; it shows how many were kept
  and the IV range in the data. The ATM IV tab shades the range and counts the days inside it.
- **Table**: ATM+10 down to ATM−10, ATM taken from the spot at that moment. One-sided
  strategies (butterfly, ratio, single option) show a PE and a CE column; mixed strategies
  (straddle, strangle, iron butterfly) show one price column.
- **Click a price**: the chart shows that strategy for every past expiry at the same days to
  expiry and time. Hover a point for the date, spot and every leg's strike and price.
- **Right side has three tabs** for the price you clicked:
  - **Price history**: the strategy for every past expiry, with ATM IV (%) on a second axis
    (toggle). Hover a point for date, spot, ATM IV (CE / PE) and every leg.
    *See one expiry through the day* shows the strategy and ATM IV minute by minute, ATM IV
    from open to close, and a describe() table for that day.
  - **ATM IV by day**: ATM IV on every trading day at your time of day (nearest expiry),
    with CE and PE IV, a one-day lookup, describe() statistics, a by-month table and CSV download.
    ATM = spot rounded to 50 at that minute; ATM IV = average of `ce_iv` and `pe_iv` at that strike.
  - **Statistics (describe)**: pandas describe() of the strategy price, ATM IV, spot and each
    leg (count, mean, median, std dev, min, 5/25/75/95 %, max, range, skew), % of expiries with
    debit vs credit, correlation of the price with ATM IV, and a by-month table.
- **Downloads**: the selected history (CSV) or every strike for every expiry (Excel).
- **Table strike step** (sidebar): distance between table rows / builder strikes. NIFTY strikes
  exist every 50 points (the *strike gap*, fixed by the index); choose 100 to make ATM+1 = ATM+100.
- **Sidebar**: wing width (distance between legs), number of strikes, and whether the table
  shows the latest expiry, the average or the median.
- **Period** (sidebar): use only expiries in a period - All data, Last 3 / 6 / 12 months, or a
  custom from/to range. The table, chart, averages and downloads all follow it. Nothing is
  rebuilt or deleted; change it back any time.

Price = sum of (quantity × leg price): positive = net debit you pay, negative = net credit.
Leg prices use LTP, or the bid-ask mid when LTP is 0. IV comes from `ce_iv` / `pe_iv`
(0 is treated as missing; values stored as decimals like 0.14 are shown as 14%).
Data built before IV was added has no IV - delete `app_data` and run `build_data.py` again.

## Building or keeping only a period (optional)

The Period filter in the app is usually all you need. If you want the built data itself limited:

```
python build_data.py --from 2025-11-01 --to 2026-04-30           # build only these days
python build_data.py --from 2025-11-01 --to 2026-04-30 --clean   # also delete built days outside it
python build_data.py                                             # build everything again (restores)
```

`--clean` only touches `app_data\chain`; your NIFTY_WEEKLY files are never changed. To start
completely fresh, delete the `app_data` folder and run `build_data.py`.

## India VIX (optional)

VIX is not in the option-chain files, so it comes from separate files in the **`vix`** folder
next to these scripts. Use either or both:

1. **Daily VIX from NSE** (free): nseindia.com → Historical Data → India VIX → download CSV.
   Save the CSV(s) in `vix\` as they are - several files (e.g. one per year) are fine.
2. **1-minute VIX** from your broker: a CSV/Excel with a date-time column (or date + time
   columns) and a close/VIX column. With this, VIX is shown at your exact time of day and
   minute by minute in *through the day*.
3. **Automatic daily download** (Yahoo Finance `^INDIAVIX`):
   ```
   python build_data.py --vix-download
   ```

Then run `python build_data.py` (it re-reads the vix folder every time; `--vix <folder>` uses
another folder). The app shows VIX in the hover box, as a line on the charts, next to ATM IV
(with ATM IV − VIX) in *ATM IV by day*, in the one-day lookup (open/high/low/close and at your
time), and in the describe() tables and correlations.
"VIX" means the value at your time of day if you have 1-minute data, otherwise that day's close.

## Build your own strategy (on screen)

Choose **Custom (build your own)** in the Strategy list. A table appears - one row per leg:

| Column | Choices |
|---|---|
| Buy / Sell | Buy or Sell |
| Lots | 1, 2, 3 ... (quantity of that leg) |
| CE / PE | CE or PE |
| Strike | ATM+10 ... ATM ... ATM−10, relative to the centre strike of each table row |
| Expiry | **Near week** (the expiry picked by Days to expiry), **Next week**, **Far week (+2)**, **Week +3** (the 1st / 2nd / 3rd expiry after the near one, from the same data set), or **Chosen far (below)** (the far expiry you set with the week / month controls - can also come from another data set, e.g. NIFTY Monthly) |

Every leg can have its own expiry, so one strategy can mix near week, next week, far week and a
monthly leg. Example - *buy near-week CE ATM+2, sell next-week CE ATM+2, buy near-week PE ATM−2,
sell next-week PE ATM−2*:

| Buy / Sell | Lots | CE / PE | Strike | Expiry |
|---|---|---|---|---|
| Buy | 1 | CE | ATM+2 | Near week |
| Sell | 1 | CE | ATM+2 | Next week |
| Buy | 1 | PE | ATM−2 | Near week |
| Sell | 1 | PE | ATM−2 | Next week |

(This one is also built in as **Week calendar ATM±2 (buy near, sell next week)**.) Above the table the
app shows, for each extra expiry, which expiry it used and how many near expiries have its data.
Next / far week legs need each day's weekly file to contain those expiries' option chains too.

Add legs with the **+** under the table; remove one by selecting its row and pressing Delete.
The strategy is priced and shown exactly like the built-in ones (option-chain table, history chart,
through the day, ATM IV, VIX, statistics, downloads, near-week filter, far-leg week/month).
Price = sum of (±lots × leg price) in points; the **₹ value** = points × the index's `lot_size` in `datasets.json`.

Type a name and press **Save strategy**: it is stored in `custom_strategies.json` and appears in the
Strategy list as **Saved: <name>**. Open a saved one to see its legs, **Edit in builder**, or **Delete**.
Copy `custom_strategies.json` to another laptop to take your strategies with you.

## Adding or changing a strategy (engine.py → STRATEGIES)

### Calendar strategy (two expiries) - you choose the weeks

**Calendar CE/PE (near vs next month)** = sell near CE + buy far CE + buy near PE + sell far PE,
all at the same strike. When it is selected, extra controls appear:

- **Near expiry: week of month** (also works for every other strategy): use all weekly expiries,
  or only the 1st / 2nd / 3rd / 4th / 5th / last expiry of each month. Week numbers count the
  expiries of that calendar month found in your data (1st = first expiry of the month).
- **Far leg expiry**, either
  - **Week of a month**: *Far leg month* (same / next / 2 / 3 months ahead) and *Far leg week*
    (1st ... 5th, last) - e.g. "2nd week of next month", or "last week of same month"; or
  - **Weeks after near expiry**: the 1st, 2nd, 3rd ... expiry after the near one.

**Far leg data**: take the far legs from the same data set, or from another one of the same index -
e.g. near leg from NIFTY Weekly, far leg from NIFTY Monthly. The data sets stay separate; they are
only combined for that strategy's price.

The near leg is still chosen by *Days to expiry* and *Time of day*; the far leg is priced on the
same day and minute. Each day's parquet file must contain the option chain of the far expiry too
(several weekly expiries per file) - if it doesn't, or the chosen far expiry is not after the near
one, the app says so. The hover box, formulas and downloads show each leg's expiry.

Each leg is `(quantity, side, distance)` or `(quantity, side, distance, expiry)`:

- `quantity`: `+1` buy, `-3` sell 3, ...
- `side`: `"X"` = the CE or PE column you click; `"CE"` / `"PE"` = always that side
- `distance`: from the centre strike in wing units (`-1` = K − wing, `0` = K, `+2` = K + 2×wing)
- `"units": "steps"`: distances in strike steps (× the index's strike gap) instead of wing units
- `"mirror_pe": True` flips the distances for the PE version (put = mirror of the call)
- `expiry`: `"near"` (default), `"w1"` next week, `"w2"` far week (+2), `"w3"` week +3, or `"far"`; `"far_expiry"` picks which later expiry:
  `"next_month"` (monthly expiry of the next month), `"next_monthly"` (first monthly after near),
  `"next_weekly"` (the very next expiry)

Example - a 1-2 call ratio spread (buy 1 at K, sell 2 one wing higher):

```python
"Ratio 1-2": {
    "legs": [(+1, "X", 0), (-2, "X", +1)],
    "mirror_pe": True,
    "about": "Buy 1 at the centre strike, sell 2 one wing further out",
},
```

Save the file and reload the browser page - the new strategy appears in the list. No other
file needs changing. If a strategy's legs reach beyond about 1000 points from spot, raise
`STRIKE_WINDOW` in `build_data.py` and rebuild with `--rebuild`.
