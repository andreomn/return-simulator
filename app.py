import os
import re
import io
import math
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, date as date_cls, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, request, Response
from google.cloud import storage

app = Flask(__name__)

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "debentures-anbima-am")
GCS_PREFIX_PRE = os.environ.get("GCS_PREFIX_PRE", "B3-predi/")
GCS_PREFIX_FX = os.environ.get("GCS_PREFIX_FX", "b3-fx/")
GCS_HOLIDAYS_FILENAME = os.environ.get("GCS_HOLIDAYS_FILENAME", "holidays.xlsx")
CACHE_SECONDS = int(os.environ.get("CACHE_SECONDS", "600"))

_cache = {"ts": 0, "curve_date_key": None, "pre": None, "fx": None, "pre_date": None, "fx_date": None, "pre_name": None, "fx_name": None}
_holidays_cache = {"ts": 0, "holidays": None, "name": None, "count": 0}


def norm_prefix(prefix: str) -> str:
    return prefix.strip("/") + "/"


def extract_date_from_name(name: str) -> Optional[pd.Timestamp]:
    m = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", name)
    if m:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(\d{2})[-_](\d{2})[-_](20\d{2})", name)
    if m:
        return pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return None


def read_curve_csv_bytes(data: bytes) -> pd.DataFrame:
    last_error = None
    for enc in ["utf-8-sig", "latin1", "cp1252", "utf-8"]:
        try:
            df = pd.read_csv(io.BytesIO(data), sep=";", encoding=enc, decimal=",")
            if len(df) > 0 and df.shape[1] >= 4:
                break
        except Exception as e:
            last_error = e
    else:
        raise RuntimeError(f"Could not parse CSV: {last_error}")

    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if "descr" in lc:
            rename[c] = "description"
        elif "úteis" in lc or "uteis" in lc:
            rename[c] = "business_days"
        elif "corridos" in lc:
            rename[c] = "calendar_days"
        elif "pre" in lc or "taxa" in lc:
            rename[c] = "rate"
    df = df.rename(columns=rename)
    needed = ["description", "business_days", "calendar_days", "rate"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns {missing}; got {df.columns.tolist()}")
    df = df[needed].copy()
    df["business_days"] = pd.to_numeric(df["business_days"], errors="coerce")
    df["calendar_days"] = pd.to_numeric(df["calendar_days"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["business_days", "calendar_days", "rate"])
    df = df.sort_values("calendar_days").drop_duplicates("calendar_days", keep="first")
    return df.reset_index(drop=True)


def curve_from_gcs(prefix: str, curve_date: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, pd.Timestamp, str]:
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=norm_prefix(prefix)))
    csv_blobs = []
    for blob in blobs:
        if not blob.name.lower().endswith(".csv"):
            continue
        dt = extract_date_from_name(blob.name)
        if dt is not None:
            csv_blobs.append((dt.normalize(), blob))
    if not csv_blobs:
        raise RuntimeError(f"No CSV files found in gs://{GCS_BUCKET_NAME}/{prefix}")

    if curve_date is not None:
        target = pd.Timestamp(curve_date).normalize()
        exact = [(dt, blob) for dt, blob in csv_blobs if dt == target]
        if not exact:
            available = sorted({dt.strftime("%Y-%m-%d") for dt, _ in csv_blobs}, reverse=True)[:10]
            raise RuntimeError(
                f"No curve found for {target.strftime('%Y-%m-%d')} in gs://{GCS_BUCKET_NAME}/{prefix}. "
                f"Latest available dates: {', '.join(available)}"
            )
        dt, blob = exact[0]
    else:
        dt, blob = sorted(csv_blobs, key=lambda x: x[0], reverse=True)[0]

    data = blob.download_as_bytes()
    df = read_curve_csv_bytes(data)
    return df, dt, blob.name


def parse_curve_date(value: str) -> Optional[pd.Timestamp]:
    value = (value or "").strip()
    if not value:
        return None
    return pd.Timestamp(value).normalize()


def parse_user_number(value, default=0.0) -> float:
    """Accepts 100000000, 100,000,000, 100.000.000, 3,00, etc."""
    if value is None:
        return float(default)
    s = str(value).strip()
    if not s:
        return float(default)
    s = s.replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts[-1]) == 3 and len(parts) > 1:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts[-1]) == 3 and len(parts) > 1:
            s = s.replace(".", "")
    return float(s)


def get_curves(curve_date: Optional[pd.Timestamp] = None):
    now = datetime.utcnow().timestamp()
    curve_date_key = pd.Timestamp(curve_date).strftime("%Y-%m-%d") if curve_date is not None else "latest"
    if _cache["pre"] is not None and _cache.get("curve_date_key") == curve_date_key and now - _cache["ts"] < CACHE_SECONDS:
        return _cache["pre"], _cache["fx"], _cache["pre_date"], _cache["fx_date"], _cache["pre_name"], _cache["fx_name"]
    pre, pre_date, pre_name = curve_from_gcs(GCS_PREFIX_PRE, curve_date)
    fx, fx_date, fx_name = curve_from_gcs(GCS_PREFIX_FX, curve_date)
    _cache.update({"ts": now, "curve_date_key": curve_date_key, "pre": pre, "fx": fx, "pre_date": pre_date, "fx_date": fx_date, "pre_name": pre_name, "fx_name": fx_name})
    return pre, fx, pre_date, fx_date, pre_name, fx_name


def excel_serial_to_date(value: float) -> Optional[date_cls]:
    try:
        v = float(value)
    except Exception:
        return None
    if v < 20000 or v > 90000:
        return None
    return date_cls(1899, 12, 30) + timedelta(days=int(v))


def read_shared_strings(z: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    out = []
    for si in root.findall("a:si", ns):
        texts = [t.text or "" for t in si.findall(".//a:t", ns)]
        out.append("".join(texts))
    return out


def parse_holidays_xlsx_bytes(data: bytes) -> list[date_cls]:
    holidays = set()
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        shared = read_shared_strings(z)
        sheet_names = [n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
        for sheet_name in sheet_names:
            root = ET.fromstring(z.read(sheet_name))
            for c in root.findall(".//a:c", ns):
                ref = c.attrib.get("r", "")
                col = re.sub(r"\d", "", ref).upper()
                if col != "A":
                    continue
                v = c.find("a:v", ns)
                if v is None or v.text is None:
                    continue
                raw = v.text.strip()
                parsed = None
                if c.attrib.get("t") == "s":
                    try:
                        raw = shared[int(raw)]
                    except Exception:
                        raw = ""
                    try:
                        parsed = pd.Timestamp(raw, dayfirst=True).date()
                    except Exception:
                        parsed = None
                else:
                    parsed = excel_serial_to_date(raw)
                    if parsed is None:
                        try:
                            parsed = pd.Timestamp(raw, dayfirst=True).date()
                        except Exception:
                            parsed = None
                if parsed is not None and 1990 <= parsed.year <= 2100:
                    holidays.add(parsed)
    if not holidays:
        raise RuntimeError("Could not read any holiday dates from holidays.xlsx. Expected dates in column A.")
    return sorted(holidays)


def get_holidays() -> list[date_cls]:
    now = datetime.utcnow().timestamp()
    if _holidays_cache["holidays"] is not None and now - _holidays_cache["ts"] < CACHE_SECONDS:
        return _holidays_cache["holidays"]
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_HOLIDAYS_FILENAME)
    if not blob.exists():
        raise RuntimeError(f"Holiday file not found: gs://{GCS_BUCKET_NAME}/{GCS_HOLIDAYS_FILENAME}")
    data = blob.download_as_bytes()
    holidays = parse_holidays_xlsx_bytes(data)
    _holidays_cache.update({"ts": now, "holidays": holidays, "name": GCS_HOLIDAYS_FILENAME, "count": len(holidays)})
    return holidays


def interp_curve(df: pd.DataFrame, target_days: float, x_col: str = "calendar_days") -> float:
    if df.empty:
        return 0.0
    target_days = max(float(target_days), 1.0)
    if target_days <= df[x_col].min():
        return float(df.iloc[0]["rate"])
    if target_days >= df[x_col].max():
        return float(df.iloc[-1]["rate"])
    return float(np.interp(target_days, df[x_col].values, df["rate"].values))


def curve_days_to_date(curve_dt: pd.Timestamp, target_dt: pd.Timestamp) -> int:
    """Calendar days from the selected curve date to a specific table date."""
    return max((pd.Timestamp(target_dt).normalize() - pd.Timestamp(curve_dt).normalize()).days, 1)


def fx_at_table_date(fx_df: pd.DataFrame, fx_curve_dt: pd.Timestamp, target_dt: pd.Timestamp) -> float:
    """FX forward from the selected FX curve for the actual date shown in the table."""
    return interp_curve(fx_df, curve_days_to_date(fx_curve_dt, target_dt), "calendar_days")


def parse_month(value: str) -> pd.Timestamp:
    value = (value or "").strip()
    if not value:
        today = pd.Timestamp.today().normalize()
        return pd.Timestamp(today.year, today.month, 1)
    if re.match(r"^\d{4}-\d{2}$", value):
        y, m = value.split("-")
        return pd.Timestamp(int(y), int(m), 1)
    return pd.Timestamp(value).replace(day=1)


def month_end(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts) + pd.offsets.MonthEnd(0)


def quarter_end_schedule(disbursement_month: pd.Timestamp, maturity_months: int) -> list[pd.Timestamp]:
    """
    Builds the deal schedule every 3 months from the disbursement month.
    Example: if disbursement is Apr-26, the next columns are Jul-26, Oct-26, Jan-27, etc.
    This is intentionally not aligned to calendar quarter-ends such as Jun/Sep/Dec.
    """
    maturity_eom = month_end(disbursement_month + pd.DateOffset(months=maturity_months))

    dates = []
    q = month_end(disbursement_month + pd.DateOffset(months=3))

    while q < maturity_eom:
        dates.append(pd.Timestamp(q))
        q = month_end(q + pd.DateOffset(months=3))

    if not dates or dates[-1] != maturity_eom:
        dates.append(maturity_eom)

    return dates
def busdays_between(a: pd.Timestamp, b: pd.Timestamp, holidays: Optional[list[date_cls]] = None) -> int:
    start = pd.Timestamp(a).date()
    end = pd.Timestamp(b).date()
    if holidays:
        holiday_arr = np.array(holidays, dtype="datetime64[D]")
        return max(int(np.busday_count(start, end, holidays=holiday_arr)), 1)
    return max(int(np.busday_count(start, end)), 1)




def busdays_from_curve_date(curve_dt: pd.Timestamp, target_dt: pd.Timestamp, holidays: Optional[list[date_cls]] = None) -> int:
    """Business days from the selected curve date to the actual table date.
    This is used to read the PRE zero curve at each table date, not at days
    counted only from the disbursement date.
    """
    start = pd.Timestamp(curve_dt).normalize().date()
    end = pd.Timestamp(target_dt).normalize().date()
    if end <= start:
        return 0
    if holidays:
        holiday_arr = np.array(holidays, dtype="datetime64[D]")
        return max(int(np.busday_count(start, end, holidays=holiday_arr)), 0)
    return max(int(np.busday_count(start, end)), 0)

def xirr(cashflows):
    if not cashflows:
        return None
    vals = [x[1] for x in cashflows]
    if not any(v > 0 for v in vals) or not any(v < 0 for v in vals):
        return None
    t0 = cashflows[0][0]

    def npv(r):
        total = 0.0
        for dt, cf in cashflows:
            years = max((dt - t0).days / 365.0, 0.0)
            total += cf / ((1 + r) ** years)
        return total

    lo, hi = -0.95, 10.0
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-7:
            return mid
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2


def is_zero(x):
    try:
        return abs(float(x)) < 0.0000001
    except Exception:
        return False


def dynamic_decimals(x, default_decimals=1):
    try:
        return 0 if abs(float(x)) >= 100 else default_decimals
    except Exception:
        return default_decimals


def fmt_num(x, decimals=1):
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return "-"
    if is_zero(x):
        return "-"
    val = float(x)
    d = dynamic_decimals(val, decimals)
    txt = f"{abs(val):,.{d}f}"
    return f"({txt})" if val < 0 else txt


def fmt_money(x, decimals=1):
    """Money formatter without currency symbols, per user request."""
    return fmt_num(x, decimals)


def to_brl_mm(x):
    try:
        return float(x) / 1_000_000.0
    except Exception:
        return x


def to_usd_mm(x):
    try:
        return float(x) / 1_000_000.0
    except Exception:
        return x


def fmt_pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)) or pd.isna(x):
        return "-"
    if is_zero(x):
        return "-"
    val = float(x) * 100.0
    txt = f"{abs(val):.2f}%"
    return f"({txt})" if val < 0 else txt


def fmt_int(x):
    if x is None or pd.isna(x) or is_zero(x):
        return "-"
    val = int(x)
    txt = f"{abs(val):,}"
    return f"({txt})" if val < 0 else txt


def zero_rate_at_business_day(pre_df: pd.DataFrame, business_day: float) -> float:
    """Interpolates the accumulated PRE/CDI zero rate by business days."""
    return interp_curve(pre_df, max(float(business_day), 1.0), "business_days") / 100.0


def implied_period_cdi(pre_df: pd.DataFrame, start_bd: int, end_bd: int) -> float:
    """
    Converts the accumulated PRE curve into an implied CDI rate for the period.
    PRE points are zero/accumulated annual rates. For a period [t0, t1]:
      CDI factor = (1+r1)^(du1/252) / (1+r0)^(du0/252)
      CDI annualized = factor^(252/(du1-du0)) - 1
    """
    start_bd = max(int(start_bd), 0)
    end_bd = max(int(end_bd), start_bd + 1)
    period_bd = max(end_bd - start_bd, 1)

    r0 = zero_rate_at_business_day(pre_df, start_bd) if start_bd > 0 else 0.0
    r1 = zero_rate_at_business_day(pre_df, end_bd)

    acc0 = (1 + r0) ** (start_bd / 252.0) if start_bd > 0 else 1.0
    acc1 = (1 + r1) ** (end_bd / 252.0)

    cdi_factor = acc1 / acc0
    return cdi_factor ** (252.0 / period_bd) - 1




def annual_rate_to_daily_rate(annual_rate: float) -> float:
    """Effective daily rate using 252 business days: (1 + annual rate)^(1/252) - 1."""
    return (1.0 + float(annual_rate)) ** (1.0 / 252.0) - 1.0


def period_interest_rate_from_annual_rate(annual_rate: float, business_days: int) -> float:
    """Period rate from effective daily compounding over business days."""
    daily_rate = annual_rate_to_daily_rate(annual_rate)
    return (1.0 + daily_rate) ** max(int(business_days), 1) - 1.0


def simulate(form):
    curve_date = parse_curve_date(form.get("curve_date", ""))
    pre, fx, pre_date, fx_date, pre_name, fx_name = get_curves(curve_date)
    holidays = get_holidays()
    deal_size_mm = parse_user_number(form.get("deal_size", 100), 100)
    deal_size = deal_size_mm * 1_000_000.0
    spread_pct = parse_user_number(form.get("spread", 3.00), 3.00) / 100.0
    disb_month = parse_month(form.get("disbursement_month", ""))
    disb_date = month_end(disb_month)
    maturity_months = int(form.get("maturity_months", 36))
    amortization = form.get("amortization", "bullet")
    upfront_fee = parse_user_number(form.get("upfront_fee", 0.00), 0.00) / 100.0
    extension_fee_pct = parse_user_number(form.get("extension_fee", 0.00), 0.00) / 100.0
    extension_fee_month_raw = str(form.get("extension_fee_month", "") or "").strip()
    extension_fee_month = int(extension_fee_month_raw) if extension_fee_month_raw else 0
    extension_fee_treatment = str(form.get("extension_fee_treatment", "pik") or "pik").lower()
    extension_fee_is_cash = extension_fee_treatment == "cash"

    extension_fee_pct_2 = parse_user_number(form.get("extension_fee_2", 0.00), 0.00) / 100.0
    extension_fee_month_raw_2 = str(form.get("extension_fee_month_2", "") or "").strip()
    extension_fee_month_2 = int(extension_fee_month_raw_2) if extension_fee_month_raw_2 else 0
    extension_fee_treatment_2 = str(form.get("extension_fee_treatment_2", "pik") or "pik").lower()
    extension_fee_is_cash_2 = extension_fee_treatment_2 == "cash"

    interest_frequency_months = int(form.get("interest_frequency_months", 3) or 3)

    payment_dates = quarter_end_schedule(disb_month, maturity_months)
    # All FX rates, including disbursement FX, come from the selected FX curve
    # using the actual table date (month-end) versus the selected curve date.
    initial_fx = fx_at_table_date(fx, fx_date, disb_date)
    principal_step = deal_size / len(payment_dates) if amortization == "linear" else 0.0

    rows = []
    brl_cashflows = []
    usd_cashflows = []

    # Day-one disbursement column. This is the first cash flow and the debt starts here.
    disb_upfront_fee = deal_size * upfront_fee
    disb_total_cf_brl = -deal_size + disb_upfront_fee
    rows.append({
        "payment_date": disb_date,
        "period_label": disb_date.strftime("%b-%y"),
        "quarter": f"{disb_date.year}Q{((disb_date.month - 1) // 3) + 1}",
        "days_from_disbursement": 0,
        "business_days_from_disbursement": 0,
        "period_business_days": 0,
        "pre_rate_start": 0.0,
        "pre_rate_end": 0.0,
        "cdi_period_rate": 0.0,
        "spread": spread_pct,
        "deal_rate": 0.0,
        "debt_bop_brl": 0.0,
        "issuance_brl": deal_size,
        "interest_accrual_brl": 0.0,
        "extension_fee_accrual_brl": 0.0,
        "cash_extension_fee_brl": 0.0,
        "extension_fee_accrual_brl_2": 0.0,
        "cash_extension_fee_brl_2": 0.0,
        "cash_interest_brl": 0.0,
        "principal_brl": 0.0,
        "debt_eop_brl": deal_size,
        "upfront_fee_brl": disb_upfront_fee,
        "disbursement_brl": -deal_size,
        "total_cf_brl": disb_total_cf_brl,
        "fx_forward": initial_fx,
        "cash_interest_usd": 0.0,
        "cash_extension_fee_usd": 0.0,
        "cash_extension_fee_usd_2": 0.0,
        "principal_usd": 0.0,
        "upfront_fee_usd": disb_upfront_fee / initial_fx if initial_fx else 0.0,
        "disbursement_usd": -deal_size / initial_fx if initial_fx else 0.0,
        "total_cf_usd": disb_total_cf_brl / initial_fx if initial_fx else 0.0,
        "is_disbursement": True,
    })
    brl_cashflows.append((disb_date.to_pydatetime().date(), disb_total_cf_brl))
    usd_cashflows.append((disb_date.to_pydatetime().date(), disb_total_cf_brl / initial_fx if initial_fx else 0.0))

    outstanding = deal_size
    prev_bd_from_start = 0
    prev_pre_curve_bd = busdays_from_curve_date(pre_date, disb_date, holidays)
    accrued_interest_balance = 0.0
    last_interest_payment_months = 0

    for i, pay_date in enumerate(payment_dates, start=1):
        days_from_start = max((pay_date - disb_date).days, 1)
        bd_from_start = max(busdays_between(disb_date, pay_date, holidays), 1)
        period_bd = max(bd_from_start - prev_bd_from_start, 1)

        # PRE/CDI must be read from the selected PRE curve using the actual
        # table dates. Therefore, the start and end points are business days
        # from the PRE curve date to the previous/current table dates, not
        # business days counted only from disbursement.
        pay_pre_curve_bd = busdays_from_curve_date(pre_date, pay_date, holidays)
        pre_rate_start = zero_rate_at_business_day(pre, prev_pre_curve_bd) if prev_pre_curve_bd > 0 else 0.0
        pre_rate_end = zero_rate_at_business_day(pre, pay_pre_curve_bd) if pay_pre_curve_bd > 0 else 0.0
        cdi_period_rate = implied_period_cdi(pre, prev_pre_curve_bd, pay_pre_curve_bd)

        deal_rate = (1 + cdi_period_rate) * (1 + spread_pct) - 1

        # Effective daily interest accrual:
        # daily rate = (1 + annual total rate)^(1/252) - 1
        # period interest rate = (1 + daily rate)^(period business days) - 1
        period_interest_rate = period_interest_rate_from_annual_rate(deal_rate, period_bd)

        debt_bop = outstanding
        issuance = 0.0
        interest_accrual = debt_bop * period_interest_rate
        accrued_interest_balance += interest_accrual

        months_since_disb = max((pay_date.year - disb_date.year) * 12 + (pay_date.month - disb_date.month), 0)
        is_extension_fee_date = extension_fee_pct > 0 and extension_fee_month > 0 and months_since_disb == extension_fee_month
        extension_fee_amount = debt_bop * extension_fee_pct if is_extension_fee_date else 0.0
        extension_fee_accrual = extension_fee_amount if is_extension_fee_date and not extension_fee_is_cash else 0.0
        cash_extension_fee = extension_fee_amount if is_extension_fee_date and extension_fee_is_cash else 0.0

        is_extension_fee_date_2 = extension_fee_pct_2 > 0 and extension_fee_month_2 > 0 and months_since_disb == extension_fee_month_2
        extension_fee_amount_2 = debt_bop * extension_fee_pct_2 if is_extension_fee_date_2 else 0.0
        extension_fee_accrual_2 = extension_fee_amount_2 if is_extension_fee_date_2 and not extension_fee_is_cash_2 else 0.0
        cash_extension_fee_2 = extension_fee_amount_2 if is_extension_fee_date_2 and extension_fee_is_cash_2 else 0.0

        is_final = i == len(payment_dates)

        if interest_frequency_months >= 999:
            # Bullet interest: pay only at maturity.
            is_cash_interest_date = is_final
        else:
            # Count interest payment dates from the disbursement month, not from the first
            # calendar quarter-end. Example: Apr-26 + annual = Apr-27, Apr-28, etc.
            is_cash_interest_date = months_since_disb > 0 and months_since_disb % interest_frequency_months == 0
            if is_final:
                is_cash_interest_date = True

        cash_interest = accrued_interest_balance if is_cash_interest_date else 0.0
        if is_cash_interest_date:
            accrued_interest_balance = 0.0
            last_interest_payment_months = months_since_disb

        debt_before_amort = debt_bop + issuance + interest_accrual + extension_fee_accrual + extension_fee_accrual_2 - cash_interest
        if amortization == "bullet":
            principal = debt_before_amort if is_final else 0.0
        else:
            principal = min(debt_before_amort, principal_step if not is_final else debt_before_amort)

        debt_eop = debt_before_amort - principal
        total_cf = cash_interest + cash_extension_fee + cash_extension_fee_2 + principal
        fx_rate = fx_at_table_date(fx, fx_date, pay_date)
        usd_cf = total_cf / fx_rate if fx_rate else 0.0

        rows.append({
            "payment_date": pay_date,
            "period_label": pay_date.strftime("%b-%y"),
            "quarter": f"{pay_date.year}Q{((pay_date.month - 1) // 3) + 1}",
            "days_from_disbursement": days_from_start,
            "business_days_from_disbursement": bd_from_start,
            "period_business_days": period_bd,
            "pre_rate_start": pre_rate_start,
            "pre_rate_end": pre_rate_end,
            "cdi_period_rate": cdi_period_rate,
            "spread": spread_pct,
            "deal_rate": deal_rate,
            "debt_bop_brl": debt_bop,
            "issuance_brl": issuance,
            "interest_accrual_brl": interest_accrual,
            "extension_fee_accrual_brl": extension_fee_accrual,
            "cash_extension_fee_brl": cash_extension_fee,
            "extension_fee_accrual_brl_2": extension_fee_accrual_2,
            "cash_extension_fee_brl_2": cash_extension_fee_2,
            "cash_interest_brl": cash_interest,
            "principal_brl": principal,
            "debt_eop_brl": debt_eop,
            "upfront_fee_brl": 0.0,
            "disbursement_brl": 0.0,
            "total_cf_brl": total_cf,
            "fx_forward": fx_rate,
            "cash_interest_usd": cash_interest / fx_rate if fx_rate else 0.0,
            "cash_extension_fee_usd": cash_extension_fee / fx_rate if fx_rate else 0.0,
            "cash_extension_fee_usd_2": cash_extension_fee_2 / fx_rate if fx_rate else 0.0,
            "principal_usd": principal / fx_rate if fx_rate else 0.0,
            "upfront_fee_usd": 0.0,
            "disbursement_usd": 0.0,
            "total_cf_usd": usd_cf,
            "is_disbursement": False,
        })
        brl_cashflows.append((pay_date.to_pydatetime().date(), total_cf))
        usd_cashflows.append((pay_date.to_pydatetime().date(), usd_cf))
        outstanding = debt_eop
        prev_bd_from_start = bd_from_start
        prev_pre_curve_bd = pay_pre_curve_bd

    df = pd.DataFrame(rows)
    summary = {
        "pre_date": pre_date.strftime("%Y-%m-%d"),
        "fx_date": fx_date.strftime("%Y-%m-%d"),
        "pre_name": pre_name,
        "fx_name": fx_name,
        "deal_size": deal_size,
        "spread_pct": spread_pct,
        "disbursement_date": disb_date.strftime("%Y-%m-%d"),
        "maturity_date": payment_dates[-1].strftime("%Y-%m-%d") if payment_dates else "-",
        "total_interest": float(df["cash_interest_brl"].sum()) if not df.empty else 0.0,
        "total_brl": float(df["total_cf_brl"].sum()) if not df.empty else 0.0,
        "total_usd": float(df["total_cf_usd"].sum()) if not df.empty else 0.0,
        "curve_date": curve_date.strftime("%Y-%m-%d") if curve_date is not None else pre_date.strftime("%Y-%m-%d"),
        "initial_fx": initial_fx,
        "holidays_count": _holidays_cache.get("count", len(holidays)),
        "holidays_name": _holidays_cache.get("name", GCS_HOLIDAYS_FILENAME),
        "irr_brl": xirr(brl_cashflows),
        "irr_usd": xirr(usd_cashflows),
    }
    return summary, df


def table_row(label, unit, values, kind="number", decimals=1, strong=False, italic=False):
    row_classes = []
    if strong:
        row_classes.append("strong-row")
    if italic:
        row_classes.append("italic-row")
    row_cls = " ".join(row_classes)
    cells = [f"<th class='metric'>{label}</th><td class='unit'>{unit}</td>"]
    for v in values:
        if kind == "date":
            txt = pd.Timestamp(v).strftime("%d-%b-%y")
        elif kind == "pct":
            txt = fmt_pct(v)
        elif kind == "fx":
            txt = "-" if is_zero(v) else fmt_num(v, 2)
        elif kind == "brl":
            txt = fmt_money(to_brl_mm(v), decimals)
        elif kind == "usd_mm":
            txt = fmt_money(to_usd_mm(v), decimals)
        elif kind == "usd":
            txt = fmt_money(v, decimals)
        elif kind == "int":
            txt = fmt_int(v)
        elif kind == "text":
            txt = str(v) if pd.notna(v) and str(v).strip() else "-"
        else:
            txt = fmt_num(v, decimals)
        cells.append(f"<td>{txt}</td>")
    return f"<tr class='{row_cls}'>" + "".join(cells) + "</tr>"


def table_spacer(title=""):
    label = f"<th class='metric spacer-label'>{title}</th>" if title else "<th class='metric spacer-label'>&nbsp;</th>"
    return "<tr class='spacer'>" + label + "<td class='unit'>&nbsp;</td><td colspan='999'>&nbsp;</td></tr>"


def table_blank():
    return "<tr class='blank-row'><th class='metric'>&nbsp;</th><td class='unit'>&nbsp;</td><td colspan='999'>&nbsp;</td></tr>"


def render_horizontal_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<div class='hint'>Fill the inputs and click Simulate.</div>"
    header = "<tr><th class='metric'>Metric</th><th class='unit'>Unit</th>" + "".join([f"<th>{x}</th>" for x in df["period_label"]]) + "</tr>"
    rows = []

    # 1) Rates and support
    rows.append(table_spacer("Rates & Support"))
    rows.append(table_row("Payment date", "Date", df["payment_date"], "date"))
    rows.append(table_row("Quarter", "Text", df["quarter"], "text"))
    rows.append(table_row("Period business days", "BD", df["period_business_days"], "int"))
    rows.append(table_row("PRE rate at period start", "% p.a.", df["pre_rate_start"], "pct"))
    rows.append(table_row("PRE rate at period end", "% p.a.", df["pre_rate_end"], "pct"))
    rows.append(table_row("Implied CDI for period", "% p.a.", df["cdi_period_rate"], "pct"))
    rows.append(table_row("Spread over CDI", "% p.a.", df["spread"], "pct"))
    rows.append(table_row("Total Rate", "% p.a.", df["deal_rate"], "pct"))
    rows.append(table_blank())
    rows.append(table_blank())

    # 2) Debt balance bridge
    rows.append(table_spacer("Debt balance bridge"))
    rows.append(table_row("Debt BoP", "BRL mm", df["debt_bop_brl"], "brl"))
    rows.append(table_row("(+) Issuance", "BRL mm", df["issuance_brl"], "brl"))
    rows.append(table_row("(+) Interest Accrual", "BRL mm", df["interest_accrual_brl"], "brl"))
    rows.append(table_row("(+) Extension Fee #1", "BRL mm", df["extension_fee_accrual_brl"], "brl"))
    rows.append(table_row("(+) Extension Fee #2", "BRL mm", df["extension_fee_accrual_brl_2"], "brl"))
    rows.append(table_row("(-) Cash Interest", "BRL mm", -df["cash_interest_brl"], "brl"))
    rows.append(table_row("(-) Debt Amortization", "BRL mm", -df["principal_brl"], "brl"))
    rows.append(table_row("(=) Debt EoP", "BRL mm", df["debt_eop_brl"], "brl", strong=True))
    rows.append(table_blank())
    rows.append(table_blank())

    # 3) Cash flow and FX conversion
    rows.append(table_spacer("Cash flow / FX"))
    rows.append(table_row("(-) Disbursement", "BRL mm", df["disbursement_brl"], "brl"))
    rows.append(table_row("(+) OID", "BRL mm", df["upfront_fee_brl"], "brl"))
    rows.append(table_row("(+) Cash Interest", "BRL mm", df["cash_interest_brl"], "brl"))
    rows.append(table_row("(+) Extension Fee #1", "BRL mm", df["cash_extension_fee_brl"], "brl"))
    rows.append(table_row("(+) Extension Fee #2", "BRL mm", df["cash_extension_fee_brl_2"], "brl"))
    rows.append(table_row("(+) Debt Repayment", "BRL mm", df["principal_brl"], "brl"))
    rows.append(table_row("(=) Total Debt Cash Flow", "BRL mm", df["total_cf_brl"], "brl", strong=True))
    rows.append(table_row("BRL/USD forward FX", "BRL/USD", df["fx_forward"], "fx", italic=True))
    rows.append(table_row("Total Debt Cash Flow", "USD mm", df["total_cf_usd"], "usd_mm", strong=True))

    return f"<div class='horizontal-wrap'><table class='horizontal'><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table></div>"


CSS = """
:root{--navy:#100058;--navy-2:#100058;--line:#cfd6e6;--soft:#f5f6fb;--text:#1c1f37;--muted:#5f6782;--white:#fff;--green:#eaf9f0;--greenline:#bde2cb}
*{box-sizing:border-box}
body{margin:0;font-family:Georgia,'Times New Roman',serif;color:var(--text);background:#ececf1}
.top{max-width:1680px;margin:22px auto 0;border-radius:10px;background:linear-gradient(90deg,var(--navy),var(--navy-2));color:#fff;padding:22px 28px}
.top h1{margin:0;font-size:28px;font-weight:700;letter-spacing:.2px}
.top p{margin:10px 0 0;font-size:12px;opacity:.95}
.wrap{max-width:1680px;margin:0 auto;padding:18px 30px 28px}
.model-layout{display:grid;grid-template-columns:380px minmax(900px,1fr);gap:20px;align-items:start}
.side-panel{position:sticky;top:18px}
.side-card,.table-card{border:1px solid var(--line);border-radius:8px;padding:20px;background:var(--white)}
.side-card h2{margin:0 0 14px;font-size:18px;color:var(--navy);text-align:center}
.terms-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 14px}
.term-row.full{grid-column:1/-1}
.fee-card{grid-column:1/-1;border:1px solid var(--line);border-radius:8px;padding:12px;background:#fbfbfe}
.fee-card-grid{display:grid;grid-template-columns:1fr 1fr 92px;gap:12px;align-items:end}
.fee-card .inline-check{display:grid;gap:8px;margin-top:0}
.fee-card .inline-check label{font-size:14px;color:var(--text)}
.inline-check{display:flex;align-items:center;gap:12px;margin-top:5px}
.inline-check label{display:flex;align-items:center;gap:5px;margin:0;font-size:12px;color:var(--text)}
.inline-check input{width:auto;padding:0;margin:0}
.bottom-summary{margin-top:16px}
.term-row label{font-size:13px;color:var(--muted);display:block;margin-bottom:6px}
.term-row input,.term-row select{width:100%;padding:10px;border:1px solid #aeb8d0;border-radius:4px;font-size:14px;font-family:inherit}
button{background:var(--navy);color:#fff;border:1px solid var(--navy);padding:12px 22px;border-radius:999px;font-weight:700;cursor:pointer;width:100%;margin-top:4px}
.kpis{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.kpi{background:var(--soft);border:1px solid var(--line);border-radius:8px;padding:12px}
.kpi .t{font-size:12px;color:var(--muted)}
.kpi .v{font-size:18px;font-weight:700;margin-top:5px;color:var(--navy)}
.hint{color:var(--muted);font-size:12px;margin-top:10px;line-height:1.45}
.error{background:#fff3f3;color:#8a1f1f;border:1px solid #ffd0d0;border-radius:8px;padding:14px;margin-bottom:18px}
.table-card{min-width:0;overflow:hidden}
.horizontal-wrap{overflow:auto;border:1px solid var(--line);border-radius:8px;max-height:540px}
.horizontal{border-collapse:separate;border-spacing:0;font-size:12.5px;min-width:1250px}
.horizontal th,.horizontal td{border-bottom:1px solid #e3e7f2;border-right:1px solid #e3e7f2;padding:9px 10px;text-align:center;white-space:nowrap;font-weight:400}
.horizontal thead th{background:var(--navy);color:#f8fafc;position:sticky;top:0;z-index:3;font-weight:700}
.horizontal .metric{position:sticky;left:0;text-align:left;background:#fff;z-index:2;min-width:245px;color:#1f2b50;font-weight:400}
.horizontal .unit{position:sticky;left:245px;text-align:center;background:#fff;z-index:2;min-width:82px;color:#65738c;font-weight:400}
.horizontal thead .metric,.horizontal thead .unit{z-index:4;background:var(--navy);color:#f8fafc;font-weight:700}
.horizontal tr:nth-child(even) .metric,.horizontal tr:nth-child(even) .unit,.horizontal tbody tr:nth-child(even) td{background:#fbfcff}
.horizontal tr.strong-row th,.horizontal tr.strong-row td{background:var(--green);border-bottom:1px solid var(--greenline);font-weight:700}
.horizontal tr.spacer th,.horizontal tr.spacer td{height:34px;background:#fff;border-bottom:2px solid #cfd6e6;font-weight:700;text-decoration:underline}
.horizontal tr.blank-row th,.horizontal tr.blank-row td{height:30px;background:#fff!important;border-bottom:0!important;font-weight:400}
.horizontal tr.spacer .spacer-label{font-weight:700;color:var(--navy);background:#fff;text-decoration:underline}
.horizontal tr.italic-row th,.horizontal tr.italic-row td{font-style:italic}
.section-title{display:flex;justify-content:space-between;align-items:flex-end;gap:16px}
h2{margin:0 0 14px;font-size:21px;color:var(--navy)}
.small{font-size:12px;color:var(--muted)}
@media(max-width:1100px){.model-layout{grid-template-columns:1fr}.side-panel{position:static}.kpis{grid-template-columns:repeat(2,1fr)}}
"""


def render_page(error=None):
    defaults = {
        "deal_size": f"{parse_user_number(request.values.get('deal_size', '100'), 100):,.2f}",
        "spread": request.values.get("spread", "3.00"),
        "maturity_months": request.values.get("maturity_months", "36"),
        "disbursement_month": request.values.get("disbursement_month", pd.Timestamp.today().strftime("%Y-%m")),
        "amortization": request.values.get("amortization", "bullet"),
        "upfront_fee": request.values.get("upfront_fee", "0.00"),
        "extension_fee": request.values.get("extension_fee", "0.00"),
        "extension_fee_month": request.values.get("extension_fee_month", ""),
        "extension_fee_treatment": request.values.get("extension_fee_treatment", "pik"),
        "extension_fee_2": request.values.get("extension_fee_2", "0.00"),
        "extension_fee_month_2": request.values.get("extension_fee_month_2", ""),
        "extension_fee_treatment_2": request.values.get("extension_fee_treatment_2", "pik"),
        "interest_frequency_months": request.values.get("interest_frequency_months", "3"),
        "curve_date": request.values.get("curve_date", ""),
    }
    summary, df = None, pd.DataFrame()
    if request.args or request.method == "POST":
        try:
            summary, df = simulate(request.values)
        except Exception as e:
            error = str(e)

    kpis = ""
    if summary:
        kpis = f"""<div class="small">Curve date used: {summary['curve_date']}</div><div class="kpis"><div class="kpi"><div class="t">IRR BRL</div><div class="v">{fmt_pct(summary['irr_brl'])}</div></div><div class="kpi"><div class="t">IRR USD</div><div class="v">{fmt_pct(summary['irr_usd'])}</div></div></div>"""
    else:
        kpis = '<div class="hint">Fill the inputs and click Simulate.</div>'

    err = f"<div class='error'>{error}</div>" if error else ""
    table = render_horizontal_table(df)

    query = request.query_string.decode("utf-8")
    download_btn = f"<a href='/download.xlsx?{query}' style='display:inline-block;margin-top:8px;text-decoration:none;'><button type='button'>Download Excel</button></a>" if summary else ""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Return Viewer</title><style>{CSS}</style></head><body><div class="top"><h1>Return Viewer</h1><p>Indicative cash-flow simulation for private credit deals with PRE/CDI and FX curves.</p></div><div class="wrap">{err}<div class="model-layout"><aside class="side-panel"><form class="side-card" method="get"><h2>Deal Terms</h2><div class="terms-grid"><div class="term-row"><label>Curve Date</label><input name="curve_date" type="date" value="{defaults['curve_date']}"></div><div class="term-row"><label>Deal size BRL mm</label><input name="deal_size" type="number" step="20" value="{parse_user_number(defaults['deal_size'], 100):.2f}"></div><div class="term-row"><label>CDI+ spread, % p.a.</label><input name="spread" type="number" step="0.25" value="{defaults['spread']}"></div><div class="term-row"><label>Disbursement month</label><input name="disbursement_month" type="month" value="{defaults['disbursement_month']}"></div><div class="term-row"><label>Maturity, months</label><input name="maturity_months" type="number" step="3" value="{defaults['maturity_months']}"></div><div class="term-row"><label>Amortization</label><select name="amortization"><option value="bullet" {'selected' if defaults['amortization']=='bullet' else ''}>Bullet</option><option value="linear" {'selected' if defaults['amortization']=='linear' else ''}>Linear quarterly</option></select></div><div class="term-row"><label>OID, % of principal</label><input name="upfront_fee" type="number" step="0.25" value="{defaults['upfront_fee']}"></div><div></div><div class="fee-card"><div class="fee-card-grid"><div class="term-row"><label>Extension fee #1</label><input name="extension_fee" type="number" step="0.25" value="{defaults['extension_fee']}"></div><div class="term-row"><label>Month</label><input name="extension_fee_month" type="number" step="3" value="{defaults['extension_fee_month']}"></div><div class="inline-check"><label><input type="radio" name="extension_fee_treatment" value="pik" {'checked' if defaults['extension_fee_treatment']=='pik' else ''}> PIK</label><label><input type="radio" name="extension_fee_treatment" value="cash" {'checked' if defaults['extension_fee_treatment']=='cash' else ''}> Cash</label></div></div></div><div class="fee-card"><div class="fee-card-grid"><div class="term-row"><label>Extension fee #2</label><input name="extension_fee_2" type="number" step="0.25" value="{defaults['extension_fee_2']}"></div><div class="term-row"><label>Month</label><input name="extension_fee_month_2" type="number" step="3" value="{defaults['extension_fee_month_2']}"></div><div class="inline-check"><label><input type="radio" name="extension_fee_treatment_2" value="pik" {'checked' if defaults['extension_fee_treatment_2']=='pik' else ''}> PIK</label><label><input type="radio" name="extension_fee_treatment_2" value="cash" {'checked' if defaults['extension_fee_treatment_2']=='cash' else ''}> Cash</label></div></div></div><div class="term-row full"><label>Interest payment frequency, months</label><select name="interest_frequency_months"><option value="3" {'selected' if defaults['interest_frequency_months']=='3' else ''}>Quarterly</option><option value="6" {'selected' if defaults['interest_frequency_months']=='6' else ''}>Semiannual</option><option value="12" {'selected' if defaults['interest_frequency_months']=='12' else ''}>Annual</option><option value="999" {'selected' if defaults['interest_frequency_months']=='999' else ''}>Bullet / accrue until maturity</option></select></div><div class="term-row full"><button type="submit">Simulate</button>{download_btn}</div></div><div class="hint">Schedule is every 3 months from the disbursement month. Business days use holidays.xlsx. FX and PRE/CDI come from the selected Curve Date and are interpolated to each table date. Extension fee can be PIK into debt or paid in cash.</div></form></aside><main class="table-card"><div class="section-title"><h2>Quarterly deal cash flow</h2><div class="small">All BRL amounts are shown in BRL mm. USD figures remain in full amount.</div></div>{table}<div class="bottom-summary"><h2>Summary</h2>{kpis}</div></main></div></div>
</body></html>"""
    return html

@app.route("/", methods=["GET", "POST"])
def index():
    return Response(render_page(), mimetype="text/html")


@app.route("/healthz")
def healthz():
    return "ok"


@app.route("/download.xlsx")
def download_xlsx():
    summary, df = simulate(request.args)
    out = io.BytesIO()
    engine = "xlsxwriter"
    try:
        import xlsxwriter  # noqa: F401
    except Exception:
        engine = "openpyxl"
    with pd.ExcelWriter(out, engine=engine) as writer:
        # Summary sheet
        summary_df = pd.DataFrame(
            [
                ["Curve date", summary["curve_date"]],
                ["IRR BRL", summary["irr_brl"]],
                ["IRR USD", summary["irr_usd"]],
                ["PRE source", summary["pre_name"]],
                ["FX source", summary["fx_name"]],
            ],
            columns=["Metric", "Value"],
        )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        ws_sum = writer.sheets["Summary"]
        wb = writer.book
        if engine == "xlsxwriter":
            header_fmt = wb.add_format({"bold": True, "bg_color": "#100058", "font_color": "white", "border": 1})
            note_fmt = wb.add_format({"font_color": "#4b5a7a", "italic": True, "font_size": 10})
            pct_fmt = wb.add_format({"num_format": "0.00%"})
            link_fmt = wb.add_format({"font_color": "blue", "underline": 1})
            for col, name in enumerate(summary_df.columns):
                ws_sum.write(0, col, name, header_fmt)
        else:
            pct_fmt = None
        ws_sum.set_column("A:A", 22)
        ws_sum.set_column("B:B", 42)
        if engine == "xlsxwriter":
            ws_sum.write_number(2, 1, summary["irr_brl"] if summary["irr_brl"] is not None else 0, pct_fmt)
            ws_sum.write_number(3, 1, summary["irr_usd"] if summary["irr_usd"] is not None else 0, pct_fmt)
            ws_sum.write_url(4, 1, f"external:gs://{GCS_BUCKET_NAME}/{summary['pre_name']}", link_fmt, summary["pre_name"])
            ws_sum.write_url(5, 1, f"external:gs://{GCS_BUCKET_NAME}/{summary['fx_name']}", link_fmt, summary["fx_name"])
            ws_sum.write(7, 0, "Generated from Return Viewer", note_fmt)
        else:
            ws_sum.write(8, 1, f"gs://{GCS_BUCKET_NAME}/{summary['pre_name']}")
            ws_sum.write(9, 1, f"gs://{GCS_BUCKET_NAME}/{summary['fx_name']}")

        # Cashflow sheet - horizontal layout mirroring the site table
        if engine != "xlsxwriter":
            df.to_excel(writer, sheet_name="Cashflow", index=False)
        else:
            ws = wb.add_worksheet("Cashflow")
            writer.sheets["Cashflow"] = ws
            periods = list(df["period_label"])
            header = ["Metric", "Unit"] + periods
            rows_spec = [
            ("Rates & Support", "-", "spacer", None),
            ("Payment date", "Date", "date", df["payment_date"]),
            ("Quarter", "Text", "text", df["quarter"]),
            ("Period business days", "BD", "int", df["period_business_days"]),
            ("PRE rate at period start", "% p.a.", "pct", df["pre_rate_start"]),
            ("PRE rate at period end", "% p.a.", "pct", df["pre_rate_end"]),
            ("Implied CDI for period", "% p.a.", "pct", df["cdi_period_rate"]),
            ("Spread over CDI", "% p.a.", "pct", df["spread"]),
            ("Total Rate", "% p.a.", "pct", df["deal_rate"]),
            ("", "", "blank", None),
            ("Debt balance bridge", "-", "spacer", None),
            ("Debt BoP", "BRL mm", "brl", df["debt_bop_brl"]),
            ("(+) Issuance", "BRL mm", "brl", df["issuance_brl"]),
            ("(+) Interest Accrual", "BRL mm", "brl", df["interest_accrual_brl"]),
            ("(+) Extension Fee #1", "BRL mm", "brl", df["extension_fee_accrual_brl"]),
            ("(+) Extension Fee #2", "BRL mm", "brl", df["extension_fee_accrual_brl_2"]),
            ("(-) Cash Interest", "BRL mm", "brl", -df["cash_interest_brl"]),
            ("(-) Debt Amortization", "BRL mm", "brl", -df["principal_brl"]),
            ("(=) Debt EoP", "BRL mm", "brl", df["debt_eop_brl"]),
            ("", "", "blank", None),
            ("Cash flow / FX", "-", "spacer", None),
            ("(-) Disbursement", "BRL mm", "brl", df["disbursement_brl"]),
            ("(+) OID", "BRL mm", "brl", df["upfront_fee_brl"]),
            ("(+) Cash Interest", "BRL mm", "brl", df["cash_interest_brl"]),
            ("(+) Extension Fee #1", "BRL mm", "brl", df["cash_extension_fee_brl"]),
            ("(+) Extension Fee #2", "BRL mm", "brl", df["cash_extension_fee_brl_2"]),
            ("(+) Debt Repayment", "BRL mm", "brl", df["principal_brl"]),
            ("(=) Total Debt Cash Flow", "BRL mm", "brl", df["total_cf_brl"]),
            ("BRL/USD forward FX", "BRL/USD", "fx", df["fx_forward"]),
            ("Total Debt Cash Flow", "USD mm", "usdmm", df["total_cf_usd"]),
        ]

            fmt_header = wb.add_format({"bold": True, "bg_color": "#100058", "font_color": "white", "border": 1, "align": "center"})
            fmt_metric = wb.add_format({"border": 1})
            fmt_spacer = wb.add_format({"bold": True, "underline": 1, "border": 1})
            fmt_num = wb.add_format({"border": 1, "align": "center"})
            fmt_int = wb.add_format({"border": 1, "align": "center"})
            fmt_pct = wb.add_format({"border": 1, "align": "center"})
            fmt_fx = wb.add_format({"border": 1, "align": "center"})
            fmt_date = wb.add_format({"border": 1, "align": "center"})
            fmt_text = wb.add_format({"border": 1})
            fmt_strong = wb.add_format({"border": 1, "bold": True, "align": "center"})

            for c, val in enumerate(header):
                ws.write(0, c, val, fmt_header)

            r = 1
            for label, unit, kind, series in rows_spec:
                if kind == "blank":
                    r += 1
                    continue
                row_fmt = fmt_spacer if kind == "spacer" else fmt_metric
                ws.write(r, 0, label, row_fmt)
                ws.write(r, 1, unit, row_fmt)
                if kind == "spacer":
                    for c in range(2, len(header)):
                        ws.write(r, c, "", row_fmt)
                    r += 1
                    continue

                vals = list(series.values) if series is not None else []
                for i, v in enumerate(vals, start=2):
                    def pct_txt(x):
                        if is_zero(x):
                            return "-"
                        return f"{float(x)*100:.1f}%".replace(".", ",")
                    def num_txt(x, decimals=0):
                        if is_zero(x):
                            return "-"
                        val = float(x)
                        txt = f"{abs(val):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
                        return f"({txt})" if val < 0 else txt
                    if kind == "date":
                        ws.write(r, i, pd.Timestamp(v).strftime("%d-%b-%y"), fmt_date)
                    elif kind == "int":
                        ws.write(r, i, num_txt(v, 0), fmt_int)
                    elif kind == "pct":
                        ws.write(r, i, pct_txt(v), fmt_pct)
                    elif kind == "fx":
                        ws.write(r, i, num_txt(v, 0), fmt_fx)
                    elif kind == "brl":
                        mm = float(v) / 1_000_000.0 if not is_zero(v) else 0.0
                        ws.write(r, i, num_txt(mm, 0), fmt_num if "(=)" not in label else fmt_strong)
                    elif kind == "usdmm":
                        mm = float(v) / 1_000_000.0 if not is_zero(v) else 0.0
                        ws.write(r, i, num_txt(mm, 0), fmt_num if "Total Debt Cash Flow" not in label else fmt_strong)
                    else:
                        ws.write(r, i, str(v), fmt_text)
                r += 1

            ws.set_column(0, 0, 30)
            ws.set_column(1, 1, 10)
            ws.set_column(2, len(header), 12)
            ws.freeze_panes(1, 2)
    out.seek(0)
    return Response(
        out.getvalue(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=return_viewer.xlsx"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=True)
