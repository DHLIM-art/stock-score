# -*- coding: utf-8 -*-
"""
stock_score.py  —  매수 평가 점수 엔진 (Windows PC 버전)
==========================================================
- CAN SLIM + 퀀트 팩터 + 수학 지표(허스트/칼만/Z-Score) + 심리
- 점수 공식은 웹(React) 버전과 동일합니다.  METRICS 배열만 고치면 기준이 바뀝니다.
- 데이터는 FinanceDataReader / pykrx / yfinance 에서 자동 수집하며,
  수집 실패 시 기본값을 쓰고 app.py 사이드바에서 수동 보정할 수 있습니다.

단독 실행:  python stock_score.py 000660
"""
from __future__ import annotations
import math
import sys

ENGINE_VERSION = "v5 (2026-06-02, ROE=TTM)"  # 배포 확인용 — 화면에 표시됨

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 유틸
# ----------------------------------------------------------------------------
def clamp(x, lo=0.0, hi=100.0):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo

def r1(x):
    try:
        return round(float(x), 1)
    except (TypeError, ValueError):
        return 0.0

def safe(x, default=0.0):
    try:
        v = float(x)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


# ----------------------------------------------------------------------------
# 기본 입력값 (수집 실패 시 사용 / SK하이닉스 예시값)
# ----------------------------------------------------------------------------
DEFAULTS = dict(
    ticker="000660", name="SK하이닉스", sector="HBM·낸드",
    price=1819000.0, changePct=-7.66, high52w=1995000.0, low52w=175400.0,
    rsi=71.8, volumeRatioVsAvg=1.5, breakoutSignal=False, pivotBreak=True,
    roeAnnual=61.0, roeQuarter=None, roeThreshold=17.0, epsGrowthQoQ=396.6, epsAccelQuarters=2,
    rsRating=99, return12m=930.8, mddPct=-8.0,
    zScoreMeanRev=2.1, zScoreStat=1.2, hurst=0.76, adx=50.0, mfi=70.0,
    shortRatioPct=0.0, dcfFairValue=3179188.0, targetPrice=2003200.0,
    factorAlpha=12.8, alignedTimeframes=3, obvTrend="up", vwapPosition="above",
    marketRegime="STRONG_BULL", kalmanSignal="neutral",
    upVolPct=60.0, closeStrength=57.0, gapDir=1, volMultiplier=1.18, atr=95000.0,
    # --- 가치투자 지표 ---
    per=12.0, pbr=2.1, dividendYield=1.2, debtRatio=35.0, pegRatio=0.4,
    # --- 매크로 (시장 공통) ---
    usdkrwTrend="down", rateTrend="flat", vix=16.0,
)


# ----------------------------------------------------------------------------
# 기술적 지표 계산 (pandas/numpy 만 사용)
# ----------------------------------------------------------------------------
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def adx(high, low, close, period=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/period, adjust=False).mean()
    pdi = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr_
    mdi = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def atr(high, low, close, period=14):
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(), 0.0).rolling(period).sum()
    neg = mf.where(tp < tp.shift(), 0.0).rolling(period).sum()
    mr = pos / neg.replace(0, np.nan)
    return 100 - 100 / (1 + mr)

def obv(close, volume):
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()

def hurst_exponent(ts, max_lag=20):
    ts = np.asarray(ts, dtype=float)
    ts = ts[~np.isnan(ts)]
    if len(ts) < max_lag + 2:
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = ts[lag:] - ts[:-lag]
        sd = np.std(diff)
        tau.append(sd if sd > 0 else 1e-9)
    try:
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return float(poly[0])
    except Exception:
        return 0.5

def kalman_signal(close, window=40):
    """초간단 1D 칼만 필터로 추세 방향 추정 (bullish/neutral/bearish)."""
    z = close.values[-window:]
    if len(z) < 5:
        return "neutral"
    x, p, q, r = z[0], 1.0, 1e-4, 1.0
    est = []
    for m in z:
        p += q
        k = p / (p + r)
        x = x + k * (m - x)
        p = (1 - k) * p
        est.append(x)
    slope = (est[-1] - est[max(0, len(est) - 6)]) / (abs(est[-1]) + 1e-9)
    if slope > 0.01:
        return "bullish"
    if slope < -0.01:
        return "bearish"
    return "neutral"

def max_drawdown(close, window=252):
    c = close.tail(window)
    roll_max = c.cummax()
    dd = c / roll_max - 1
    return float(dd.min() * 100)


def compute_indicators(df: pd.DataFrame) -> dict:
    """OHLCV DataFrame(컬럼: Open/High/Low/Close/Volume) → 지표 dict + 차트용 df."""
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    df["EMA20"], df["EMA50"], df["EMA200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    df["RSI"] = rsi(c)
    df["OBV"] = obv(c, v)
    last, prev = c.iloc[-1], c.iloc[-2] if len(c) > 1 else c.iloc[-1]

    win = min(252, len(c))
    high52, low52 = float(c.tail(win).max()), float(c.tail(win).min())
    z20 = (c - c.rolling(20).mean()) / c.rolling(20).std()
    z60 = (c - c.rolling(60).mean()) / c.rolling(60).std()
    vwap = (c * v).rolling(20).sum() / v.rolling(20).sum()
    obv_s = df["OBV"]
    up_days = (c.diff() > 0)
    up_vol = (v[up_days].tail(20).sum()) / (v.tail(20).sum() + 1e-9) * 100
    close_strength = ((c - l) / (h - l).replace(0, np.nan)).tail(20).mean() * 100
    ret12 = (last / c.iloc[-win] - 1) * 100 if len(c) >= win else 0.0

    ind = dict(
        price=float(last),
        changePct=float((last / prev - 1) * 100),
        high52w=high52, low52w=low52,
        rsi=safe(df["RSI"].iloc[-1], 50),
        adx=safe(adx(h, l, c).iloc[-1], 20),
        mfi=safe(mfi(h, l, c, v).iloc[-1], 50),
        atr=safe(atr(h, l, c).iloc[-1], last * 0.03),
        hurst=r1(hurst_exponent(c.values)),
        zScoreMeanRev=r1(safe(z20.iloc[-1], 0)),
        zScoreStat=r1(safe(z60.iloc[-1], 0)),
        mddPct=r1(max_drawdown(c)),
        return12m=r1(ret12),
        volumeRatioVsAvg=r1(safe(v.iloc[-1] / v.tail(20).mean(), 1)),
        obvTrend="up" if obv_s.iloc[-1] > obv_s.iloc[-min(20, len(obv_s))] else "down",
        vwapPosition="above" if last >= safe(vwap.iloc[-1], last) else "below",
        kalmanSignal=kalman_signal(c),
        upVolPct=r1(safe(up_vol, 50)),
        closeStrength=r1(safe(close_strength, 50)),
        gapDir=1 if df["Open"].iloc[-1] > prev else (-1 if df["Open"].iloc[-1] < prev else 0),
        alignedTimeframes=int((last > df["EMA20"].iloc[-1]) + (df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]) + (df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1])),
        pivotBreak=bool(last >= high52 * 0.93),
        breakoutSignal=bool(last >= high52 * 0.98 and safe(v.iloc[-1] / v.tail(20).mean(), 1) > 1.4),
    )
    return {"ind": ind, "df": df}


# ----------------------------------------------------------------------------
# 데이터 수집 (한국/해외 자동 대응)
# ----------------------------------------------------------------------------
def _norm_krx(t):
    return t.replace(".KS", "").replace(".KQ", "").strip()

def fetch_ohlcv(ticker: str, period_days: int = 400):
    """FinanceDataReader → yfinance 순으로 OHLCV 수집."""
    warnings = []
    start = (pd.Timestamp.today() - pd.Timedelta(days=int(period_days * 1.6))).strftime("%Y-%m-%d")
    # 1) FinanceDataReader
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(_norm_krx(ticker), start)
        if df is not None and len(df) > 30:
            df = df.rename(columns=str.title)
            return df[["Open", "High", "Low", "Close", "Volume"]].dropna(), warnings
    except Exception as e:
        warnings.append(f"FinanceDataReader 실패: {e}")
    # 2) yfinance (코스피 .KS / 코스닥 .KQ 양쪽 시도)
    try:
        import yfinance as yf
        for yt in _yf_candidates(ticker):
            df = yf.download(yt, start=start, progress=False, auto_adjust=False)
            if df is not None and len(df) > 30:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                return df[["Open", "High", "Low", "Close", "Volume"]].dropna(), warnings
    except Exception as e:
        warnings.append(f"yfinance 실패: {e}")
    return None, warnings + ["가격 데이터를 가져오지 못했습니다. 인터넷 연결을 확인하세요."]

def fetch_fundamentals(ticker: str):
    """pykrx 로 PER/PBR/EPS/BPS → ROE 근사. 실패 시 빈 dict."""
    out, warnings = {}, []
    try:
        from pykrx import stock
        code = _norm_krx(ticker)
        today = pd.Timestamp.today().strftime("%Y%m%d")
        ago = (pd.Timestamp.today() - pd.Timedelta(days=10)).strftime("%Y%m%d")
        fnd = stock.get_market_fundamental_by_date(ago, today, code)
        if fnd is not None and len(fnd):
            row = fnd.iloc[-1]
            # ROE는 pykrx EPS/BPS로 추정하지 않음(시점값이라 부정확).
            # ROE는 DART(TTM)→네이버→yfinance 우선순위에서만 계산.
        try:
            nm = stock.get_market_ticker_name(code)
            if nm:
                out["name"] = nm
        except Exception:
            pass
    except Exception as e:
        warnings.append(f"pykrx 펀더멘털 생략: {e}")
    return out, warnings

def market_regime():
    """KOSPI(KS11) 의 EMA200 대비 위치로 시장 국면 판정."""
    try:
        import FinanceDataReader as fdr
        idx = fdr.DataReader("KS11", (pd.Timestamp.today() - pd.Timedelta(days=400)).strftime("%Y-%m-%d"))
        c = idx["Close"]
        e200 = ema(c, 200).iloc[-1]
        a = safe(adx(idx["High"], idx["Low"], c).iloc[-1], 20)
        if c.iloc[-1] > e200 and a >= 25:
            return "STRONG_BULL"
        if c.iloc[-1] > e200:
            return "BULL"
        return "BEAR" if c.iloc[-1] < e200 * 0.97 else "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def fetch_macro():
    """시장 환경(매크로) 지표: USD/KRW 추세, VIX. (종목 공통)"""
    out, warns = {}, []
    try:
        import FinanceDataReader as fdr
        start = (pd.Timestamp.today() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        # 환율: 최근 20일 추세 (원화 약세 = 위험회피)
        try:
            fx = fdr.DataReader("USD/KRW", start)["Close"].dropna()
            if len(fx) > 21:
                out["usdkrwTrend"] = "up" if fx.iloc[-1] > fx.iloc[-21] * 1.005 else \
                                     ("down" if fx.iloc[-1] < fx.iloc[-21] * 0.995 else "flat")
        except Exception:
            pass
        # 변동성: VIX
        try:
            vix = fdr.DataReader("VIX", start)["Close"].dropna()
            if len(vix):
                out["vix"] = r1(float(vix.iloc[-1]))
        except Exception:
            pass
    except Exception as e:
        warns.append(f"매크로 수집 생략: {e}")
    return out, warns


_LISTING = {}  # KRX 종목명 목록 캐시 (프로세스 1회만 다운로드)
_YF_SUFFIX = {}  # 종목코드 → ".KS"/".KQ" 캐시

def _yf_symbol(ticker: str):
    """yfinance 심볼 결정: 코스피=.KS, 코스닥=.KQ (KRX 목록의 Market으로 판별)."""
    code = _norm_krx(ticker)
    if "." in ticker or not code.isdigit():
        return ticker                       # 해외/이미 접미사 있음
    if code in _YF_SUFFIX:
        return code + _YF_SUFFIX[code]
    suffix = ".KS"
    try:
        import FinanceDataReader as fdr
        if "KRX" not in _LISTING:
            _LISTING["KRX"] = fdr.StockListing("KRX")
        lst = _LISTING["KRX"]
        cc = "Code" if "Code" in lst.columns else ("Symbol" if "Symbol" in lst.columns else None)
        mc = "Market" if "Market" in lst.columns else None
        if cc and mc:
            row = lst[lst[cc].astype(str).str.zfill(6) == code]
            if len(row):
                mkt = str(row.iloc[0][mc]).upper()
                suffix = ".KQ" if "KOSDAQ" in mkt else ".KS"
    except Exception:
        pass
    _YF_SUFFIX[code] = suffix
    return code + suffix

def _yf_candidates(ticker: str):
    """yfinance에 시도할 심볼 목록(코스피/코스닥 양쪽 폴백)."""
    code = _norm_krx(ticker)
    if "." in ticker or not code.isdigit():
        return [ticker]
    primary = _yf_symbol(ticker)
    other = code + (".KQ" if primary.endswith(".KS") else ".KS")
    return [primary, other]

def fetch_name(ticker: str):
    """종목명을 여러 소스에서 시도: pykrx → FDR 종목목록 → yfinance."""
    code = _norm_krx(ticker)
    # 1) pykrx (로컬에선 빠름, 클라우드에선 막힐 수 있음)
    try:
        from pykrx import stock
        nm = stock.get_market_ticker_name(code)
        if nm and str(nm).strip():
            return str(nm).strip()
    except Exception:
        pass
    # 2) FinanceDataReader 종목 목록 (클라우드에서도 비교적 안정적)
    try:
        import FinanceDataReader as fdr
        if "KRX" not in _LISTING:
            _LISTING["KRX"] = fdr.StockListing("KRX")
        lst = _LISTING["KRX"]
        cc = "Code" if "Code" in lst.columns else ("Symbol" if "Symbol" in lst.columns else None)
        nc = "Name" if "Name" in lst.columns else None
        if cc and nc:
            hit = lst[lst[cc].astype(str).str.zfill(6) == code]
            if len(hit):
                return str(hit.iloc[0][nc]).strip()
    except Exception:
        pass
    # 3) yfinance (코스피 .KS / 코스닥 .KQ 양쪽 시도)
    try:
        import yfinance as yf
        for yt in _yf_candidates(ticker):
            info = getattr(yf.Ticker(yt), "info", None) or {}
            nm = info.get("shortName") or info.get("longName")
            if nm:
                return str(nm).strip()
    except Exception:
        pass
    return None


_UNIVERSE = None
def build_universe(dart_key: str | None = None):
    """검색용 종목 목록: 코스피·코스닥·미국(NASDAQ·NYSE·AMEX) 의 (종목명, 심볼, 시장).
    한국은 FDR KRX → 막히면 DART corpCode 로 폴백(클라우드에서 안정적)."""
    global _UNIVERSE
    if _UNIVERSE is not None:
        return _UNIVERSE
    rows = []
    try:
        import FinanceDataReader as fdr

        # --- 한국 1순위: FDR KRX (Market 으로 코스피/코스닥 분리) ---
        kr_rows = []
        try:
            krx = fdr.StockListing("KRX")
            cc = "Code" if "Code" in krx.columns else ("Symbol" if "Symbol" in krx.columns else None)
            nc = "Name" if "Name" in krx.columns else None
            mc = "Market" if "Market" in krx.columns else None
            if cc and nc:
                codes = krx[cc].astype(str)
                names = krx[nc].astype(str)
                mkts = krx[mc].astype(str) if mc else ["" for _ in range(len(krx))]
                for sym, nm, mk in zip(codes, names, mkts):
                    sym, nm, mk = sym.strip().zfill(6), nm.strip(), str(mk).upper()
                    if not nm or nm.lower() == "nan" or not sym.isdigit():
                        continue
                    if "KOSDAQ" in mk or "코스닥" in mk:
                        label = "코스닥"
                    elif "KONEX" in mk or "코넥스" in mk:
                        continue
                    else:
                        label = "코스피"
                    kr_rows.append((nm, sym, label))
        except Exception:
            pass

        # --- 한국 2순위: DART corpCode (KRX가 비었을 때) ---
        if not kr_rows and dart_key:
            for nm, sym in _dart_stock_list(dart_key):
                kr_rows.append((nm, sym, "한국"))

        rows += kr_rows

        # --- 미국: 나스닥·NYSE·AMEX (S&P/다우/러셀 종목 포함) ---
        for market, label in [("NASDAQ", "나스닥"), ("NYSE", "NYSE"), ("AMEX", "AMEX")]:
            try:
                lst = fdr.StockListing(market)
                cc = "Symbol" if "Symbol" in lst.columns else ("Code" if "Code" in lst.columns else None)
                nc = "Name" if "Name" in lst.columns else None
                if not cc or not nc:
                    continue
                for sym, nm in zip(lst[cc].astype(str), lst[nc].astype(str)):
                    sym, nm = sym.strip(), nm.strip()
                    if not sym or not nm or nm.lower() == "nan":
                        continue
                    rows.append((nm, sym, label))
            except Exception:
                continue
    except Exception:
        pass

    seen, uni = set(), []
    for nm, sym, mk in rows:
        if sym in seen:
            continue
        seen.add(sym)
        uni.append({"name": nm, "symbol": sym, "market": mk})
    _UNIVERSE = uni
    return uni


def resolve_ticker(query: str):
    """입력을 종목코드로 변환. 6자리코드/해외티커는 그대로, 한글 종목명은 '정확히 일치'할 때만 코드로.
    반환: (코드 또는 None, 경고 또는 None)."""
    q = (query or "").strip()
    if not q:
        return q, None
    code = _norm_krx(q)
    if code.isdigit() and len(code) == 6:          # 이미 6자리 코드
        return code, None
    # KRX 종목목록에서 종목명 정확 일치 검색
    try:
        import FinanceDataReader as fdr
        if "KRX" not in _LISTING:
            _LISTING["KRX"] = fdr.StockListing("KRX")
        lst = _LISTING["KRX"]
        nc = "Name" if "Name" in lst.columns else None
        cc = "Code" if "Code" in lst.columns else ("Symbol" if "Symbol" in lst.columns else None)
        if nc and cc:
            hit = lst[lst[nc].astype(str).str.strip() == q]   # 대소문자·공백 그대로 정확 일치
            if len(hit):
                return str(hit.iloc[0][cc]).zfill(6), None
    except Exception:
        pass
    if any("가" <= ch <= "힣" for ch in q):          # 한글인데 못 찾음
        return None, f"'{q}'와 정확히 일치하는 종목명을 찾지 못했어요. 정확한 종목명 또는 6자리 코드를 입력하세요."
    return q, None                                    # 영문 → 해외 티커로 간주, 그대로 사용


def _to_num(x):
    try:
        s = str(x).replace(",", "").replace("%", "").strip()
        if s in ("", "-", "nan", "N/A", "None"):
            return None
        return float(s)
    except Exception:
        return None

def _parse_naver_fin(fin):
    """네이버 '기업실적분석' 표 → dict(roe, epsGrowthQoQ, per, pbr, dividendYield, debtRatio).
    가장 최근 '분기' 실적을 우선 사용(분기 칸이 비면 연간으로 폴백).
    EPS 성장은 분기 YoY(4분기 전 대비) 우선, 없으면 분기 QoQ, 그것도 없으면 연간 YoY."""
    out = {}
    try:
        cols = [(" ".join(map(str, c)) if isinstance(c, tuple) else str(c)) for c in fin.columns]
        fin = fin.copy()
        fin.columns = cols
        label_col = cols[0]
        labels = fin[label_col].astype(str)
        data_cols = cols[1:]
        q_cols = [c for c in data_cols if "분기" in c]      # 분기 실적 칼럼
        a_cols = [c for c in data_cols if "연간" in c]      # 연간 실적 칼럼
        if not q_cols and not a_cols:                        # MultiIndex 아님 → 전체를 분기 후보로
            q_cols = data_cols

        def row(key):
            idx = labels[labels.str.contains(key, na=False)].index
            return fin.loc[idx[0]] if len(idx) else None

        def actuals(rr, group):
            return [v for v in (_to_num(rr[c]) for c in group if "(E)" not in c) if v is not None]

        def latest(key, positive=False):
            """분기 실적 우선 → 연간 실적 → 아무 값 순으로 최신값. positive=True면 0 이하 무시."""
            rr = row(key)
            if rr is None:
                return None
            def vals(group):
                vs = actuals(rr, group)
                return [v for v in vs if v > 0] if positive else vs
            for group in (q_cols, a_cols, data_cols):
                av = vals(group)
                if av:
                    return av[-1]
            anyv = [v for v in (_to_num(rr[c]) for c in data_cols) if v is not None]
            if positive:
                anyv = [v for v in anyv if v > 0]
            return anyv[-1] if anyv else None

        roe = latest("ROE")
        if roe is not None:
            out["roeAnnual"] = r1(roe)

        # EPS 성장: 분기 YoY > 분기 QoQ > 연간 YoY
        eps_row = row("EPS")
        if eps_row is not None:
            qv, av = actuals(eps_row, q_cols), actuals(eps_row, a_cols)
            pair = None
            if len(qv) >= 5:
                pair = (qv[-5], qv[-1])        # 분기 YoY (4분기 전 대비)
            elif len(qv) >= 2:
                pair = (qv[-2], qv[-1])        # 분기 QoQ
            elif len(av) >= 2:
                pair = (av[-2], av[-1])        # 연간 YoY
            if pair and pair[0] not in (0, None):
                out["epsGrowthQoQ"] = r1((pair[1] - pair[0]) / abs(pair[0]) * 100)
                out["epsAccelQuarters"] = 2 if out["epsGrowthQoQ"] > 0 else 0

        per = latest("PER", positive=True)
        if per is not None:
            out["per"] = r1(per)
        pbr = latest("PBR", positive=True)
        if pbr is not None:
            out["pbr"] = r1(pbr)
        div = latest("시가배당")   # 시가배당률(%)
        if div is not None and div > 0:
            out["dividendYield"] = r1(div)
        debt = latest("부채비율")
        if debt is not None and debt > 0:
            out["debtRatio"] = r1(debt)
    except Exception:
        pass
    return out

def fetch_naver(ticker: str):
    """네이버 금융에서 ROE·EPS성장·증권사 목표주가 수집 (best-effort, 실패 시 빈 dict)."""
    out, warns = {}, []
    code = _norm_krx(ticker)
    if not (code.isdigit() and len(code) == 6):
        return out, warns  # 국내 6자리 종목만
    try:
        import requests
        from io import StringIO
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(f"https://finance.naver.com/item/main.naver?code={code}",
                         headers=headers, timeout=8)
        r.encoding = r.apparent_encoding or "euc-kr"
        html = r.text
        # 1) ROE / EPS 성장률 / PER / PBR / 배당 / 부채
        try:
            for t in pd.read_html(StringIO(html)):
                flat = " ".join(sum(t.astype(str).values.tolist(), []))
                if "ROE" in flat and "EPS" in flat:
                    out.update(_parse_naver_fin(t))
                    break
        except Exception:
            pass
        # 2) 증권사 목표주가
        try:
            from bs4 import BeautifulSoup
            text = BeautifulSoup(html, "lxml").get_text(" ")
            import re
            m = re.search(r"목표주가[^0-9]{0,8}([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})", text)
            if m:
                tp = _to_num(m.group(1))
                if tp and tp > 0:
                    out["targetPrice"] = tp
        except Exception:
            pass
        if not out:
            warns.append("네이버 재무/목표가 파싱 실패(페이지 구조 변경 가능). 추정값 사용.")
    except Exception as e:
        warns.append(f"네이버 접속 실패: {e}")
    return out, warns


def fetch_yf_fundamentals(ticker: str):
    """yfinance(.info)에서 ROE·PER·PBR·배당·부채·EPS성장·목표주가 보강.
    미국 기반 클라우드(예: Streamlit Cloud)에서 네이버보다 안정적일 때가 많음.
    코스피 .KS / 코스닥 .KQ 양쪽을 시도해 값이 있는 쪽을 사용."""
    out, warns = {}, []
    try:
        import yfinance as yf
        info = {}
        for yt in _yf_candidates(ticker):
            cand = getattr(yf.Ticker(yt), "info", None) or {}
            # 가격/시총이 잡히면 유효한 심볼로 간주
            if cand.get("regularMarketPrice") or cand.get("marketCap") or cand.get("trailingPE"):
                info = cand
                break
        roe = info.get("returnOnEquity")
        if roe is not None:
            out["roeAnnual"] = r1(roe * 100)
        per = info.get("trailingPE") or info.get("forwardPE")
        if (not per or per <= 0):
            # 폴백: 주가 ÷ 주당순이익(EPS)
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
            if price and eps and eps > 0:
                per = price / eps
        if per and per > 0:
            out["per"] = r1(per)
        pbr = info.get("priceToBook")
        if (not pbr or pbr <= 0):
            # 폴백: 주가 ÷ 주당순자산(BVPS)
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            bvps = info.get("bookValue")
            if price and bvps and bvps > 0:
                pbr = price / bvps
        if pbr and pbr > 0:
            out["pbr"] = r1(pbr)
        dy = info.get("dividendYield")
        if dy is not None:
            out["dividendYield"] = r1(dy * 100 if dy < 1 else dy)   # 0.012 → 1.2%
        de = info.get("debtToEquity")
        if de is not None:
            out["debtRatio"] = r1(de)
        g = info.get("earningsQuarterlyGrowth")
        if g is None:
            g = info.get("earningsGrowth")
        if g is not None:
            out["epsGrowthQoQ"] = r1(g * 100)
            out["epsAccelQuarters"] = 2 if g > 0 else 0
        tp = info.get("targetMeanPrice")
        if tp:
            out["targetPrice"] = float(tp)
    except Exception as e:
        warns.append(f"yfinance 펀더멘털 생략: {e}")
    return out, warns


def fetch_yf_quarter_roe(ticker: str):
    """yfinance 분기 재무제표에서 TTM(최근 4개 분기 합산) ROE 계산.
    ROE = 최근 4분기 순이익 합 ÷ 최근 자기자본 × 100. (분기 1개뿐이면 연환산) 실패 시 None."""
    code = _norm_krx(ticker)
    try:
        import yfinance as yf
        for yt in _yf_candidates(ticker):
            t = yf.Ticker(yt)
            qf = getattr(t, "quarterly_income_stmt", None)
            if qf is None or qf.empty:
                qf = getattr(t, "quarterly_financials", None)
            qb = getattr(t, "quarterly_balance_sheet", None)
            if qf is None or qb is None or qf.empty or qb.empty:
                continue

            ni_row = None
            for nm in ["Net Income", "NetIncome", "Net Income Common Stockholders",
                       "Net Income From Continuing Operation Net Minority Interest"]:
                if nm in qf.index:
                    s = qf.loc[nm].dropna()
                    if len(s):
                        ni_row = s
                        break
            eq = None
            for nm in ["Stockholders Equity", "Total Stockholder Equity",
                       "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                if nm in qb.index:
                    s = qb.loc[nm].dropna()
                    if len(s):
                        eq = float(s.iloc[0])      # 최근 분기 자기자본
                        break
            if ni_row is None or not eq or eq <= 0:
                continue
            vals = [float(v) for v in ni_row.values if v == v][:4]  # NaN 제거, 최근 4개
            if len(vals) < 4:
                continue                                   # 4개 미만이면 TTM 불가 → 연환산 안 함
            ttm_ni = sum(vals)                             # 최근 4개 분기 합 = 정확한 TTM
            return r1(ttm_ni / eq * 100)
    except Exception:
        pass
    return None


def fetch_yf_quarter_eps_growth(ticker: str):
    """yfinance 분기 손익계산서에서 EPS(또는 순이익) 성장률 계산.
    분기 YoY(4분기 전 대비) 우선, 없으면 분기 QoQ. 실패 시 None."""
    try:
        import yfinance as yf
        for yt in _yf_candidates(ticker):
            t = yf.Ticker(yt)
            qf = getattr(t, "quarterly_income_stmt", None)
            if qf is None or qf.empty:
                qf = getattr(t, "quarterly_financials", None)
            if qf is None or qf.empty:
                continue
            row = None
            for nm in ["Diluted EPS", "Basic EPS", "Net Income",
                       "Net Income Common Stockholders"]:
                if nm in qf.index:
                    s = qf.loc[nm].dropna()
                    if len(s) >= 2:
                        row = s
                        break
            if row is None:
                continue
            vals = [float(v) for v in row.values]   # 최신이 맨 앞
            if len(vals) >= 5 and vals[4] not in (0, None):
                return r1((vals[0] - vals[4]) / abs(vals[4]) * 100)   # 분기 YoY
            if len(vals) >= 2 and vals[1] not in (0, None):
                return r1((vals[0] - vals[1]) / abs(vals[1]) * 100)   # 분기 QoQ
    except Exception:
        pass
    return None


_DART_CORP = {}   # stock_code(6) → corp_code(8) 매핑 캐시

def _dart_corp_code(stock_code, api_key):
    """DART corpCode.xml(zip)을 1회 받아 종목코드→corp_code 매핑."""
    if not _DART_CORP:
        try:
            import requests, zipfile, io
            import xml.etree.ElementTree as ET
            r = requests.get("https://opendart.fss.or.kr/api/corpCode.xml",
                             params={"crtfc_key": api_key}, timeout=20)
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            root = ET.fromstring(zf.read(zf.namelist()[0]))
            for el in root.iter("list"):
                sc = (el.findtext("stock_code") or "").strip()
                cc = (el.findtext("corp_code") or "").strip()
                if sc and cc and sc != " ":
                    _DART_CORP[sc.zfill(6)] = cc
        except Exception:
            return None
    return _DART_CORP.get(str(stock_code).zfill(6))


_DART_STOCKS = None
def _dart_stock_list(api_key):
    """DART corpCode.xml에서 상장사(종목코드 보유) 이름+코드 목록. 클라우드에서도 안정적."""
    global _DART_STOCKS
    if _DART_STOCKS is not None:
        return _DART_STOCKS
    out = []
    try:
        import requests, zipfile, io
        import xml.etree.ElementTree as ET
        r = requests.get("https://opendart.fss.or.kr/api/corpCode.xml",
                         params={"crtfc_key": api_key}, timeout=20)
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        root = ET.fromstring(zf.read(zf.namelist()[0]))
        for el in root.iter("list"):
            sc = (el.findtext("stock_code") or "").strip()
            nm = (el.findtext("corp_name") or "").strip()
            if sc and sc != " " and len(sc) == 6 and sc.isdigit() and nm:
                out.append((nm, sc.zfill(6)))
    except Exception:
        pass
    _DART_STOCKS = out
    return out


def _dart_accounts(corp, year, reprt_code, api_key):
    """단일회사 주요계정 → 당기순이익·자본총계·부채총계 (연결 CFS 우선, 없으면 별도 OFS)."""
    try:
        import requests
        r = requests.get("https://opendart.fss.or.kr/api/fnlttSinglAcnt.json",
                         params={"crtfc_key": api_key, "corp_code": corp,
                                 "bsns_year": str(year), "reprt_code": reprt_code}, timeout=12)
        j = r.json()
        if j.get("status") != "000":
            return None
        rows = j.get("list", [])

        def grab(names):
            for fs in ("CFS", "OFS"):
                for it in rows:
                    if it.get("fs_div") == fs and it.get("account_nm", "").replace(" ", "") in names:
                        v = _to_num(it.get("thstrm_amount"))
                        if v is not None:
                            return v
            return None

        out = {}
        ni = grab({"당기순이익", "당기순이익(손실)"})
        eq = grab({"자본총계"})
        debt = grab({"부채총계"})
        if ni is not None:
            out["당기순이익"] = ni
        if eq is not None:
            out["자본총계"] = eq
        if debt is not None:
            out["부채총계"] = debt
        return out or None
    except Exception:
        return None


def _dart_shares(corp, year, reprt_code, api_key):
    """DART 주식총수현황(stockTotqySttus)에서 유통주식수(보통주) 추정."""
    try:
        import requests
        r = requests.get("https://opendart.fss.or.kr/api/stockTotqySttus.json",
                         params={"crtfc_key": api_key, "corp_code": corp,
                                 "bsns_year": str(year), "reprt_code": reprt_code}, timeout=12)
        j = r.json()
        if j.get("status") != "000":
            return None
        # '보통주' 행의 유통주식수(distb_stock_co) 우선, 없으면 발행주식총수(istc_totqy)
        for it in j.get("list", []):
            se = it.get("se", "")
            if "보통주" in se or "합계" in se:
                v = _to_num(it.get("distb_stock_co")) or _to_num(it.get("istc_totqy"))
                if v and v > 0:
                    return v
        # 못 찾으면 첫 행
        for it in j.get("list", []):
            v = _to_num(it.get("istc_totqy"))
            if v and v > 0:
                return v
    except Exception:
        pass
    return None


def fetch_dart(ticker, api_key, price=None):
    """DART 공시에서 ROE(분기 연환산)·부채비율·순이익 YoY 성장 수집."""
    out, warns = {}, []
    code = _norm_krx(ticker)
    if not api_key or not (code.isdigit() and len(code) == 6):
        return out, warns
    corp = _dart_corp_code(code, api_key)
    if not corp:
        warns.append("DART: 종목코드에 맞는 회사를 못 찾았어요(키 오류이거나 비상장).")
        return out, warns
    import datetime
    y = datetime.date.today().year
    # (사업연도, 보고서코드, 연환산배수, 라벨) — 최신 보고서부터 탐색
    candidates = [
        (y, "11014", 4 / 3, "3분기"), (y, "11012", 2.0, "반기"), (y, "11013", 4.0, "1분기"),
        (y - 1, "11011", 1.0, "연간"), (y - 1, "11014", 4 / 3, "전년3분기"),
        (y - 1, "11012", 2.0, "전년반기"),
    ]
    picked = None
    for (by, rc, ann, lab) in candidates:
        accts = _dart_accounts(corp, by, rc, api_key)
        if accts and accts.get("자본총계"):
            picked = (by, rc, ann, lab, accts)
            break
    if not picked:
        warns.append("DART: 재무 데이터를 받지 못했어요.")
        return out, warns
    by, rc, ann, lab, accts = picked
    ni, eq, debt = accts.get("당기순이익"), accts.get("자본총계"), accts.get("부채총계")
    prev_same = _dart_accounts(corp, by - 1, rc, api_key)        # 전년 동기 누적
    # TTM 순이익 = 올해 누적 + (작년 연간 − 작년 동기 누적). 연간보고서면 그대로.
    ttm_ni = None
    if ni is not None:
        if rc == "11011":
            ttm_ni = ni
        else:
            prev_annual = _dart_accounts(corp, by - 1, "11011", api_key)
            psn = prev_same.get("당기순이익") if prev_same else None
            pan = prev_annual.get("당기순이익") if prev_annual else None
            if psn is not None and pan is not None:
                ttm_ni = ni + pan - psn          # 정확한 TTM
            # 전년 자료를 못 받으면 TTM 계산 불가 → ROE 생략(다른 소스/중립에 맡김)
    if ttm_ni is not None and eq:
        out["roeAnnual"] = r1(ttm_ni / eq * 100)  # TTM ROE
    if debt is not None and eq:
        out["debtRatio"] = r1(debt / eq * 100)
    # EPS 성장(순이익 YoY): 올해 누적 vs 전년 동기 누적
    if prev_same and ni is not None:
        psn = prev_same.get("당기순이익")
        if psn not in (None, 0):
            out["epsGrowthQoQ"] = r1((ni - psn) / abs(psn) * 100)
            out["epsAccelQuarters"] = 2 if out["epsGrowthQoQ"] > 0 else 0

    # PER·PBR: 주식수를 받아 직접 계산 (주가 필요)
    if price and price > 0:
        shares = _dart_shares(corp, by, rc, api_key) or _dart_shares(corp, by, "11011", api_key)
        if shares and shares > 0:
            if ttm_ni is not None and ttm_ni > 0:
                eps_ps = ttm_ni / shares
                if eps_ps > 0:
                    out["per"] = r1(price / eps_ps)        # PER = 주가 / 주당순이익
            if eq is not None and eq > 0:
                bps = eq / shares
                if bps > 0:
                    out["pbr"] = r1(price / bps)            # PBR = 주가 / 주당순자산

    out["_dartLabel"] = f"{by} {lab}"
    return out, warns


def build_inputs(ticker: str, overrides: dict | None = None, period_days: int = 400, dart_key: str | None = None):
    """ticker(코드 또는 정확한 종목명) → 엔진 입력 dict + 차트 df + 경고 목록."""
    resolved, rwarn = resolve_ticker(ticker)
    data = dict(DEFAULTS)
    if not resolved:                       # 종목명을 못 찾음
        data["ticker"] = ticker
        data["name"] = ticker
        data["sector"] = ""
        return data, None, [rwarn] if rwarn else ["종목을 찾지 못했어요."]
    ticker = resolved
    data["ticker"] = ticker
    data["name"] = ticker      # 이름을 못 가져오면 종목코드를 그대로 표시(오해 방지)
    data["sector"] = ""
    warnings, chart_df = [], None

    df, w1 = fetch_ohlcv(ticker, period_days)
    warnings += w1
    if df is not None:
        res = compute_indicators(df)
        data.update(res["ind"])
        chart_df = res["df"]
        data["marketRegime"] = market_regime()
    fnd, w2 = fetch_fundamentals(ticker)
    data.update(fnd)
    warnings += w2

    nm = fetch_name(ticker)    # 종목명 별도 다중 소스 조회
    if nm:
        data["name"] = nm

    # 자동 수집이 어려운 항목 → 새 종목 기준 중립값으로 초기화
    # (이렇게 안 하면 예시 종목(SK하이닉스) 값이 남아 점수/익절가가 안 바뀜)
    if df is not None:
        data["dcfFairValue"] = data["price"] * 1.1   # 보수적 추정(수동 보정 가능)
        data["targetPrice"]  = data["price"] * 1.1   # 보수적 추정(수동 보정 가능)
        data["factorAlpha"]  = 0.0                    # 가치·퀄리티 알파 미수집 → 중립
        data["volMultiplier"] = 1.0                   # 변동성 배율 중립
        data["shortRatioPct"] = 0.0
        # RS 등급 근사 (12M 수익률 기반)
        data["rsRating"] = int(clamp(round(60 + data.get("return12m", 0) * 0.05), 1, 99))

        # 펀더멘털: DART(공식 공시) → 네이버 → yfinance → 중립값 순
        dt, wdt = (fetch_dart(ticker, dart_key, price=data.get("price")) if dart_key else ({}, []))
        warnings += wdt
        nv, w3 = fetch_naver(ticker)
        warnings += w3
        yf_f, w5 = fetch_yf_fundamentals(ticker)
        warnings += w5

        sources = [(dt, "DART(공시)"), (nv, "네이버(최근분기)"), (yf_f, "yfinance(TTM·연간)")]

        def fill(key, neutral):
            for src, _ in sources:
                if key in src:
                    data[key] = src[key]
                    return
            data[key] = neutral

        fill("roeAnnual", data["roeThreshold"])
        fill("per", 15.0)
        fill("pbr", 1.5)
        fill("dividendYield", 0.0)
        fill("debtRatio", 80.0)
        fill("epsGrowthQoQ", 0.0)

        # 각 항목이 어디서 왔는지 기록 (화면에 출처 표시용)
        _LABELS = {"roeAnnual": "ROE", "per": "PER", "pbr": "PBR",
                   "dividendYield": "배당", "debtRatio": "부채비율", "epsGrowthQoQ": "EPS성장"}
        srcmap = {}
        for key, lab in _LABELS.items():
            chosen = "기본값(중립)"
            for src, label in sources:
                if key in src:
                    chosen = label
                    break
            srcmap[lab] = chosen
        data["_src"] = srcmap

        # 분기 재무제표 기반 TTM ROE: DART/네이버/.info 에서 ROE를 못 받았을 때 채움
        if srcmap.get("ROE") == "기본값(중립)":
            roeT = fetch_yf_quarter_roe(ticker)
            if roeT is not None:
                data["roeAnnual"] = roeT
                data["roeQuarter"] = roeT
                srcmap["ROE"] = "yfinance(분기 TTM)"

        # EPS 성장이 DART·네이버·yfinance(.info) 모두 없으면 → 분기 재무제표로 직접 계산
        if not any("epsGrowthQoQ" in s for s, _ in sources):
            g2 = fetch_yf_quarter_eps_growth(ticker)
            if g2 is not None:
                data["epsGrowthQoQ"] = g2
                data["epsAccelQuarters"] = 2 if g2 > 0 else 0
                srcmap["EPS성장"] = "yfinance(분기 재무제표)"

        # EPS 가속 분기수 / 목표주가는 출처 dict에서 따라옴
        esrc = next((s for s, _ in sources if "epsGrowthQoQ" in s), {})
        data["epsAccelQuarters"] = esrc.get("epsAccelQuarters", 2 if data["epsGrowthQoQ"] > 0 else 0)
        if "targetPrice" in nv:
            data["targetPrice"] = nv["targetPrice"]
        elif "targetPrice" in yf_f:
            data["targetPrice"] = yf_f["targetPrice"]

        # PEG = PER / (EPS성장률) — 성장 대비 밸류
        g = data.get("epsGrowthQoQ", 0)
        data["pegRatio"] = r1(data["per"] / g) if g and g > 0 else 3.0

        # 매크로(시장 공통) 수집
        mac, w4 = fetch_macro()
        data.update(mac)
        warnings += w4

        if not dt and not nv and not yf_f:
            warnings.append("ROE·EPS성장·PER·PBR·배당은 자동수집이 안 돼 중립값입니다. 사이드바에서 직접 보정하세요.")

    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    return data, chart_df, warnings


# ----------------------------------------------------------------------------
# 지표 점수 정의 (웹 버전과 동일한 공식)
# ----------------------------------------------------------------------------
def _distHigh(d): return (1 - d["price"] / d["high52w"]) * 100
def _distLow(d):  return (d["price"] / d["low52w"] - 1) * 100
def _dcfUp(d):    return (d["dcfFairValue"] / d["price"] - 1) * 100
def _tgtUp(d):    return (d["targetPrice"] / d["price"] - 1) * 100

def _roe_eff(d):
    """점수용 ROE = TTM(최근 1년) ROE. 최근 분기는 TTM에 이미 포함됨."""
    v = d.get("roeAnnual")
    if v is None:
        v = d.get("roeQuarter")
    return v if v is not None else d["roeThreshold"]

_REGIME = {"STRONG_BULL": 85, "BULL": 68, "NEUTRAL": 50, "BEAR": 25}
_OBV = {"up": 75, "flat": 50, "down": 25}          # 스마트머니: 점수로 직접 사용
_KAL = {"bullish": 80, "neutral": 50, "bearish": 20}

METRICS = [
    dict(tag="C", title="EPS 가속도", sub="분기 순이익 성장(최근 분기 기준)",
         score=lambda d: clamp(50 + d["epsGrowthQoQ"]*0.4 + (10 if d["epsAccelQuarters"] >= 2 else 0)),
         comment=lambda d: f"최근 분기 순이익이 {'+' if d['epsGrowthQoQ']>=0 else ''}{r1(d['epsGrowthQoQ'])}% 변동했어요. " +
                           ("성장세예요." if d["epsAccelQuarters"] >= 2 else "성장 신호는 약해요.")),
    dict(tag="A", title="ROE 실적", sub="자기자본이익률(TTM·최근1년)",
         score=lambda d: clamp(55 + (_roe_eff(d) - d["roeThreshold"]) * 1.8) if _roe_eff(d) >= d["roeThreshold"]
                         else clamp(_roe_eff(d) / d["roeThreshold"] * 55, 0, 55),
         comment=lambda d: (f"ROE(TTM) {r1(_roe_eff(d))}% (기준 {r1(d['roeThreshold'])}%). " +
                            ("기준 통과예요." if _roe_eff(d) >= d["roeThreshold"] else "기준 미달이에요.") +
                            (" 최근 분기 실적은 EPS 가속도에 반영돼요." ))),
    dict(tag="N", title="신고가·피벗 돌파", sub="52주 최고가 및 패턴 돌파",
         score=lambda d: clamp(90 - _distHigh(d)*2.0 + (6 if d["pivotBreak"] else 0)),
         comment=lambda d: f"52주 최고가에서 {r1(_distHigh(d))}% 아래에 있어요. " +
                           ("아직 신고가까지 거리가 있어요. " if _distHigh(d) > 3 else "신고가 부근이에요. ") +
                           ("컵앤핸들 피벗 돌파가 감지됐어요." if d["pivotBreak"] else "")),
    dict(tag="S", title="거래량 확인 돌파", sub="기관 참여를 동반한 거래량 급증",
         score=lambda d: clamp(60 + (d["volumeRatioVsAvg"]-1)*40) if d["breakoutSignal"]
                         else clamp(35 + (d["volumeRatioVsAvg"]-1)*20),
         comment=lambda d: f"거래량이 평소의 {r1(d['volumeRatioVsAvg'])}배예요. " +
                           ("거래량을 동반한 돌파가 확인됐어요." if d["breakoutSignal"]
                            else "아직 돌파 신호는 없어요(거래량 수준만 평가).")),
    dict(tag="L", title="주도주 판별", sub="시장 대비 상대강도(RS)",
         score=lambda d: clamp(d["rsRating"]),
         comment=lambda d: f"시장 대비 상대강도(RS) {int(d['rsRating'])}점이에요. " +
                           ("시장을 이끄는 주도주예요." if d["rsRating"] >= 80 else "주도력이 보통이에요." if d["rsRating"] >= 50 else "주도력이 약해요.")),
    dict(tag="I", title="기관 수급", sub="기관 자금의 매수-매도 흐름",
         score=lambda d: clamp(d["mfi"]),
         comment=lambda d: f"기관 자금 흐름: '{'매수 우위' if d['mfi']>=70 else '관망' if d['mfi']>=45 else '매도 우위'}'이에요 (MFI {int(d['mfi'])})."),
    dict(tag="M", title="시장 방향", sub="전체 시장 추세와 방향성",
         score=lambda d: clamp(_REGIME.get(d["marketRegime"], 50) + clamp(d["adx"]-20, 0, 30)*0.5),
         comment=lambda d: f"현재 시장 방향: '[M] {d['marketRegime']}'이에요. 추세 강도 ADX {int(d['adx'])}."),
    dict(tag="Quant", title="가치·퀄리티 팩터", sub="저평가+고품질 종목 선별",
         score=lambda d: clamp(60 + d["factorAlpha"]*1.5),
         comment=lambda d: f"가치·퀄리티 팩터 알파 {'+' if d['factorAlpha']>=0 else ''}{r1(d['factorAlpha'])}점이에요. " +
                           ("저평가 매력이 있어요." if d["factorAlpha"] > 0 else "중립이에요." if d["factorAlpha"] == 0 else "프리미엄 구간이에요.")),
    dict(tag="Quant", title="평균 회귀", sub="RSI·Z-Score 기반(과매도=기회)",
         score=lambda d: clamp(60 - d["zScoreMeanRev"]*15),
         comment=lambda d: f"RSI {int(d['rsi'])}로 {'과매수' if d['rsi']>=70 else '과매도' if d['rsi']<=30 else '중립'} 구간이에요. " +
                           ("단기 조정 가능성이 있어요. " if d["zScoreMeanRev"] >= 1.5 else "되돌림 여유가 있어요. " if d["zScoreMeanRev"] <= -1 else "") + f"Z-Score {'+' if d['zScoreMeanRev']>=0 else ''}{r1(d['zScoreMeanRev'])}."),
    dict(tag="Quant", title="모멘텀", sub="12개월 수익률 기반 추세 지속력",
         score=lambda d: clamp(50 + d["return12m"]*0.35),
         comment=lambda d: f"1년간 수익률 {'+' if d['return12m']>=0 else ''}{r1(d['return12m'])}%로 모멘텀이 {'강해요' if d['return12m']>50 else '보통이에요' if d['return12m']>0 else '약해요'}."),
    dict(tag="Quant", title="다중 시간대", sub="단기·중기·장기 추세 종합",
         score=lambda d: clamp(40 + d["alignedTimeframes"]*20),
         comment=lambda d: f"단기·중기·장기 추세 종합: {'상승' if d['alignedTimeframes']>=2 else '혼조'} 추세예요 (정렬 {d['alignedTimeframes']}/3)."),
    dict(tag="Quant", title="낙폭 위험도", sub="최근 최대 하락폭(MDD) 평가",
         score=lambda d: clamp(95 - abs(d["mddPct"])*1.5),
         comment=lambda d: f"최근 최대 낙폭(MDD) {r1(d['mddPct'])}%이에요. 위험도는 '{'낮음' if abs(d['mddPct'])<15 else '보통' if abs(d['mddPct'])<30 else '높음'}'으로 평가돼요."),
    dict(tag="Quant", title="스마트머니 흐름", sub="기관 자금 흐름과 OBV 추세",
         score=lambda d: _OBV.get(d["obvTrend"], 50),
         comment=lambda d: "스마트머니 흐름 — OBV 추세: " + {"up": "상승", "flat": "횡보", "down": "하락"}.get(d["obvTrend"], "횡보") + "이에요."),
    dict(tag="Quant", title="DCF 적정가", sub="DCF 적정가 대비 상승 여력",
         score=lambda d: clamp(50 + _dcfUp(d)*0.5),
         comment=lambda d: f"DCF 적정가 대비 상승 여력 {'+' if _dcfUp(d)>=0 else ''}{r1(_dcfUp(d))}%이에요. 전망은 '{'강력 매수' if _dcfUp(d)>30 else '매수' if _dcfUp(d)>0 else '관망'}'예요."),
    dict(tag="Quant", title="공매도 비율", sub="공매도 부담 수준 평가",
         score=lambda d: clamp(75 - d["shortRatioPct"]*7),
         comment=lambda d: f"공매도 비율 {r1(d['shortRatioPct'])}%로 부담은 '{'낮음' if d['shortRatioPct']<5 else '높음'}'이에요."),
    dict(tag="Math", title="허스트 지수", sub="추세 지속성과 방향 예측력",
         score=lambda d: clamp(50 + (d["hurst"]-0.5)*160),
         comment=lambda d: f"허스트 지수 {r1(d['hurst'])}로 '{'강한 추세' if d['hurst']>0.55 else '평균회귀' if d['hurst']<0.45 else '랜덤워크'}'를 나타내요."),
    dict(tag="Math", title="칼만 필터", sub="노이즈 제거 후 추세 신호",
         score=lambda d: _KAL.get(d["kalmanSignal"], 50),
         comment=lambda d: "칼만 필터 신호: " + {"bullish": "상승", "neutral": "중립", "bearish": "하락"}.get(d["kalmanSignal"], "중립") + "이에요."),
    dict(tag="Math", title="통계적 Z-Score", sub="통계적 과매수/과매도 위치",
         score=lambda d: clamp(65 - d["zScoreStat"]*12),
         comment=lambda d: f"통계적 Z-Score {'+' if d['zScoreStat']>=0 else ''}{r1(d['zScoreStat'])}이에요. {'과매수 구간' if d['zScoreStat']>1.5 else '과매도 구간' if d['zScoreStat']<-1.5 else '정상 범위'}예요."),
    dict(tag="Adj", title="변동성 조정", sub="낙폭 대비 수익률 효율(칼마형)",
         score=lambda d: clamp(40 + (d["return12m"] / max(abs(d["mddPct"]), 1.0)) * 8),
         comment=lambda d: f"12개월 수익률 {r1(d['return12m'])}% 대비 최대낙폭 {r1(d['mddPct'])}% — "
                           f"위험 대비 효율이 {'높아요' if (d['return12m']/max(abs(d['mddPct']),1))>=2 else '보통이에요' if (d['return12m']/max(abs(d['mddPct']),1))>=0.5 else '낮아요'}."),
    dict(tag="Sentiment", title="시장 심리 추정", sub="가격·거래량 기반 투자 심리",
         score=lambda d: clamp(50 + (d["upVolPct"]-50)*0.8 + (d["closeStrength"]-50)*0.6 + d["gapDir"]*5),
         comment=lambda d: f"가격·거래량으로 추정한 심리는 '{'상승 우위' if d['upVolPct']>=60 else '중립'}'이에요. 상승 거래량 {int(d['upVolPct'])}%, 종가 강도 {int(d['closeStrength'])}%."),

    # ---------------- 가치투자 (Value) ----------------
    dict(tag="Value", title="PER 밸류", sub="이익 대비 주가(낮을수록 저평가)",
         score=lambda d: clamp(105 - d["per"]*4) if d["per"] > 0 else 20,
         comment=lambda d: f"PER {r1(d['per'])}배예요. {'저평가 구간' if 0<d['per']<10 else '적정' if d['per']<20 else '고평가 구간' if d['per']>0 else '적자(PER 의미 없음)'}이에요."),
    dict(tag="Value", title="PBR 밸류", sub="순자산 대비 주가(낮을수록 저평가)",
         score=lambda d: clamp(100 - d["pbr"]*25),
         comment=lambda d: f"PBR {r1(d['pbr'])}배예요. {'자산가치 대비 싸요' if d['pbr']<1 else '적정' if d['pbr']<2.5 else '프리미엄 구간'}이에요."),
    dict(tag="Value", title="배당 매력", sub="시가배당률",
         score=lambda d: clamp(d["dividendYield"] * 20),
         comment=lambda d: f"시가배당률 {r1(d['dividendYield'])}%예요. {'배당 매력이 높아요' if d['dividendYield']>=3 else '배당은 보통이에요' if d['dividendYield']>0 else '배당이 거의 없어요'}."),
    dict(tag="Value", title="재무 안정성", sub="부채비율(낮을수록 안전)",
         score=lambda d: clamp(120 - d["debtRatio"]*0.8),
         comment=lambda d: f"부채비율 {r1(d['debtRatio'])}%예요. {'재무가 탄탄해요' if d['debtRatio']<50 else '보통이에요' if d['debtRatio']<100 else '부채 부담이 있어요'}."),
    dict(tag="Value", title="안전마진", sub="DCF 적정가 대비 괴리",
         score=lambda d: clamp(50 + _dcfUp(d) * 0.5),
         comment=lambda d: f"적정가 대비 {'+' if _dcfUp(d)>=0 else ''}{r1(_dcfUp(d))}%의 안전마진이에요. {'매력적' if _dcfUp(d)>20 else '보통' if _dcfUp(d)>0 else '여유 없음'}이에요."),
    dict(tag="Value", title="성장 대비 밸류(PEG)", sub="PER ÷ 이익성장률",
         score=lambda d: clamp(100 - d["pegRatio"] * 35),
         comment=lambda d: f"PEG {r1(d['pegRatio'])}예요. {'성장 대비 저평가' if d['pegRatio']<1 else '적정' if d['pegRatio']<2 else '성장 대비 비쌈'}이에요."),

    # ---------------- 매크로 (시장 환경) ----------------
    dict(tag="Macro", title="시장 추세", sub="KOSPI 국면",
         score=lambda d: {"STRONG_BULL": 90, "BULL": 70, "NEUTRAL": 50, "BEAR": 25}.get(d["marketRegime"], 50),
         comment=lambda d: f"전체 시장은 '{d['marketRegime']}' 국면이에요. {'위험자산에 우호적' if 'BULL' in d['marketRegime'] else '신중 구간'}이에요."),
    dict(tag="Macro", title="환율(USD/KRW)", sub="원화 방향(약세=위험회피)",
         score=lambda d: {"down": 75, "flat": 52, "up": 32}.get(d["usdkrwTrend"], 50),
         comment=lambda d: "원/달러 환율이 " + {"down": "하락(원화 강세) — 외국인 우호적", "flat": "횡보 — 중립", "up": "상승(원화 약세) — 위험회피"}.get(d["usdkrwTrend"], "중립") + "이에요."),
    dict(tag="Macro", title="금리 방향", sub="완화=우호 / 긴축=부담",
         score=lambda d: {"down": 75, "flat": 55, "up": 35}.get(d["rateTrend"], 55),
         comment=lambda d: "금리가 " + {"down": "하락(완화) — 주식에 우호적", "flat": "횡보 — 중립", "up": "상승(긴축) — 밸류에 부담"}.get(d["rateTrend"], "중립") + "이에요."),
    dict(tag="Macro", title="변동성(VIX)", sub="공포지수(낮을수록 안정)",
         score=lambda d: clamp(115 - d["vix"] * 3),
         comment=lambda d: f"VIX {r1(d['vix'])}예요. {'시장이 안정적' if d['vix']<20 else '경계 구간' if d['vix']<30 else '공포 구간'}이에요."),
]


# ----------------------------------------------------------------------------
# 종합 계산
# ----------------------------------------------------------------------------
def compute_all(d: dict) -> dict:
    m = [dict(tag=x["tag"], title=x["title"], sub=x["sub"],
              canNeg=x.get("canNeg", False),
              value=r1(x["score"](d)), text=x["comment"](d)) for x in METRICS]
    get = lambda t: next(x["value"] for x in m if x["title"] == t)

    trend = ((d["adx"] >= 25) + (d["hurst"] >= 0.55) + (d["alignedTimeframes"] >= 3)
             + ("BULL" in d["marketRegime"]) + (_distHigh(d) < 15))
    momentum = ((2 if d["return12m"] > 100 else 1 if d["return12m"] > 0 else 0)
                + (2 if d["rsRating"] >= 90 else 1 if d["rsRating"] >= 70 else 0)
                - (2 if d["rsi"] >= 70 else 0))
    supply = ((d["obvTrend"] == "up") + (d["vwapPosition"] == "above")
              + (d["mfi"] >= 60) + (d["mfi"] >= 70))
    volat = 4 if abs(d["mddPct"]) < 7 else 3 if abs(d["mddPct"]) < 12 else 2

    cats = dict(
        추세=dict(v=int(clamp(trend, 0, 5)), sub="강한 상승 추세" if d["adx"] >= 40 else "추세 형성 중"),
        모멘텀=dict(v=int(clamp(momentum, 0, 5)), sub="RSI 과매수 — 차익 경계" if d["rsi"] >= 70 else "양호"),
        변동성=dict(v=int(clamp(volat, 0, 5)), sub="관망 (BB폭 상위 분위)"),
        수급=dict(v=int(clamp(supply, 0, 5)), sub="기관 우위 (VWAP 위 + OBV↑)" if d["vwapPosition"] == "above" else "관망"),
    )

    regimeAvg = (get("시장 방향") + get("허스트 지수") + get("다중 시간대")) / 3
    valueAvg = (get("가치·퀄리티 팩터") + get("DCF 적정가")) / 2
    momNet = (get("모멘텀") + get("평균 회귀")) / 2
    supplyAvg = (get("기관 수급") + get("스마트머니 흐름")) / 2
    fundAvg = (get("EPS 가속도") + get("ROE 실적")) / 2
    riskAvg = (get("낙폭 위험도") + get("공매도 비율")) / 2
    composite_tech = round(0.24*regimeAvg + 0.18*valueAvg + 0.14*momNet
                           + 0.14*supplyAvg + 0.14*fundAvg + 0.16*riskAvg)

    # ---- 투자 스타일별 점수 ----
    def avg_titles(titles):
        vals = [get(t) for t in titles]
        return round(sum(vals) / len(vals)) if vals else 0

    def avg_tag(tag):
        vals = [x["value"] for x in m if x["tag"] == tag]
        return round(sum(vals) / len(vals)) if vals else 0

    styles = {
        "CAN SLIM": avg_titles(["EPS 가속도", "ROE 실적", "신고가·피벗 돌파",
                                "거래량 확인 돌파", "주도주 판별", "기관 수급", "시장 방향"]),
        "가치": avg_tag("Value"),
        "모멘텀": avg_titles(["모멘텀", "다중 시간대", "주도주 판별", "신고가·피벗 돌파"]),
        "퀄리티": avg_titles(["ROE 실적", "가치·퀄리티 팩터", "재무 안정성"]),
        "매크로": avg_tag("Macro"),
    }
    # ---- 스타일 통합 종합 점수 (가중치는 자유 튜닝 가능) ----
    composite = round(0.28*styles["CAN SLIM"] + 0.24*styles["가치"]
                      + 0.20*styles["모멘텀"] + 0.16*styles["퀄리티"]
                      + 0.12*styles["매크로"])

    entryTiming = round(clamp(60 + supply*4 - (28 if d["rsi"] >= 70 else 0)
                              - (8 if d["zScoreMeanRev"] >= 2 else 0), 0, 100))

    a = d["atr"]
    buy = round(d["price"] - 1.0 * a)        # 눌림목 대기: 현재가 − 1 ATR
    stop = round(buy - 2.0 * a)              # 변동성 기반 손절: 매수 − 2 ATR
    risk = buy - stop                        # = 2 ATR
    t1 = round(buy + 2.0 * risk)             # 1차 익절: 손익비 2:1
    t2 = round(buy + 3.0 * risk)             # 2차 익절: 손익비 3:1
    rr = r1(2.0) if risk > 0 else 0.0        # 1차 익절 기준 손익비

    verdict = ("강력 매수" if composite >= 80 else
               "매수" if composite >= 68 else
               "비중 확대" if composite >= 58 else
               "관망" if composite >= 45 else
               "비중 축소" if composite >= 35 else
               "회피")
    return dict(metrics=m, cats=cats, composite=composite, composite_tech=composite_tech,
                styles=styles, entryTiming=entryTiming,
                buy=buy, stop=stop, t1=t1, t2=t2, rr=rr, verdict=verdict,
                distHigh=r1(_distHigh(d)), distLow=r1(_distLow(d)),
                dcfUpside=r1(_dcfUp(d)), targetUpside=r1(_tgtUp(d)))


# ----------------------------------------------------------------------------
# 동일업종(관련주) 코드 자동 수집
# ----------------------------------------------------------------------------
def fetch_peers(ticker: str, limit: int = 5):
    """네이버 금융 '동일업종비교'에서 같은 업종 종목코드 수집 (best-effort)."""
    code, rwarn = resolve_ticker(ticker)
    if not code or not (str(code).isdigit() and len(str(code)) == 6):
        return [], [rwarn] if rwarn else ["국내 6자리 종목코드/정확한 종목명만 동일업종 탐색이 됩니다."]
    code = str(code)
    try:
        import requests, re
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(f"https://finance.naver.com/item/main.naver?code={code}",
                         headers=headers, timeout=8)
        r.encoding = r.apparent_encoding or "euc-kr"
        soup = BeautifulSoup(r.text, "lxml")

        codes = []
        head = soup.find(string=re.compile("동일업종"))   # '동일업종비교' 영역
        if head is not None:
            node = head.find_parent()
            table = node.find_next("table") if node else None
            if table is not None:
                for a in table.find_all("a", href=True):
                    m = re.search(r"code=(\d{6})", a["href"])
                    if m:
                        codes.append(m.group(1))
        # 자기 자신 제거 + 중복 제거(순서 유지)
        seen, peers = set(), []
        for c in codes:
            if c != code and c not in seen:
                seen.add(c)
                peers.append(c)
        peers = peers[:limit]
        if not peers:
            return [], ["동일업종 자동 탐색 실패(페이지 구조 변경 가능). 직접 입력하세요."]
        return peers, []
    except Exception as e:
        return [], [f"동일업종 탐색 실패: {e}"]


# ----------------------------------------------------------------------------
# 관련주 비교용 요약 평가
# ----------------------------------------------------------------------------
def evaluate(ticker: str, period_days: int = 400) -> dict:
    """단일 종목을 평가해 비교용 요약 dict 반환."""
    d, _df, _w = build_inputs(ticker, period_days=period_days)
    r = compute_all(d)
    return {
        "ticker": ticker,
        "name": d.get("name", ticker),
        "price": d["price"],
        "composite": r["composite"],
        "verdict": r["verdict"],
        "CAN SLIM": r["styles"]["CAN SLIM"],
        "가치": r["styles"]["가치"],
        "모멘텀": r["styles"]["모멘텀"],
        "퀄리티": r["styles"]["퀄리티"],
        "매크로": r["styles"]["매크로"],
        "PER": d.get("per"),
        "PBR": d.get("pbr"),
        "ROE": d.get("roeAnnual"),
        "배당%": d.get("dividendYield"),
        "12M%": d.get("return12m"),
        "RSI": r1(d.get("rsi", 0)),
    }

def compare(tickers, period_days: int = 400):
    """여러 종목을 평가해 리스트로 반환 (실패 종목은 건너뜀)."""
    rows = []
    for t in tickers:
        t = str(t).strip()
        if not t:
            continue
        try:
            rows.append(evaluate(t, period_days))
        except Exception:
            pass
    return rows


# ----------------------------------------------------------------------------
# 콘솔 단독 실행
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    tk = sys.argv[1] if len(sys.argv) > 1 else "000660"
    d, df, warns = build_inputs(tk)
    res = compute_all(d)
    print(f"\n=== {d.get('name', tk)} ({tk}) ===")
    print(f"현재가 {d['price']:,.0f}  /  종합 점수 {res['composite']}점  →  {res['verdict']}")
    print("스타일별: " + "  ".join(f"{k} {v}" for k, v in res["styles"].items()))
    print(f"추세 {res['cats']['추세']['v']}/5  모멘텀 {res['cats']['모멘텀']['v']}/5  "
          f"변동성 {res['cats']['변동성']['v']}/5  수급 {res['cats']['수급']['v']}/5")
    print(f"진입타이밍 {res['entryTiming']}/100  |  매수 {res['buy']:,}  손절 {res['stop']:,}  "
          f"1차익절 {res['t1']:,}  2차익절 {res['t2']:,}  (R:R {res['rr']}:1)\n")
    for x in res["metrics"]:
        print(f"  [{x['tag']:>9}] {x['title']:<14} {x['value']:>6}  {x['text']}")
    if warns:
        print("\n경고:", *warns, sep="\n  - ")
