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
    roeAnnual=61.0, roeThreshold=17.0, epsGrowthQoQ=396.6, epsAccelQuarters=2,
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
    # 2) yfinance
    try:
        import yfinance as yf
        yt = ticker if "." in ticker or not ticker.isdigit() else ticker + ".KS"
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
            eps, bps = safe(row.get("EPS")), safe(row.get("BPS"))
            if bps > 0:
                out["roeAnnual"] = r1(eps / bps * 100)   # ROE ≈ EPS/BPS
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
    # 3) yfinance
    try:
        import yfinance as yf
        yt = ticker if ("." in ticker or not code.isdigit()) else code + ".KS"
        info = yf.Ticker(yt).info
        nm = info.get("shortName") or info.get("longName")
        if nm:
            return str(nm).strip()
    except Exception:
        pass
    return None


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
    EPS 성장은 연간 실적(추정치 'E' 제외)의 YoY로 계산해 안정적으로."""
    out = {}
    try:
        cols = [(" ".join(map(str, c)) if isinstance(c, tuple) else str(c)) for c in fin.columns]
        fin = fin.copy()
        fin.columns = cols
        label_col = cols[0]
        labels = fin[label_col].astype(str)
        data_cols = cols[1:]
        annual_cols = [c for c in data_cols if "연간" in c] or data_cols  # 연간 칼럼 우선

        def row(key):
            idx = labels[labels.str.contains(key, na=False)].index
            return fin.loc[idx[0]] if len(idx) else None

        def last_num(key, columns=None):
            rr = row(key)
            if rr is None:
                return None
            use = columns if columns is not None else annual_cols
            actual = [_to_num(rr[c]) for c in use if "(E)" not in c]
            actual = [v for v in actual if v is not None]
            if actual:
                return actual[-1]               # 최근 연간 실적 우선
            anyv = [_to_num(rr[c]) for c in use]
            anyv = [v for v in anyv if v is not None]
            return anyv[-1] if anyv else None    # 없으면 추정치라도

        roe = last_num("ROE")
        if roe is not None:
            out["roeAnnual"] = r1(roe)

        # EPS YoY: 연간 칼럼에서 추정치(E) 제외한 실적 2개로 계산
        eps_row = row("EPS")
        if eps_row is not None:
            actual = [(c, _to_num(eps_row[c])) for c in annual_cols if "(E)" not in c]
            actual = [(c, v) for c, v in actual if v is not None]
            pair = None
            if len(actual) >= 2:
                pair = (actual[-2][1], actual[-1][1])
            elif len(actual) == 1:  # 실적 1개뿐이면 추정치와 비교
                est = [_to_num(eps_row[c]) for c in annual_cols if "(E)" in c]
                est = [v for v in est if v is not None]
                if est:
                    pair = (actual[-1][1], est[-1])
            if pair and pair[0] not in (0, None):
                out["epsGrowthQoQ"] = r1((pair[1] - pair[0]) / abs(pair[0]) * 100)
                out["epsAccelQuarters"] = 2 if out["epsGrowthQoQ"] > 0 else 0

        per = last_num("PER")
        if per is not None and per > 0:
            out["per"] = r1(per)
        pbr = last_num("PBR")
        if pbr is not None and pbr > 0:
            out["pbr"] = r1(pbr)
        div = last_num("시가배당")   # 시가배당률(%) — 주당배당금이 아님
        if div is not None:
            out["dividendYield"] = r1(div)
        debt = last_num("부채비율")
        if debt is not None:
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
    미국 기반 클라우드(예: Streamlit Cloud)에서 네이버보다 안정적일 때가 많음."""
    out, warns = {}, []
    code = _norm_krx(ticker)
    yt = ticker if ("." in ticker or not code.isdigit()) else code + ".KS"
    try:
        import yfinance as yf
        info = getattr(yf.Ticker(yt), "info", None) or {}
        roe = info.get("returnOnEquity")
        if roe is not None:
            out["roeAnnual"] = r1(roe * 100)
        per = info.get("trailingPE") or info.get("forwardPE")
        if per and per > 0:
            out["per"] = r1(per)
        pbr = info.get("priceToBook")
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


def build_inputs(ticker: str, overrides: dict | None = None, period_days: int = 400):
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

        # 펀더멘털: 네이버 → yfinance(미국 야후, 클라우드에서 더 안정적) → 중립값 순
        nv, w3 = fetch_naver(ticker)
        warnings += w3
        yf_f, w5 = fetch_yf_fundamentals(ticker)
        warnings += w5

        def fill(key, neutral):
            if key in nv:
                data[key] = nv[key]          # 1순위: 네이버
            elif key in yf_f:
                data[key] = yf_f[key]        # 2순위: yfinance
            else:
                data[key] = neutral          # 3순위: 중립값

        fill("roeAnnual", data["roeThreshold"])
        fill("per", 15.0)
        fill("pbr", 1.5)
        fill("dividendYield", 0.0)
        fill("debtRatio", 80.0)
        fill("epsGrowthQoQ", 0.0)
        # EPS 가속 분기수 / 목표주가는 출처 dict에서 따라옴
        src = nv if "epsGrowthQoQ" in nv else (yf_f if "epsGrowthQoQ" in yf_f else {})
        data["epsAccelQuarters"] = src.get("epsAccelQuarters", 2 if data["epsGrowthQoQ"] > 0 else 0)
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

        if not nv and not yf_f:
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

_REGIME = {"STRONG_BULL": 80, "BULL": 62, "NEUTRAL": 45, "BEAR": 20}
_OBV = {"up": 9, "flat": 0, "down": -25}
_KAL = {"bullish": 75, "neutral": 37.5, "bearish": 15}

METRICS = [
    dict(tag="C", title="EPS 가속도", sub="분기 순이익 성장 가속 여부",
         score=lambda d: clamp(45 + d["epsGrowthQoQ"]/25 + (15 if d["epsAccelQuarters"] >= 2 else 0)),
         comment=lambda d: f"지난 분기 순이익이 {'+' if d['epsGrowthQoQ']>=0 else ''}{r1(d['epsGrowthQoQ'])}% 변동했어요. " +
                           ("연속 성장 중이에요." if d["epsAccelQuarters"] >= 2 else "가속 신호는 약해요.")),
    dict(tag="A", title="연간 ROE 실적", sub="자기자본이익률 기준 충족 여부",
         score=lambda d: clamp(12 + math.log2(max(d["roeAnnual"]/d["roeThreshold"], 1e-6))*4) if d["roeAnnual"] >= d["roeThreshold"]
                         else clamp(d["roeAnnual"]/d["roeThreshold"]*35, 0, 40),
         comment=lambda d: f"자기자본이익률(ROE) {r1(d['roeAnnual'])}%이고, 기준({r1(d['roeThreshold'])}%)을 " +
                           ("통과했어요. 돈을 잘 버는 회사예요." if d["roeAnnual"] >= d["roeThreshold"] else "미달이에요.")),
    dict(tag="N", title="신고가·피벗 돌파", sub="52주 최고가 및 패턴 돌파",
         score=lambda d: clamp(50 - _distHigh(d)*4 + (5 if d["pivotBreak"] else 0)),
         comment=lambda d: f"52주 최고가에서 {r1(_distHigh(d))}% 아래에 있어요. " +
                           ("아직 신고가까지 거리가 있어요. " if _distHigh(d) > 3 else "신고가 부근이에요. ") +
                           ("컵앤핸들 피벗 돌파가 감지됐어요." if d["pivotBreak"] else "")),
    dict(tag="S", title="거래량 확인 돌파", sub="기관 참여를 동반한 거래량 급증",
         score=lambda d: clamp(20 + (d["volumeRatioVsAvg"]-1)*60) if d["breakoutSignal"]
                         else r1(-(d["volumeRatioVsAvg"]-0.3)*15),
         comment=lambda d: f"거래량이 평소의 {r1(d['volumeRatioVsAvg'])}배예요. " +
                           ("돌파를 거래량이 확인했어요." if d["breakoutSignal"] else "돌파 신호는 없어요.")),
    dict(tag="L", title="주도주 판별", sub="시장 대비 상대강도 측정",
         score=lambda d: clamp(d["rsRating"] - 79),
         comment=lambda d: f"시장 대비 상대강도(RS) {int(d['rsRating'])}점이에요. " +
                           ("시장을 이끄는 주도주예요." if d["rsRating"] >= 80 else "주도력이 약해요.")),
    dict(tag="I", title="기관 수급", sub="기관 자금의 매수-매도 흐름",
         score=lambda d: clamp(d["mfi"]*0.84),
         comment=lambda d: f"기관 자금 흐름: '{'매수 우위' if d['mfi']>=80 else '관망' if d['mfi']>=60 else '매도 우위'}'이에요. " +
                           (f"매수 압력이 강해요 (MFI {int(d['mfi'])})." if d["mfi"] >= 60 else "")),
    dict(tag="M", title="시장 방향", sub="전체 시장 추세와 방향성",
         score=lambda d: clamp(_REGIME.get(d["marketRegime"], 45) + clamp(d["adx"]-25, 0, 25)*0.64),
         comment=lambda d: f"현재 시장 방향: '[M] {d['marketRegime']} ✅'이에요. 추세 강도 ADX {int(d['adx'])}."),
    dict(tag="Quant", title="가치·퀄리티 팩터", sub="저평가+고품질 종목 선별",
         score=lambda d: clamp(73 + d["factorAlpha"]),
         comment=lambda d: f"가치·퀄리티 팩터 알파 {'+' if d['factorAlpha']>=0 else ''}{r1(d['factorAlpha'])}점이에요. " +
                           ("저평가 매력이 있어요." if d["factorAlpha"] > 0 else "프리미엄 구간이에요.")),
    dict(tag="Quant", title="평균 회귀", sub="RSI·Z-Score 기반 반등/조정 가능성",
         score=lambda d: clamp(60 - d["zScoreMeanRev"]*17.33),
         comment=lambda d: f"RSI {int(d['rsi'])}로 {'과매수' if d['rsi']>=70 else '과매도' if d['rsi']<=30 else '중립'} 구간이에요. " +
                           ("단기 조정 가능성이 있어요. " if d["zScoreMeanRev"] >= 1.5 else "") + f"Z-Score +{r1(d['zScoreMeanRev'])}."),
    dict(tag="Quant", title="모멘텀", sub="12개월 수익률 기반 추세 지속력",
         score=lambda d: clamp(40 + d["return12m"]/20),
         comment=lambda d: f"1년간 수익률 +{r1(d['return12m'])}%로 모멘텀이 {'강해요' if d['return12m']>100 else '보통이에요'}."),
    dict(tag="Quant", title="다중 시간대", sub="단기·중기·장기 추세 종합",
         score=lambda d: clamp(45 + d["alignedTimeframes"]*10),
         comment=lambda d: f"단기·중기·장기 추세 종합: {'상승' if d['alignedTimeframes']>=2 else '혼조'} 추세예요."),
    dict(tag="Quant", title="낙폭 위험도", sub="최근 최대 하락폭(MDD) 평가",
         score=lambda d: clamp(100 - abs(d["mddPct"])*7.3),
         comment=lambda d: f"최근 최대 낙폭(MDD) {r1(d['mddPct'])}%이에요. 위험도는 '{'낮음' if abs(d['mddPct'])<7 else '보통' if abs(d['mddPct'])<15 else '높음'}'으로 평가돼요."),
    dict(tag="Quant", title="스마트머니 흐름", sub="기관 자금 흐름과 OBV 추세",
         score=lambda d: clamp(50 + _OBV.get(d["obvTrend"], 0)),
         comment=lambda d: "스마트머니 흐름 — OBV 추세: " + {"up": "상승", "flat": "횡보", "down": "하락"}.get(d["obvTrend"], "횡보") + "이에요."),
    dict(tag="Quant", title="DCF 적정가", sub="DCF 적정가 대비 상승 여력",
         score=lambda d: clamp(50 + _dcfUp(d)*0.44),
         comment=lambda d: f"DCF 적정가 대비 상승 여력 +{r1(_dcfUp(d))}%이에요. 전망은 '{'강력 매수' if _dcfUp(d)>30 else '매수' if _dcfUp(d)>0 else '관망'}'예요."),
    dict(tag="Quant", title="공매도 비율", sub="공매도 부담 수준 평가",
         score=lambda d: clamp(50 - d["shortRatioPct"]*5),
         comment=lambda d: f"공매도 비율 {r1(d['shortRatioPct'])}%로 위험도는 '{'NORMAL' if d['shortRatioPct']<5 else 'HIGH'}'이에요."),
    dict(tag="Math", title="허스트 지수", sub="추세 지속성과 방향 예측력",
         score=lambda d: clamp(50 + (d["hurst"]-0.5)*144),
         comment=lambda d: f"허스트 지수 {r1(d['hurst'])}로 '{'강한 추세' if d['hurst']>0.55 else '평균회귀' if d['hurst']<0.45 else '랜덤워크'}'를 나타내요."),
    dict(tag="Math", title="칼만 필터", sub="노이즈 제거 후 추세 신호",
         score=lambda d: _KAL.get(d["kalmanSignal"], 37.5),
         comment=lambda d: "칼만 필터 신호: " + {"bullish": "상승", "neutral": "중립", "bearish": "하락"}.get(d["kalmanSignal"], "중립") + "이에요."),
    dict(tag="Math", title="통계적 Z-Score", sub="통계적 과매수/과매도 위치",
         score=lambda d: clamp(50 - (abs(d["zScoreStat"])-1.2)*20),
         comment=lambda d: f"통계적 Z-Score +{r1(d['zScoreStat'])}이에요. {'정상 범위예요.' if abs(d['zScoreStat'])<1.5 else '극단 구간이에요.'}"),
    dict(tag="Adj", title="변동성 조정", sub="변동성 대비 수익률 효율성", canNeg=True,
         score=lambda d: r1(-(d["volMultiplier"]-1)*22.8),
         comment=lambda d: f"변동성 대비 수익률 효율을 반영해 최종 점수에 ×{r1(d['volMultiplier'])} 배율이 적용됐어요."),
    dict(tag="Sentiment", title="시장 심리 추정", sub="가격·거래량 기반 투자 심리",
         score=lambda d: clamp(55 + (d["upVolPct"]-50) + (d["closeStrength"]-50)*0.6 + d["gapDir"]*8),
         comment=lambda d: f"가격·거래량으로 추정한 심리는 '{'약한 상승' if d['upVolPct']>=60 else '중립'}'이에요. 상승 거래량 {int(d['upVolPct'])}%, 종가 강도 {int(d['closeStrength'])}%."),

    # ---------------- 가치투자 (Value) ----------------
    dict(tag="Value", title="PER 밸류", sub="이익 대비 주가(낮을수록 저평가)",
         score=lambda d: clamp(100 - (d["per"] - 5) * 4),
         comment=lambda d: f"PER {r1(d['per'])}배예요. {'저평가 구간' if d['per']<10 else '적정' if d['per']<20 else '고평가 구간'}이에요."),
    dict(tag="Value", title="PBR 밸류", sub="순자산 대비 주가(낮을수록 저평가)",
         score=lambda d: clamp(100 - (d["pbr"] - 0.5) * 30),
         comment=lambda d: f"PBR {r1(d['pbr'])}배예요. {'자산가치 대비 싸요' if d['pbr']<1 else '적정' if d['pbr']<2.5 else '프리미엄 구간'}이에요."),
    dict(tag="Value", title="배당 매력", sub="시가배당률",
         score=lambda d: clamp(d["dividendYield"] * 18),
         comment=lambda d: f"시가배당률 {r1(d['dividendYield'])}%예요. {'배당 매력이 높아요' if d['dividendYield']>=3 else '배당은 보통이에요' if d['dividendYield']>0 else '배당이 거의 없어요'}."),
    dict(tag="Value", title="재무 안정성", sub="부채비율(낮을수록 안전)",
         score=lambda d: clamp(110 - d["debtRatio"]),
         comment=lambda d: f"부채비율 {r1(d['debtRatio'])}%예요. {'재무가 탄탄해요' if d['debtRatio']<50 else '보통이에요' if d['debtRatio']<100 else '부채 부담이 있어요'}."),
    dict(tag="Value", title="안전마진", sub="DCF 적정가 대비 괴리",
         score=lambda d: clamp(50 + _dcfUp(d) * 0.5),
         comment=lambda d: f"적정가 대비 +{r1(_dcfUp(d))}%의 안전마진이에요. {'매력적' if _dcfUp(d)>20 else '보통' if _dcfUp(d)>0 else '여유 없음'}이에요."),
    dict(tag="Value", title="성장 대비 밸류(PEG)", sub="PER ÷ 이익성장률",
         score=lambda d: clamp(100 - d["pegRatio"] * 40),
         comment=lambda d: f"PEG {r1(d['pegRatio'])}예요. {'성장 대비 저평가' if d['pegRatio']<1 else '적정' if d['pegRatio']<2 else '성장 대비 비쌈'}이에요."),

    # ---------------- 매크로 (시장 환경) ----------------
    dict(tag="Macro", title="시장 추세", sub="KOSPI 국면",
         score=lambda d: {"STRONG_BULL": 90, "BULL": 70, "NEUTRAL": 45, "BEAR": 20}.get(d["marketRegime"], 45),
         comment=lambda d: f"전체 시장은 '{d['marketRegime']}' 국면이에요. {'위험자산에 우호적' if 'BULL' in d['marketRegime'] else '신중 구간'}이에요."),
    dict(tag="Macro", title="환율(USD/KRW)", sub="원화 방향(약세=위험회피)",
         score=lambda d: {"down": 75, "flat": 52, "up": 32}.get(d["usdkrwTrend"], 50),
         comment=lambda d: "원/달러 환율이 " + {"down": "하락(원화 강세) — 외국인 우호적", "flat": "횡보 — 중립", "up": "상승(원화 약세) — 위험회피"}.get(d["usdkrwTrend"], "중립") + "이에요."),
    dict(tag="Macro", title="금리 방향", sub="완화=우호 / 긴축=부담",
         score=lambda d: {"down": 75, "flat": 55, "up": 35}.get(d["rateTrend"], 55),
         comment=lambda d: "금리가 " + {"down": "하락(완화) — 주식에 우호적", "flat": "횡보 — 중립", "up": "상승(긴축) — 밸류에 부담"}.get(d["rateTrend"], "중립") + "이에요."),
    dict(tag="Macro", title="변동성(VIX)", sub="공포지수(낮을수록 안정)",
         score=lambda d: clamp(100 - (d["vix"] - 10) * 3),
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
    fundAvg = (get("EPS 가속도") + get("연간 ROE 실적")) / 2
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
        "CAN SLIM": avg_titles(["EPS 가속도", "연간 ROE 실적", "신고가·피벗 돌파",
                                "거래량 확인 돌파", "주도주 판별", "기관 수급", "시장 방향"]),
        "가치": avg_tag("Value"),
        "모멘텀": avg_titles(["모멘텀", "다중 시간대", "주도주 판별", "신고가·피벗 돌파"]),
        "퀄리티": avg_titles(["연간 ROE 실적", "가치·퀄리티 팩터", "재무 안정성"]),
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

    verdict = ("강력 매수" if composite >= 80 else "매집" if composite >= 60
               else "관망" if composite >= 40 else "회피")
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
