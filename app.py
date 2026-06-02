# -*- coding: utf-8 -*-
"""
app.py — 매수 평가 대시보드 (Streamlit / Windows PC 버전)
실행:  streamlit run app.py   (또는 run.bat 더블클릭)
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import stock_score as S

st.set_page_config(page_title="매수 평가 대시보드", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- 스타일
st.markdown("""
<style>
  .block-container{padding-top:1.2rem;max-width:1280px;}
  .stApp{background:#f1f5f9;}
  .card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px 16px;}
  .cat-name{font-size:12px;color:#64748b;}
  .cat-v{font-size:26px;font-weight:800;color:#0f172a;line-height:1;}
  .cat-sub{font-size:11.5px;color:#64748b;margin-top:6px;}
  .tag{font-size:11px;font-weight:700;padding:2px 8px;border-radius:6px;}
  .mscore{text-align:center;font-size:30px;font-weight:800;font-variant-numeric:tabular-nums;}
  .bar{height:6px;background:#eef2f7;border-radius:4px;overflow:hidden;margin:8px 0;}
  .row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #eef2f7;font-size:13px;}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
  h1,h2,h3{color:#0f172a;}
</style>
""", unsafe_allow_html=True)

GREEN, ORANGE, RED, BLUE, INK, SLATE = "#16a34a", "#f59e0b", "#ef4444", "#2563eb", "#0f172a", "#64748b"
TAGCOL = {"C": ("#2563eb", "#dbeafe"), "A": ("#2563eb", "#dbeafe"), "N": ("#64748b", "#f1f5f9"),
          "S": ("#2563eb", "#dbeafe"), "L": ("#2563eb", "#dbeafe"), "I": ("#2563eb", "#dbeafe"),
          "M": ("#7c3aed", "#ede9fe"), "Quant": ("#0891b2", "#cffafe"), "Math": ("#db2777", "#fce7f3"),
          "Adj": ("#ea580c", "#ffedd5"), "Sentiment": ("#059669", "#d1fae5"),
          "Value": ("#15803d", "#dcfce7"), "Macro": ("#9333ea", "#f3e8ff")}

def scol(s): return GREEN if s >= 70 else ORANGE if s >= 40 else RED
def won(x): return f"{x:,.0f}" if x is not None else "—"

@st.cache_data(ttl=3600, show_spinner=False)
def eval_cached(ticker, period):
    """관련주 비교용: 종목별 평가를 1시간 캐시."""
    return S.evaluate(ticker, period)

EDITABLE = ["price", "roeAnnual", "roeThreshold", "epsGrowthQoQ", "epsAccelQuarters",
            "rsRating", "return12m", "mddPct", "rsi", "adx", "mfi", "hurst",
            "zScoreMeanRev", "shortRatioPct", "dcfFairValue", "targetPrice",
            "atr", "factorAlpha", "volMultiplier",
            "per", "pbr", "dividendYield", "debtRatio", "vix"]

# ---------------------------------------------------------------- 상태 초기화
def do_load():
    """티커 입력칸 Enter 또는 버튼 클릭 시 실행되는 콜백."""
    ticker = (st.session_state.get("tk_input") or "").strip()
    if not ticker:
        return
    base, df, warns = S.build_inputs(
        ticker, period_days=int(st.session_state.get("period", 400)),
        dart_key=(st.session_state.get("dart_key") or "").strip() or None)
    st.session_state.base = base
    st.session_state.df = df
    st.session_state.warns = warns
    st.session_state.fetched = True
    for k in EDITABLE:
        st.session_state[k] = base.get(k, S.DEFAULTS[k])
    # 최근 검색 기록 갱신 (중복 제거, 최신순, 최대 8개) — 세션 동안 유지
    hist = [ticker] + [h for h in st.session_state.get("history", []) if h != ticker]
    st.session_state.history = hist[:8]

if "base" not in st.session_state:
    st.session_state.base = dict(S.DEFAULTS)
    st.session_state.df = None
    st.session_state.warns = []
    st.session_state.fetched = False
    st.session_state.tk_input = "005930"          # 첫 화면 기본: 삼성전자
    st.session_state.period = 400
    st.session_state.cmp_codes = ""
    st.session_state.peer_warn = []
    st.session_state.history = []
    try:
        st.session_state.dart_key = st.secrets.get("DART_API_KEY", "")
    except Exception:
        st.session_state.dart_key = ""
    for k in EDITABLE:
        st.session_state[k] = S.DEFAULTS[k]
    # 첫 진입 시 삼성전자를 자동으로 불러오기
    with st.spinner("삼성전자(005930) 데이터를 불러오는 중..."):
        try:
            do_load()
        except Exception:
            pass

def sample_series(price):
    rng = np.random.default_rng(42)
    n = 130
    v = price * 0.27
    c = []
    for i in range(n):
        drift = 1 + (0.012 + (0.02 if i > 100 else 0)) * (1 if i < 118 else -0.9)
        v = v * drift * (1 + (rng.random() - 0.5) * 0.06)
        c.append(v)
    c = np.array(c) * (price / c[-1])
    s = pd.Series(c)
    df = pd.DataFrame({
        "Close": c, "EMA20": s.ewm(span=20).mean(), "EMA50": s.ewm(span=50).mean(),
        "EMA200": s.ewm(span=200).mean(),
        "Volume": (3 + rng.random(n) * 5) * 1e6 * np.where(np.arange(n) > 100, 1.4, 1),
    })
    df["OBV"] = (np.sign(s.diff()).fillna(0) * df["Volume"]).cumsum()
    df.index = pd.date_range(end=pd.Timestamp.today(), periods=n)
    return df

# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.markdown("### 종목 평가")
    st.text_input("종목코드 또는 종목명  (입력 후 Enter)", key="tk_input", on_change=do_load,
                  help="6자리 코드(예 005930) · 정확한 종목명(예 삼성전자) · 해외 티커(예 AAPL)")

    # 최근 검색 기록 (세션 동안 유지) — 골라서 바로 다시 조회
    _HIST_PLACEHOLDER = "— 최근 검색 기록 —"
    def pick_history():
        v = st.session_state.get("hist_pick")
        if v and v != _HIST_PLACEHOLDER:
            st.session_state.tk_input = v
            do_load()
    if st.session_state.get("history"):
        st.selectbox("📜 최근 검색", [_HIST_PLACEHOLDER] + st.session_state.history,
                     key="hist_pick", on_change=pick_history)

    st.slider("데이터 기간(일)", 200, 700, step=50, key="period")
    st.button("📡 실데이터 불러오기", use_container_width=True, type="primary", on_click=do_load)

    with st.expander("🔑 DART 연동 (정확한 한국 재무)"):
        st.text_input("DART API 키", key="dart_key", type="password",
                      help="opendart.fss.or.kr 에서 무료 발급. 입력 후 다시 불러오면 ROE·부채·순이익을 공시 기준으로 사용해요.")
        st.caption("키를 넣으면 ROE·부채비율·순이익성장이 금감원 공시(분기) 기준으로 계산됩니다.")

    if st.session_state.fetched:
        if st.session_state.df is not None:
            b = st.session_state.base
            st.success(f"✅ {b.get('name','')} 불러옴 · 현재가 {b['price']:,.0f}")
            srcmap = b.get("_src") or {}
            if srcmap:
                with st.expander("📑 데이터 출처 (ROE·PER 등)"):
                    for lab, src in srcmap.items():
                        st.caption(f"· {lab}: {src}")
                    st.caption("‘yfinance(TTM)’이면 분기값이 아니라 최근 12개월 기준이에요.")
        else:
            st.error("⚠️ 가격 데이터를 못 가져왔어요. 티커/인터넷을 확인하거나 아래에서 직접 입력하세요.")

    st.divider()
    st.caption("입력값 수동 보정 (수집 안 되면 직접 입력)")
    with st.expander("가격 · 추세"):
        st.number_input("현재가", key="price", step=1000.0)
        st.number_input("RSI", key="rsi", step=0.1)
        st.number_input("ADX", key="adx", step=1.0)
        st.number_input("허스트", key="hurst", step=0.01, format="%.2f")
        st.number_input("ATR", key="atr", step=1000.0)
        st.number_input("MDD %", key="mddPct", step=1.0)
        st.number_input("평균회귀 Z", key="zScoreMeanRev", step=0.1)
    with st.expander("재무 · 펀더멘털"):
        st.number_input("ROE %", key="roeAnnual", step=1.0)
        st.number_input("ROE 기준 %", key="roeThreshold", step=1.0)
        st.number_input("EPS 성장 % (분기)", key="epsGrowthQoQ", step=1.0)
        st.number_input("EPS 가속 분기수", key="epsAccelQuarters", step=1)
        st.number_input("가치·퀄리티 알파", key="factorAlpha", step=0.1)
    with st.expander("수급 · 목표가"):
        st.number_input("RS 등급", key="rsRating", step=1)
        st.number_input("12M 수익률 %", key="return12m", step=1.0)
        st.number_input("MFI", key="mfi", step=1.0)
        st.number_input("공매도 %", key="shortRatioPct", step=0.1)
        st.number_input("DCF 적정가", key="dcfFairValue", step=1000.0)
        st.number_input("증권사 목표가", key="targetPrice", step=1000.0)
        st.number_input("변동성 배율", key="volMultiplier", step=0.01, format="%.2f")
    with st.expander("가치투자 지표"):
        st.number_input("PER (배)", key="per", step=0.1)
        st.number_input("PBR (배)", key="pbr", step=0.1)
        st.number_input("배당수익률 %", key="dividendYield", step=0.1)
        st.number_input("부채비율 %", key="debtRatio", step=1.0)
    with st.expander("매크로 (시장 환경)"):
        st.number_input("VIX (공포지수)", key="vix", step=0.5)
        st.caption("KOSPI 국면·환율·금리는 자동 수집/기본값을 사용합니다.")
    if st.session_state.warns:
        st.warning("· " + "\n· ".join(st.session_state.warns[:3]))

# ---------------------------------------------------------------- 계산
d = dict(st.session_state.base)
d.update({k: st.session_state[k] for k in EDITABLE})
res = S.compute_all(d)
df = st.session_state.df if st.session_state.df is not None else sample_series(d["price"])

# ---------------------------------------------------------------- 보기 모드 전환
st.markdown(f"<div style='font-size:13px;font-weight:700;color:{SLATE};margin-bottom:2px'>🔀 보기 모드를 선택하세요</div>", unsafe_allow_html=True)
mode = st.radio("보기 모드", ["📊 단일 종목 분석", "⚖️ 동종업계 비교"],
                horizontal=True, label_visibility="collapsed")

# ================================================================ 동종업계 비교 (독립 화면)
if mode == "⚖️ 동종업계 비교":
    st.markdown("## ⚖️ 동종업계 비교")
    st.caption("기준 종목의 동일업종을 자동으로 불러오거나, 코드·정확한 종목명을 콤마로 입력하세요 (최대 8개). 예: 삼성전자, 000660, SK하이닉스")

    def autofill_peers():
        base = (st.session_state.get("cmp_codes", "") or st.session_state.get("tk_input", "")).split(",")[0].strip()
        peers, w = S.fetch_peers(base)
        st.session_state.peer_warn = w
        if peers:
            st.session_state.cmp_codes = base + ", " + ", ".join(peers)

    if not st.session_state.get("cmp_codes"):
        st.session_state.cmp_codes = st.session_state.get("tk_input", "")

    c1, c2 = st.columns([3, 2])
    with c1:
        codes_in = st.text_input("종목 코드들", key="cmp_codes")
    with c2:
        st.write("")
        st.button("🔍 동일업종 자동 채우기", on_click=autofill_peers, use_container_width=True)
    if st.session_state.get("peer_warn"):
        st.caption("· " + " ".join(st.session_state.peer_warn))

    cperiod = st.slider("데이터 기간(일)", 200, 700, int(st.session_state.get("period", 400)), 50, key="cmp_period")
    if st.button("비교하기", type="primary", key="cmp_run"):
        codes = [c.strip() for c in codes_in.split(",")]
        codes = [c for c in dict.fromkeys(codes) if c][:8]
        rows = []
        with st.spinner("관련주 평가 중... (종목당 수 초 걸립니다)"):
            for c in codes:
                try:
                    rows.append(eval_cached(c, int(cperiod)))
                except Exception:
                    pass
        st.session_state.cmp_rows = rows

    rows = st.session_state.get("cmp_rows", [])
    if not rows:
        st.info("종목 코드를 입력하고 '비교하기'를 누르세요.")
    else:
        cmp = pd.DataFrame(rows)
        cmp["종목"] = cmp["name"].astype(str) + " (" + cmp["ticker"].astype(str) + ")"
        table = (cmp.set_index("종목")[["composite", "CAN SLIM", "가치", "모멘텀",
                 "퀄리티", "매크로", "PER", "PBR", "ROE", "배당%", "12M%"]]
                 .rename(columns={"composite": "종합"}))
        st.dataframe(table, use_container_width=True)

        order = cmp.sort_values("composite")
        bar = go.Figure(go.Bar(
            x=order["composite"], y=order["종목"], orientation="h",
            text=order["composite"], textposition="outside",
            marker_color=[GREEN if v >= 60 else ORANGE if v >= 40 else RED for v in order["composite"]]))
        bar.update_layout(title="종합 점수", height=80 + 42 * len(order),
                          margin=dict(l=10, r=30, t=40, b=10), plot_bgcolor="white",
                          paper_bgcolor="white", font=dict(size=11, color=SLATE))
        bar.update_xaxes(range=[0, 100], gridcolor="#f1f5f9")
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

        styles = ["CAN SLIM", "가치", "모멘텀", "퀄리티", "매크로"]
        scolors = ["#2563eb", "#15803d", "#0891b2", "#7c3aed", "#9333ea"]
        gb = go.Figure()
        for sname, sc in zip(styles, scolors):
            gb.add_trace(go.Bar(name=sname, x=cmp["종목"], y=cmp[sname], marker_color=sc))
        gb.update_layout(title="스타일별 점수 비교", barmode="group", height=360,
                         margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                         paper_bgcolor="white", legend=dict(orientation="h", y=1.12),
                         font=dict(size=11, color=SLATE))
        gb.update_yaxes(range=[0, 100], gridcolor="#f1f5f9")
        st.plotly_chart(gb, use_container_width=True, config={"displayModeBar": False})
        st.caption("※ 매크로 점수는 시장 공통이라 종목 간 차이가 거의 없어요. "
                   "자동 수집이 안 된 재무값은 추정/중립값일 수 있습니다.")
    st.stop()   # 비교 모드에서는 아래 단일 종목 대시보드를 렌더링하지 않음

# ================================================================ 단일 종목 분석
# ---------------------------------------------------------------- 헤더
st.markdown(f"## {d.get('name','종목')}  "
            f"<span style='font-size:14px;color:{SLATE}'>{d.get('ticker','')} · {d.get('sector','')}</span>",
            unsafe_allow_html=True)
st.markdown(f"**상승 추세 조정 — RSI {int(d['rsi'])} 유지, 단기 과열 해소 중** "
            f"<span style='color:{ORANGE}'>★★★★☆</span>", unsafe_allow_html=True)

# 카테고리 카드
cols = st.columns(4)
for col, (name, info) in zip(cols, res["cats"].items()):
    col.markdown(
        f"<div class='card'><div class='cat-name'>{name}</div>"
        f"<div class='cat-v'>{info['v']}<span style='font-size:14px;color:{SLATE};font-weight:600'>/5</span></div>"
        f"<div class='cat-sub'>{info['sub']}</div></div>", unsafe_allow_html=True)

# 스타일별 점수 (CAN SLIM · 가치 · 모멘텀 · 퀄리티 · 매크로)
st.write("")
st.markdown("##### 투자 스타일별 점수")
scols = st.columns(len(res["styles"]))
for col, (name, val) in zip(scols, res["styles"].items()):
    col.markdown(
        f"<div class='card' style='text-align:center;padding:12px 8px'>"
        f"<div class='cat-name'>{name}</div>"
        f"<div style='font-size:24px;font-weight:800;color:{scol(val)}'>{val}</div>"
        f"<div class='bar'><div style='width:{max(0,min(100,val))}%;height:100%;background:{scol(val)};border-radius:4px'></div></div>"
        f"</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------- 차트
st.write("")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                    vertical_spacing=0.04, specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
x = df.index
fig.add_trace(go.Scatter(x=x, y=df["EMA200"], name="EMA200", line=dict(color="#a78bfa", width=1.4, dash="dash")), 1, 1)
fig.add_trace(go.Scatter(x=x, y=df["EMA50"], name="EMA50", line=dict(color=ORANGE, width=1.4)), 1, 1)
fig.add_trace(go.Scatter(x=x, y=df["EMA20"], name="EMA20", line=dict(color="#60a5fa", width=1.4)), 1, 1)
fig.add_trace(go.Scatter(x=x, y=df["Close"], name="Close", line=dict(color=INK, width=2.2)), 1, 1)
fig.add_trace(go.Bar(x=x, y=df["Volume"], name="Vol", marker_color="#cbd5e1"), 2, 1, secondary_y=False)
fig.add_trace(go.Scatter(x=x, y=df["OBV"], name="OBV", line=dict(color=ORANGE, width=1.3)), 2, 1, secondary_y=True)
fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                  paper_bgcolor="white", legend=dict(orientation="h", y=1.08, x=0),
                  font=dict(size=11, color=SLATE))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="#f1f5f9", row=1, col=1)
fig.update_yaxes(showgrid=False, row=2, col=1)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# 핵심 관찰
st.markdown(
    f"<div class='card'><b>핵심 관찰</b><div style='font-size:13px;color:#334155;margin-top:6px;line-height:1.6'>"
    f"현재 주가는 52주 고점에서 {res['distHigh']}% 아래, 저점에서 {res['distLow']}% 위에 있습니다. "
    f"추세가 {'매우 강하고' if d['adx']>=40 else '형성 중이며'}, RSI {int(d['rsi'])}로 "
    f"{'과열 구간이며' if d['rsi']>=70 else '중립이며'}, 수급이 "
    f"{'강합니다' if res['cats']['수급']['v']>=4 else '관망입니다'}.</div></div>",
    unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------- 본문
left, right = st.columns([1, 2.3], gap="medium")

with left:
    v = res["composite"]
    st.markdown(
        f"<div class='card'><div class='cat-name'>종합 점수</div>"
        f"<div style='font-size:46px;font-weight:900;color:{ORANGE};line-height:1'>{v}"
        f"<span style='font-size:15px;color:{SLATE}'> 점</span></div>"
        f"<div style='background:#fff7ed;color:#c2410c;border:1px solid #fed7aa;border-radius:8px;"
        f"padding:6px 10px;font-size:12.5px;font-weight:700;margin:8px 0 12px'>⭐ 관찰 LIST — {res['verdict']}</div>"
        f"<div class='row'><span style='color:{SLATE}'>현재가</span><b>{won(d['price'])}</b></div>"
        f"<div class='row'><span style='color:{SLATE}'>등락률</span>"
        f"<b style='color:{GREEN if d['changePct']>=0 else RED}'>{'+' if d['changePct']>0 else ''}{S.r1(d['changePct'])}%</b></div>"
        f"<div class='row'><span style='color:{SLATE}'>DCF 적정가</span><b>{won(d['dcfFairValue'])}</b></div>"
        f"<div class='row'><span style='color:{SLATE}'>증권사 목표가</span>"
        f"<b style='color:{GREEN}'>{won(d['targetPrice'])} (+{res['targetUpside']}%)</b></div>"
        f"<div class='row'><span style='color:{SLATE}'>RSI</span><b>{S.r1(d['rsi'])}</b></div>"
        f"<div class='row'><span style='color:{SLATE}'>Conviction</span>"
        f"<b style='color:{BLUE}'>{'높음' if v>=65 else '중간' if v>=45 else '낮음'}</b></div>"
        f"</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='card' style='background:#fffbeb;border-color:#fde68a;margin-top:12px'>"
        f"<div style='display:flex;justify-content:space-between;font-weight:700;font-size:12.5px;margin-bottom:6px'>"
        f"<span>🟡 진입 타이밍<br><span style='font-weight:500;color:{SLATE};font-size:11px'>"
        f"{'관망 · RSI 과열' if d['rsi']>=70 else '양호'}</span></span>"
        f"<span style='font-size:18px;color:{ORANGE}'>{res['entryTiming']}<span style='font-size:11px;color:{SLATE}'>/100</span></span></div>"
        f"<div class='row'><span style='color:{SLATE}'>매수</span><b style='color:{BLUE}'>{won(res['buy'])}</b></div>"
        f"<div class='row'><span style='color:{SLATE}'>손절</span><b style='color:{RED}'>{won(res['stop'])}</b></div>"
        f"<div class='row'><span style='color:{SLATE}'>1차 익절</span><b style='color:{GREEN}'>{won(res['t1'])}</b></div>"
        f"<div class='row'><span style='color:{SLATE}'>2차 익절</span><b style='color:{GREEN}'>{won(res['t2'])}</b></div>"
        f"<div style='font-size:11px;color:{SLATE};margin-top:8px'>ATR 기반 · R:R {res['rr']}:1</div></div>",
        unsafe_allow_html=True)

    mini = [("EPS 성장", f"{S.r1(d['epsGrowthQoQ'])}%"), ("ROE", f"{S.r1(d['roeAnnual'])}%"),
            ("12M 수익률", f"{S.r1(d['return12m'])}%"), ("RS 등급", int(d['rsRating']))]
    mc = st.columns(2)
    for i, (a, b) in enumerate(mini):
        mc[i % 2].markdown(
            f"<div class='card' style='background:#f8fafc;margin-top:8px;padding:10px'>"
            f"<div style='font-size:16px;font-weight:800'>{b}</div>"
            f"<div style='font-size:10.5px;color:{SLATE}'>{a}</div></div>", unsafe_allow_html=True)

def card_html(m):
    col = RED if (m["canNeg"] and m["value"] < 0) else scol(m["value"])
    tc, tb = TAGCOL.get(m["tag"], TAGCOL["Quant"])
    pct = max(0, min(100, (m["value"] + 100) / 2 if m["canNeg"] else m["value"]))
    return (f"<div class='card'><div style='display:flex;align-items:center;gap:8px;margin-bottom:8px'>"
            f"<span class='tag' style='color:{tc};background:{tb}'>{m['tag']}</span>"
            f"<div><div style='font-size:13.5px;font-weight:700;color:{INK}'>{m['title']}</div>"
            f"<div style='font-size:11px;color:{SLATE}'>{m['sub']}</div></div></div>"
            f"<div class='mscore' style='color:{col}'>{m['value']}</div>"
            f"<div class='bar'><div style='width:{pct}%;height:100%;background:{col}'></div></div>"
            f"<div style='font-size:11.5px;color:{SLATE};line-height:1.5'>{m['text']}</div></div>")

def grid(metrics):
    st.markdown("<div class='grid'>" + "".join(card_html(m) for m in metrics) + "</div>",
                unsafe_allow_html=True)

def status_word(v):
    return ("강함" if v >= 80 else "양호" if v >= 60 else "보통" if v >= 40
            else "주의" if v >= 20 else "약함")

def mval(title):
    return next((m["value"] for m in res["metrics"] if m["title"] == title), 0)

def summary_badge_rows(items):
    """items: [(badge, desc, score), ...] → 점수 색으로 칠한 요약 행들."""
    out = ""
    for badge, desc, v in items:
        c = scol(v)
        out += (f"<div style='display:flex;align-items:center;gap:10px;padding:7px 10px;"
                f"border-left:3px solid {c};margin-bottom:4px;background:#fafcff'>"
                f"<span style='width:20px;height:20px;border-radius:5px;background:{c};color:#fff;"
                f"font-size:11px;font-weight:800;display:grid;place-items:center'>{badge}</span>"
                f"<span style='flex:1;font-size:12.5px;color:#334155'>{desc}</span>"
                f"<b style='color:{c};font-size:12.5px'>{v}</b>"
                f"<span style='font-size:10.5px;color:{SLATE};width:30px;text-align:right'>{status_word(v)}</span>"
                f"</div>")
    return out

def summary_card(heading, items):
    st.markdown(
        f"<div class='card'><div style='text-align:center;font-weight:800;color:{SLATE};"
        f"margin-bottom:10px'>═══ {heading} ═══</div>{summary_badge_rows(items)}</div>",
        unsafe_allow_html=True)

def metric_summary(heading, metrics):
    """METRICS 항목 리스트를 그대로 요약(아이콘=●, 제목, 점수, 상태)."""
    items = [("●", m["title"], m["value"]) for m in metrics]
    summary_card(heading, items)

with right:
    t1, t2, t3, t4, t5, t6 = st.tabs(
        ["CAN SLIM 분석", "가치투자", "모멘텀·기술", "매크로", "재무 지표", "공시·뉴스"])
    with t1:
        cs = [
            ("C", "EPS 가속도", f"분기 순이익 {S.r1(d['epsGrowthQoQ'])}% · {d['epsAccelQuarters']}분기 가속"),
            ("A", "ROE 실적", f"ROE {S.r1(d['roeAnnual'])}% (기준 {S.r1(d['roeThreshold'])}%)"),
            ("N", "신고가·피벗 돌파", f"52주 고점 -{res['distHigh']}% · 피벗 {'돌파' if d['pivotBreak'] else '관찰'}"),
            ("S", "거래량 확인 돌파", f"거래량 평소의 {S.r1(d['volumeRatioVsAvg'])}배"),
            ("L", "주도주 판별", f"상대강도(RS) {int(d['rsRating'])}점"),
            ("I", "기관 수급", f"MFI {int(d['mfi'])} · {'매수 우위' if d['mfi']>=80 else '관망'}"),
            ("M", "시장 방향", f"시장 국면 {d['marketRegime']}"),
        ]
        summary_card("CAN SLIM 원칙 요약",
                     [(badge, desc, mval(title)) for badge, title, desc in cs])
        st.write("")
        grid([m for m in res["metrics"] if m["tag"] in ("C", "A", "N", "S", "L", "I", "M")])
    with t2:
        st.caption(f"가치투자 종합: {res['styles']['가치']}점 — 저평가·재무안정·배당·안전마진 관점")
        metric_summary("가치투자 요약", [m for m in res["metrics"] if m["tag"] == "Value"])
        st.write("")
        grid([m for m in res["metrics"] if m["tag"] == "Value"])
    with t3:
        st.caption(f"모멘텀 종합: {res['styles']['모멘텀']}점 — 추세·상대강도·기술적 신호")
        metric_summary("모멘텀·기술 요약", [m for m in res["metrics"] if m["tag"] in ("Quant", "Math", "Adj")])
        st.write("")
        grid([m for m in res["metrics"] if m["tag"] in ("Quant", "Math", "Adj")])
    with t4:
        st.caption(f"매크로(시장 환경) 종합: {res['styles']['매크로']}점 — 종목 무관 시장 전반")
        metric_summary("매크로 요약", [m for m in res["metrics"] if m["tag"] == "Macro"])
        st.write("")
        grid([m for m in res["metrics"] if m["tag"] == "Macro"])
    with t5:
        fin_titles = ("EPS 가속도", "ROE 실적", "가치·퀄리티 팩터", "DCF 적정가",
                      "PER 밸류", "PBR 밸류", "재무 안정성")
        fin = [m for m in res["metrics"] if m["title"] in fin_titles]
        st.caption(f"퀄리티 종합: {res['styles']['퀄리티']}점 — 수익성·밸류·재무 건전성")
        metric_summary("재무 지표 요약", fin)
        st.write("")
        grid(fin)
    with t6:
        sm = next(m for m in res["metrics"] if m["title"] == "시장 심리 추정")
        st.markdown(f"<div class='card' style='line-height:1.7;font-size:13px;color:{SLATE}'>"
                    f"공시·뉴스 탭은 외부 뉴스 API(네이버 금융, DART)를 연결해야 합니다. "
                    f"현재 심리 추정치는 가격·거래량만으로 산출됐어요 → "
                    f"<b style='color:{scol(sm['value'])}'>{sm['value']}점</b><br>{sm['text']}</div>",
                    unsafe_allow_html=True)

st.caption("⚠️ 이 도구는 투자 판단을 돕는 점수화 프레임워크이며 투자 자문이 아닙니다. "
           "점수 공식은 stock_score.py 의 METRICS 에서 자유롭게 수정·튜닝할 수 있습니다.")
