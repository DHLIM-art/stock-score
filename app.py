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
          "Adj": ("#ea580c", "#ffedd5"), "Sentiment": ("#059669", "#d1fae5")}

def scol(s): return GREEN if s >= 70 else ORANGE if s >= 40 else RED
def won(x): return f"{x:,.0f}" if x is not None else "—"

EDITABLE = ["price", "roeAnnual", "roeThreshold", "epsGrowthQoQ", "epsAccelQuarters",
            "rsRating", "return12m", "mddPct", "rsi", "adx", "mfi", "hurst",
            "zScoreMeanRev", "shortRatioPct", "dcfFairValue", "targetPrice",
            "atr", "factorAlpha", "volMultiplier"]

# ---------------------------------------------------------------- 상태 초기화
if "base" not in st.session_state:
    st.session_state.base = dict(S.DEFAULTS)
    st.session_state.df = None
    st.session_state.warns = []
    for k in EDITABLE:
        st.session_state[k] = S.DEFAULTS[k]

def load(ticker, period):
    base, df, warns = S.build_inputs(ticker, period_days=period)
    st.session_state.base = base
    st.session_state.df = df
    st.session_state.warns = warns
    for k in EDITABLE:
        st.session_state[k] = base.get(k, S.DEFAULTS[k])

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
    tk = st.text_input("종목코드 / 티커", value=st.session_state.base.get("ticker", "000660"),
                       help="국내: 6자리 코드(예 000660) · 해외: 티커(예 AAPL)")
    period = st.slider("데이터 기간(일)", 200, 700, 400, 50)
    if st.button("📡 실데이터 불러오기", use_container_width=True, type="primary"):
        with st.spinner("데이터 수집 중..."):
            load(tk.strip(), period)
        st.rerun()

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
    if st.session_state.warns:
        st.warning("· " + "\n· ".join(st.session_state.warns[:3]))

# ---------------------------------------------------------------- 계산
d = dict(st.session_state.base)
d.update({k: st.session_state[k] for k in EDITABLE})
res = S.compute_all(d)
df = st.session_state.df if st.session_state.df is not None else sample_series(d["price"])

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

with right:
    t1, t2, t3, t4 = st.tabs(["CAN SLIM 분석", "기술 지표", "재무 지표", "공시·뉴스"])
    with t1:
        canslim = [
            ("C", f"분기 실적이 {d['epsAccelQuarters']}분기 연속 가속 성장 중이에요", GREEN),
            ("A", f"자기자본이익률 {S.r1(d['roeAnnual'])}%가 기준({S.r1(d['roeThreshold'])}%)을 통과했어요", GREEN),
            ("N", f"52주 최고가에서 {res['distHigh']}% 이내에 위치했어요", GREEN),
            ("N", f"컵앤핸들 패턴의 피벗을 {'돌파했어요' if d['pivotBreak'] else '관찰 중이에요'}", GREEN if d['pivotBreak'] else ORANGE),
            ("S", f"거래량이 평소의 {S.r1(d['volumeRatioVsAvg'])}배 수준이에요", GREEN),
            ("L", f"상대강도 {int(d['rsRating'])}점으로 시장 주도주에요", GREEN),
            ("I", f"기관 자금 흐름은 '{'매수' if d['mfi']>=80 else '관망'}'이에요", SLATE),
        ]
        rows = "".join(
            f"<div style='display:flex;align-items:center;gap:10px;padding:7px 10px;"
            f"border-left:3px solid {c};margin-bottom:4px;background:#fafcff'>"
            f"<span style='width:18px;height:18px;border-radius:5px;background:#f1f5f9;color:{SLATE};"
            f"font-size:11px;font-weight:800;display:grid;place-items:center'>{t}</span>"
            f"<span style='font-size:12.5px;color:#334155'>{txt}</span></div>" for t, txt, c in canslim)
        st.markdown(f"<div class='card'><div style='text-align:center;font-weight:800;color:{SLATE};"
                    f"margin-bottom:10px'>═══ CAN SLIM 원칙 요약 ═══</div>{rows}</div>", unsafe_allow_html=True)
        st.write("")
        grid(res["metrics"])
    with t2:
        grid([m for m in res["metrics"] if m["tag"] in ("Quant", "Math")])
    with t3:
        grid([m for m in res["metrics"] if m["title"] in
              ("EPS 가속도", "연간 ROE 실적", "가치·퀄리티 팩터", "DCF 적정가")])
    with t4:
        sm = next(m for m in res["metrics"] if m["title"] == "시장 심리 추정")
        st.markdown(f"<div class='card' style='line-height:1.7;font-size:13px;color:{SLATE}'>"
                    f"공시·뉴스 탭은 외부 뉴스 API(네이버 금융, DART)를 연결해야 합니다. "
                    f"현재 심리 추정치는 가격·거래량만으로 산출됐어요 → "
                    f"<b style='color:{scol(sm['value'])}'>{sm['value']}점</b><br>{sm['text']}</div>",
                    unsafe_allow_html=True)

st.caption("⚠️ 이 도구는 투자 판단을 돕는 점수화 프레임워크이며 투자 자문이 아닙니다. "
           "점수 공식은 stock_score.py 의 METRICS 에서 자유롭게 수정·튜닝할 수 있습니다.")
