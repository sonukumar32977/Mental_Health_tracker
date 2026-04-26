"""
app.py - Mental Health Sentiment Tracker
Premium Streamlit Dashboard with glassmorphism UI.
Run with: streamlit run app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from utils.database import init_db, insert_entry, fetch_all_entries
from utils.preprocessor import preprocess
from utils.aggregator import compute_wellness_score, get_alert_level
from models.emotion_model import predict_emotions, get_dominant_emotion, EMOTION_LABELS, load_emotion_model
from models.risk_model import predict_risk, get_risk_display, load_risk_model

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="MindPulse — Wellness Tracker", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
*{font-family:'Plus Jakarta Sans',sans-serif}

/* Force light mode globally */
.stApp { background-color: #f8fafc !important; color: #334155 !important; }
.main .block-container{padding:1.5rem 2rem;max-width:1300px; color: #334155 !important; }

/* Force text colors for markdown and standard text */
p, h1, h2, h3, h4, h5, h6, span, div { color: #334155; }

section[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid #e2e8f0}
section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{background:linear-gradient(135deg,#6366f1,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700}

/* Hero */
.hero{background:linear-gradient(135deg,#e0e7ff 0%,#c7d2fe 100%);border-radius:20px;padding:2.2rem 2.8rem;margin-bottom:1.8rem;position:relative;overflow:hidden;border:1px solid #c7d2fe;box-shadow:0 10px 25px rgba(99,102,241,.05)}
.hero::before{content:'';position:absolute;top:-50%;right:-30%;width:500px;height:500px;background:radial-gradient(circle,rgba(255,255,255,.6) 0%,transparent 70%);border-radius:50%}
.hero h1{font-size:2.4rem;font-weight:800;color:#1e1b4b !important;margin:0 0 .4rem;position:relative;z-index:1;letter-spacing:-.02em}
.hero .accent{color:#4f46e5;background:none;-webkit-text-fill-color:#4f46e5}
.hero p{color:#4338ca !important;font-size:.95rem;margin:0;position:relative;z-index:1;font-weight:500}

/* Metric Cards */
.kpi-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:1.4rem;text-align:center;transition:all .3s cubic-bezier(.4,0,.2,1);position:relative;overflow:hidden;box-shadow:0 4px 6px -1px rgba(0,0,0,.05)}
.kpi-card:hover{border-color:#a5b4fc;transform:translateY(-2px);box-shadow:0 10px 15px -3px rgba(99,102,241,.1)}
.kpi-card::after{content:'';position:absolute;top:0;left:0;right:0;height:4px;background:linear-gradient(90deg,#6366f1,#8b5cf6);border-radius:16px 16px 0 0;opacity:.9}
.kpi-val{font-size:2.2rem;font-weight:800;color:#1e293b;line-height:1.1}
.kpi-lbl{font-size:.7rem;color:#64748b !important;text-transform:uppercase;letter-spacing:.12em;margin-top:.4rem;font-weight:700}
.kpi-icon{font-size:1.6rem;margin-bottom:.3rem}

/* Section Titles */
.sec-title{display:flex;align-items:center;gap:.6rem;font-size:1.15rem;font-weight:700;color:#1e293b !important;margin:2rem 0 1rem;padding-bottom:.6rem;border-bottom:1px solid #e2e8f0}
.sec-badge{background:linear-gradient(135deg,#6366f1,#8b5cf6);padding:.25rem .7rem;border-radius:20px;font-size:.7rem;font-weight:600;color:#ffffff !important;letter-spacing:.05em}

/* Text area */
.stTextArea textarea{background:#ffffff!important;border:1px solid #cbd5e1!important;border-radius:14px!important;color:#334155!important;font-size:1rem!important;line-height:1.7!important;padding:1rem 1.2rem!important;box-shadow:0 1px 2px 0 rgba(0,0,0,.05)!important}
.stTextArea textarea:focus{border-color:#6366f1!important;box-shadow:0 0 0 3px rgba(99,102,241,.15)!important}
.stTextArea label{color:#475569!important;font-weight:600!important}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;color:#ffffff!important;border:none!important;border-radius:12px!important;font-weight:700!important;padding:.7rem 2rem!important;font-size:.9rem!important;letter-spacing:.03em!important;box-shadow:0 4px 14px rgba(99,102,241,.25)!important;transition:all .3s!important}
.stButton>button:hover{box-shadow:0 6px 20px rgba(99,102,241,.4)!important;transform:translateY(-1px)!important}

/* Alerts */
.alert-box{border-radius:14px;padding:1.2rem 1.5rem;margin:1rem 0;position:relative;overflow:hidden; color:#1e293b !important;box-shadow:0 2px 5px rgba(0,0,0,.02)}
.alert-good{background:#ecfdf5;border:1px solid #6ee7b7}
.alert-good::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:#10b981;border-radius:14px 0 0 14px}
.alert-moderate{background:#fffbeb;border:1px solid #fcd34d}
.alert-moderate::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:#f59e0b;border-radius:14px 0 0 14px}
.alert-low{background:#fef2f2;border:1px solid #fca5a5}
.alert-low::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:#ef4444;border-radius:14px 0 0 14px}

/* Result card */
.result-card{background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:1.3rem 1.5rem;box-shadow:0 2px 8px rgba(0,0,0,.03)}
.result-label{font-size:.75rem;color:#64748b !important;text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin-bottom:.4rem}
.result-value{font-size:1.1rem;color:#1e293b !important;font-weight:700}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{gap:8px;background:transparent}
.stTabs [data-baseweb="tab"]{background:#ffffff!important;border:1px solid #e2e8f0!important;border-radius:10px!important;padding:.5rem 1.2rem!important;color:#64748b!important;font-weight:600!important;font-size:.9rem!important;box-shadow:0 1px 3px rgba(0,0,0,.05)}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#6366f1,#8b5cf6)!important;color:#ffffff!important;border-color:transparent!important;box-shadow:0 4px 12px rgba(99,102,241,.25)!important}

/* Expander */
.streamlit-expanderHeader{background:#ffffff!important;border:1px solid #e2e8f0!important;border-radius:12px!important;color:#1e293b!important;font-weight:600!important;box-shadow:0 2px 5px rgba(0,0,0,.02)}

/* Empty state */
.empty-state{text-align:center;padding:3rem 2rem;color:#64748b;background:#ffffff;border:1px dashed #cbd5e1;border-radius:16px}
.empty-state .icon{font-size:3.5rem;margin-bottom:1rem;display:block}
.empty-state h3{color:#1e293b !important;font-weight:600;margin-bottom:.5rem}
.empty-state p{color:#64748b !important;font-size:.9rem}

/* Scrollbar */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}

#MainMenu,footer,header{visibility:hidden}
</style>""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
ECOLORS = {"joy":"#f59e0b","sadness":"#3b82f6","fear":"#8b5cf6","anger":"#ef4444","neutral":"#64748b","surprise":"#f97316","disgust":"#10b981"}
EMOJIS = {"joy":"😊","sadness":"😢","fear":"😨","anger":"😠","neutral":"😐","surprise":"😲","disgust":"🤢"}
BG = "#ffffff"
CARD = "#ffffff"

# ── Chart Helpers ────────────────────────────────────────────────────────────
def make_gauge(score, alert):
    fig = go.Figure(go.Indicator(mode="gauge+number",value=score,
        number={"font":{"size":44,"color":alert["color"],"family":"Plus Jakarta Sans"},"suffix":""},
        title={"text":"WELLNESS","font":{"size":12,"color":"#64748b","family":"Plus Jakarta Sans"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"rgba(99,102,241,.2)","tickfont":{"color":"#64748b","size":10}},
            "bar":{"color":alert["color"],"thickness":.82},
            "bgcolor":"rgba(30,27,75,.3)","borderwidth":0,
            "steps":[{"range":[0,44],"color":"rgba(239,68,68,.06)"},{"range":[45,69],"color":"rgba(245,158,11,.06)"},{"range":[70,100],"color":"rgba(16,185,129,.06)"}],
            "threshold":{"line":{"color":alert["color"],"width":3},"thickness":.85,"value":score}}))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=20,r=20,t=50,b=20),height=240,font=dict(family="Plus Jakarta Sans"))
    return fig

def make_emotion_bars(emotions):
    labels = list(emotions.keys()); scores = [round(v*100,1) for v in emotions.values()]
    colors = [ECOLORS.get(l,"#818cf8") for l in labels]
    fig = go.Figure(go.Bar(x=scores,y=[l.capitalize() for l in labels],orientation="h",
        marker=dict(color=colors,line=dict(width=0),cornerradius=6),
        text=[f"{s:.0f}%" for s in scores],textposition="outside",textfont=dict(color="#475569",size=11,family="Plus Jakarta Sans")))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0,110],showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,tickfont=dict(color="#64748b",size=12,family="Plus Jakarta Sans"),autorange="reversed"),
        margin=dict(l=5,r=30,t=10,b=10),height=260,bargap=.35)
    return fig

def make_trend(entries):
    if not entries: return go.Figure()
    df = pd.DataFrame(entries)[["timestamp","wellness_score"]].sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"],y=df["wellness_score"],mode="lines+markers",
        line=dict(color="#818cf8",width=3,shape="spline"),
        marker=dict(size=8,color="#c084fc",line=dict(color="#818cf8",width=2)),
        fill="tozeroy",fillcolor="rgba(99,102,241,.05)",
        hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Score: %{y:.1f}<extra></extra>"))
    fig.add_hline(y=70,line_dash="dot",line_color="rgba(16,185,129,.4)",annotation_text="Good",annotation_font_color="#10b981",annotation_font_size=10)
    fig.add_hline(y=45,line_dash="dot",line_color="rgba(245,158,11,.4)",annotation_text="Moderate",annotation_font_color="#f59e0b",annotation_font_size=10)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False,tickfont=dict(color="#64748b",size=10)),
        yaxis=dict(range=[0,105],showgrid=True,gridcolor="rgba(99,102,241,.06)",tickfont=dict(color="#64748b",size=10)),
        margin=dict(l=10,r=10,t=15,b=10),height=320,font=dict(family="Plus Jakarta Sans"))
    return fig

def make_emotion_timeline(entries):
    if len(entries) < 2: return go.Figure()
    se = sorted(entries, key=lambda x: x["timestamp"])
    fig = go.Figure()
    for emo in EMOTION_LABELS:
        fig.add_trace(go.Scatter(x=[e["timestamp"] for e in se],y=[e["emotions"].get(emo,0)*100 for e in se],
            mode="lines",name=emo.capitalize(),line=dict(color=ECOLORS.get(emo,"#818cf8"),width=2,shape="spline"),
            hovertemplate=f"<b>{emo.capitalize()}</b><br>%{{x|%b %d}}<br>%{{y:.1f}}%<extra></extra>"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False,tickfont=dict(color="#64748b",size=10)),
        yaxis=dict(showgrid=True,gridcolor="rgba(99,102,241,.06)",tickfont=dict(color="#64748b",size=10),title=dict(text="Confidence %",font=dict(color="#64748b",size=11))),
        margin=dict(l=10,r=10,t=15,b=10),height=320,
        legend=dict(font=dict(color="#94a3b8",size=10),orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
        font=dict(family="Plus Jakarta Sans"))
    return fig

def make_risk_donut(entries):
    if not entries: return go.Figure()
    rc = {"normal":0,"anxiety":0,"depression":0,"high-risk":0}
    for e in entries: rc[e.get("risk_level","normal")] = rc.get(e.get("risk_level","normal"),0)+1
    cols = ["#10b981","#f59e0b","#f97316","#ef4444"]
    fig = go.Figure(go.Pie(labels=[l.title() for l in rc],values=list(rc.values()),hole=.6,
        marker=dict(colors=cols,line=dict(color=BG,width=3)),textfont=dict(color="#475569",size=11),
        hovertemplate="<b>%{label}</b><br>%{value} entries (%{percent})<extra></extra>"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",legend=dict(font=dict(color="#475569",size=11)),
        margin=dict(l=10,r=10,t=15,b=10),height=280,font=dict(family="Plus Jakarta Sans"))
    return fig

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MindPulse")
    st.markdown("<p style='color:#64748b;font-size:.8rem;margin-top:-8px'>AI Wellness Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""<div style='color:#94a3b8;font-size:.85rem;line-height:1.7'>
    Analyze journal entries with AI to:<br>
    😊 Detect 7 core emotions<br>
    🔍 Assess mental health risk<br>
    📈 Track wellness over time<br>
    🚨 Get real-time alerts
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_raw = st.checkbox("Show entry text in history", value=False)
    hist_limit = st.slider("History entries", 5, 50, 10, 5)
    st.markdown("---")
    st.markdown("### 🧪 Models")
    if st.button("⚡ Pre-load AI Models"):
        with st.spinner("Loading models..."): load_emotion_model(); load_risk_model()
        st.success("✅ Ready!")
    st.markdown("---")
    st.markdown("<p style='font-size:.7rem;color:#475569;text-align:center'>⚕️ For personal insight only.<br>Not a substitute for professional care.</p>", unsafe_allow_html=True)

# ── Hero Banner ──────────────────────────────────────────────────────────────
st.markdown("""<div class="hero">
<h1>🧠 Mind<span class="accent">Pulse</span></h1>
<p>AI-powered journal analysis · Emotion detection · Wellness trends · Real-time mental health alerts</p>
</div>""", unsafe_allow_html=True)

# ── Fetch Data ───────────────────────────────────────────────────────────────
all_entries = fetch_all_entries()
recent = all_entries[:hist_limit]
total = len(all_entries)
avg_w = round(sum(e["wellness_score"] for e in all_entries)/total,1) if all_entries else 0
last_s = f'{all_entries[0]["wellness_score"]:.0f}' if all_entries else "—"
last_r = all_entries[0]["risk_level"].title() if all_entries else "—"
dom_e = max(all_entries[0]["emotions"], key=all_entries[0]["emotions"].get).title() if all_entries else "—"

# ── KPI Row ──────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
for col,icon,val,lbl in [(c1,"📝",total,"Total Entries"),(c2,"💎",avg_w,"Avg Score"),(c3,"📊",last_s,"Latest Score"),(c4,"🛡️",last_r,"Latest Risk"),(c5,"😊",dom_e,"Top Emotion")]:
    with col:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">{icon}</div><div class="kpi-val">{val}</div><div class="kpi-lbl">{lbl}</div></div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ── Journal Entry ────────────────────────────────────────────────────────────
st.markdown('<div class="sec-title">✍️ New Journal Entry <span class="sec-badge">AI POWERED</span></div>', unsafe_allow_html=True)

journal_text = st.text_area("How are you feeling today? Write freely...",
    placeholder="e.g., Today was exhausting. I couldn't focus on anything and kept feeling anxious about tomorrow's presentation...",
    height=140, max_chars=3000, key="journal_input")

col_btn, col_cnt = st.columns([1,3])
with col_btn: analyze = st.button("🔍 Analyze Entry", use_container_width=True)
with col_cnt: st.caption(f"✏️ {len(journal_text)}/3000 characters")

if analyze:
    if not journal_text.strip(): st.warning("⚠️ Please write something first.")
    elif len(journal_text.strip()) < 10: st.warning("⚠️ Too short — write at least a sentence.")
    else:
        with st.spinner("🤖 Analyzing with AI models..."):
            processed = preprocess(journal_text)
            emotions = predict_emotions(processed)
            risk_level = predict_risk(processed)
            wellness_score = compute_wellness_score(emotions, risk_level)
            alert = get_alert_level(wellness_score)
            insert_entry(journal_text.strip(), emotions, risk_level, wellness_score)

        st.markdown('<div class="sec-title">📊 Analysis Results <span class="sec-badge">LIVE</span></div>', unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns([1, 1.4, 1])
        with rc1: st.plotly_chart(make_gauge(wellness_score, alert), use_container_width=True, key="gauge")
        with rc2:
            st.markdown("**Emotion Breakdown**")
            st.plotly_chart(make_emotion_bars(emotions), use_container_width=True, key="ebar")
        with rc3:
            dominant = get_dominant_emotion(emotions)
            rm = get_risk_display(risk_level)
            st.markdown(f"""<div class="result-card">
            <div class="result-label">Dominant Emotion</div>
            <div class="result-value"><span style="color:{ECOLORS.get(dominant,'#000')}">●</span> {EMOJIS.get(dominant,'⚪')} {dominant.capitalize()}</div>
            </div><br><div class="result-card">
            <div class="result-label">Risk Level</div>
            <div class="result-value">{rm['emoji']} {rm['label']}</div>
            </div><br><div class="result-card">
            <div class="result-label">Assessment</div>
            <div style="color:#94a3b8;font-size:.85rem;margin-top:.3rem">{rm['description']}</div>
            </div>""", unsafe_allow_html=True)

        css_cls = {"Good":"alert-good","Moderate":"alert-moderate","Low":"alert-low"}.get(alert["level"],"alert-good")
        st.markdown(f'<div class="alert-box {css_cls}"><strong>{alert["emoji"]} {alert["level"]} Wellness</strong> — {alert["message"]}</div>', unsafe_allow_html=True)
        if alert["show_helpline"]: st.error(alert["message"])

        all_entries = fetch_all_entries()
        recent = all_entries[:hist_limit]

# ── Dashboard ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="sec-title">📈 Wellness Dashboard <span class="sec-badge">ANALYTICS</span></div>', unsafe_allow_html=True)

if not all_entries:
    st.markdown("""<div class="empty-state"><span class="icon">📝</span><h3>No entries yet</h3><p>Write your first journal entry above to start tracking your wellness journey.</p></div>""", unsafe_allow_html=True)
else:
    tab1, tab2, tab3 = st.tabs(["📈 Wellness Trend", "😊 Emotion Timeline", "🔍 Risk Distribution"])
    with tab1:
        st.plotly_chart(make_trend(all_entries), use_container_width=True, key="trend")
    with tab2:
        if len(all_entries) < 2: st.info("Add at least 2 entries to see the emotion timeline.")
        else: st.plotly_chart(make_emotion_timeline(all_entries), use_container_width=True, key="etl")
    with tab3:
        tc1, tc2 = st.columns([1.2, 1])
        with tc1: st.plotly_chart(make_risk_donut(all_entries), use_container_width=True, key="donut")
        with tc2:
            st.markdown("**Risk Summary**")
            rc_counts = {}
            for e in all_entries: rc_counts[e.get("risk_level","normal")] = rc_counts.get(e.get("risk_level","normal"),0)+1
            for lvl, cnt in sorted(rc_counts.items(), key=lambda x:x[1], reverse=True):
                m = get_risk_display(lvl); pct = round(cnt/len(all_entries)*100,1)
                st.markdown(f"{m['emoji']} **{m['label']}** — {cnt} entries ({pct}%)")

# ── History ──────────────────────────────────────────────────────────────────
st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="sec-title">📋 Recent Entries <span class="sec-badge">HISTORY</span></div>', unsafe_allow_html=True)

if not all_entries:
    st.info("No entries yet.")
else:
    for entry in recent:
        am = get_alert_level(entry["wellness_score"]); rm = get_risk_display(entry["risk_level"])
        de = max(entry["emotions"], key=entry["emotions"].get)
        with st.expander(f'{am["emoji"]} {entry["timestamp"]} — Score: {entry["wellness_score"]:.0f} · {rm["emoji"]} {rm["label"]} · {de.capitalize()}', expanded=False):
            hc1, hc2 = st.columns([1, 1.4])
            with hc1:
                st.markdown(f'**Dominant Emotion:** `{de.capitalize()}`')
                st.markdown(f'**Risk Level:** {rm["emoji"]} `{rm["label"]}`')
                st.markdown(f'**Wellness Score:** `{entry["wellness_score"]:.1f} / 100`')
                if show_raw: st.markdown(f'**Entry:** {entry["entry_text"][:300]}{"..." if len(entry["entry_text"])>300 else ""}')
            with hc2:
                st.plotly_chart(make_emotion_bars(entry["emotions"]), use_container_width=True, key=f"h_{entry['id']}")
