# ============================================================
# Customer Segmentation & Churn Pattern Analytics
# Streamlit Dashboard — European Banking
# Project: Unified Mentor Internship
# ============================================================
# Run with: streamlit run streamlit_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #a0a8c8 !important;
        font-size: 13px !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #e0e6ff;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #3a3f5c;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e, #0f1117);
        border-right: 1px solid #2a2d40;
    }

    /* Title */
    .dashboard-title {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .dashboard-subtitle {
        font-size: 15px;
        color: #7a85aa;
        margin-top: -8px;
        margin-bottom: 24px;
    }
    
    /* KPI Risk Badge */
    .risk-high   { background:#ff4b4b22; color:#ff4b4b; border:1px solid #ff4b4b55; border-radius:6px; padding:3px 10px; font-weight:600; }
    .risk-medium { background:#ffa62b22; color:#ffa62b; border:1px solid #ffa62b55; border-radius:6px; padding:3px 10px; font-weight:600; }
    .risk-low    { background:#21c35422; color:#21c354; border:1px solid #21c35455; border-radius:6px; padding:3px 10px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")
    df = df.drop(columns=['CustomerId', 'Surname', 'Year'], errors='ignore')

    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,30,45,60,100],
                            labels=['<30','30-45','46-60','60+'])
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[-1,2,5,10],
                               labels=['New (0-2)','Mid (3-5)','Long (6+)'])
    df['CreditBand'] = pd.cut(df['CreditScore'], bins=[0,580,669,850],
                              labels=['Low','Medium','High'])
    df['BalanceSegment'] = pd.cut(df['Balance'], bins=[-1,0,50000,1e9],
                                  labels=['Zero-balance','Low-balance','High-balance'])
    df['ChurnLabel'] = df['Exited'].map({0:'Retained', 1:'Churned'})
    return df

df = load_data()

# ── PLOTLY THEME ─────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c8d0f0', family='Inter, sans-serif'),
    xaxis=dict(gridcolor='#2a2d40', linecolor='#3a3f5c'),
    yaxis=dict(gridcolor='#2a2d40', linecolor='#3a3f5c'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#3a3f5c')
)
COLORS = {
    'churn':    '#FF4B4B',
    'retain':   '#21C354',
    'blue':     '#4B7BFF',
    'orange':   '#FF9F40',
    'purple':   '#9B59B6',
    'teal':     '#1ABC9C',
    'geo':      ['#FF4B4B','#4B7BFF','#21C354'],
}

# ─────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Churn Analytics")
    st.markdown("---")

    st.markdown("### 🔍 Filters")

    geo_options = ['All'] + sorted(df['Geography'].unique().tolist())
    selected_geo = st.selectbox("🌍 Geography", geo_options)

    gender_options = ['All'] + sorted(df['Gender'].unique().tolist())
    selected_gender = st.selectbox("👤 Gender", gender_options)

    age_options = ['All'] + list(df['AgeGroup'].cat.categories)
    selected_age = st.selectbox("📅 Age Group", age_options)

    credit_options = ['All'] + list(df['CreditBand'].cat.categories)
    selected_credit = st.selectbox("💳 Credit Band", credit_options)

    balance_options = ['All'] + list(df['BalanceSegment'].cat.categories)
    selected_balance = st.selectbox("💰 Balance Segment", balance_options)

    st.markdown("---")
    st.markdown("### 📊 Dashboard Info")
    st.info("**Project:** Customer Segmentation & Churn Pattern Analytics\n\n**Data:** European Banking Dataset\n\n**Source:** Unified Mentor Internship")

# ── APPLY FILTERS ────────────────────────────────────────────
dff = df.copy()
if selected_geo    != 'All': dff = dff[dff['Geography']      == selected_geo]
if selected_gender != 'All': dff = dff[dff['Gender']          == selected_gender]
if selected_age    != 'All': dff = dff[dff['AgeGroup']        == selected_age]
if selected_credit != 'All': dff = dff[dff['CreditBand']      == selected_credit]
if selected_balance!= 'All': dff = dff[dff['BalanceSegment']  == selected_balance]

# ── HEADER ───────────────────────────────────────────────────
st.markdown('<div class="dashboard-title">🏦 European Bank Churn Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Customer Segmentation & Churn Pattern Dashboard — Unified Mentor Internship Project</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ROW 1 — KPI METRICS
# ─────────────────────────────────────────────────────────────
total      = len(dff)
churned    = dff['Exited'].sum()
churn_rate = churned / total * 100 if total > 0 else 0
hv         = dff[dff['BalanceSegment'] == 'High-balance']
hv_churn   = hv['Exited'].mean() * 100 if len(hv) > 0 else 0
rev_risk   = hv[hv['Exited'] == 1]['Balance'].sum()
inactive_churn = dff[dff['IsActiveMember'] == 0]['Exited'].mean() * 100 if len(dff[dff['IsActiveMember']==0]) > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("👥 Total Customers",      f"{total:,}")
col2.metric("🔴 Churned Customers",    f"{churned:,}")
col3.metric("📉 Overall Churn Rate",   f"{churn_rate:.1f}%",  delta=f"{churn_rate - 20:.1f}% vs 20% bench", delta_color="inverse")
col4.metric("💎 High-Value Churn",     f"{hv_churn:.1f}%")
col5.metric("💸 Revenue at Risk",      f"€{rev_risk/1e6:.1f}M")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ROW 2 — OVERALL CHURN + GEOGRAPHY
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📍 Overall Churn & Geographic Analysis</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1.2, 1.5])

# Donut chart
with col1:
    fig_donut = go.Figure(go.Pie(
        labels=['Churned', 'Retained'],
        values=[churned, total - churned],
        hole=0.65,
        marker_colors=[COLORS['churn'], COLORS['retain']],
        textinfo='percent',
        hovertemplate='%{label}: %{value:,}<extra></extra>'
    ))
    fig_donut.add_annotation(text=f"<b>{churn_rate:.1f}%</b><br>Churn Rate",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=18, color='white'))
    fig_donut.update_layout(title="Churn Distribution", showlegend=True,
                             height=320, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_donut, use_container_width=True)

# Geography churn rate
with col2:
    geo_churn = dff.groupby('Geography')['Exited'].agg(['sum','count']).reset_index()
    geo_churn.columns = ['Geography','Churned','Total']
    geo_churn['ChurnRate'] = geo_churn['Churned'] / geo_churn['Total'] * 100
    fig_geo = px.bar(geo_churn.sort_values('ChurnRate', ascending=True),
                     x='ChurnRate', y='Geography', orientation='h',
                     color='ChurnRate', color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'],
                     text=geo_churn.sort_values('ChurnRate', ascending=True)['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                     title='Churn Rate by Country')
    fig_geo.update_traces(textposition='outside')
    fig_geo.update_layout(height=320, margin=dict(t=40,b=0,l=0,r=10),
                           coloraxis_showscale=False, **PLOT_THEME)
    st.plotly_chart(fig_geo, use_container_width=True)

# Geography choropleth-style grouped
with col3:
    geo_detail = dff.groupby('Geography').agg(
        Total=('Exited','count'),
        Churned=('Exited','sum'),
        AvgBalance=('Balance','mean'),
        AvgAge=('Age','mean')
    ).reset_index()
    geo_detail['ChurnRate'] = (geo_detail['Churned'] / geo_detail['Total'] * 100).round(1)
    fig_geo2 = go.Figure()
    fig_geo2.add_trace(go.Bar(name='Retained', x=geo_detail['Geography'],
                               y=geo_detail['Total'] - geo_detail['Churned'],
                               marker_color=COLORS['retain'], opacity=0.85))
    fig_geo2.add_trace(go.Bar(name='Churned', x=geo_detail['Geography'],
                               y=geo_detail['Churned'],
                               marker_color=COLORS['churn'], opacity=0.85))
    fig_geo2.update_layout(barmode='stack', title='Customer Volume by Country',
                            height=320, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_geo2, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ROW 3 — AGE & TENURE
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📅 Age & Tenure Churn Comparison</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age_data = dff.groupby('AgeGroup', observed=True)['Exited'].mean().reset_index()
    age_data.columns = ['AgeGroup','ChurnRate']
    age_data['ChurnRate'] = age_data['ChurnRate'] * 100
    fig_age = px.bar(age_data, x='AgeGroup', y='ChurnRate',
                     color='ChurnRate',
                     color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'],
                     text=age_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                     title='Churn Rate by Age Group')
    fig_age.update_traces(textposition='outside')
    fig_age.update_layout(height=350, coloraxis_showscale=False,
                           margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    tenure_data = dff.groupby('TenureGroup', observed=True)['Exited'].mean().reset_index()
    tenure_data.columns = ['TenureGroup','ChurnRate']
    tenure_data['ChurnRate'] = tenure_data['ChurnRate'] * 100
    fig_ten = px.bar(tenure_data, x='TenureGroup', y='ChurnRate',
                     color='ChurnRate',
                     color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'],
                     text=tenure_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                     title='Churn Rate by Tenure Group')
    fig_ten.update_traces(textposition='outside')
    fig_ten.update_layout(height=350, coloraxis_showscale=False,
                           margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_ten, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ROW 4 — HIGH-VALUE CUSTOMER CHURN EXPLORER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💎 High-Value Customer Churn Explorer</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    sample = dff.sample(min(3000, len(dff)), random_state=42)
    fig_scatter = px.scatter(
        sample, x='Balance', y='EstimatedSalary',
        color='ChurnLabel',
        color_discrete_map={'Churned': COLORS['churn'], 'Retained': COLORS['retain']},
        opacity=0.5, size_max=6,
        hover_data=['Age', 'Geography', 'CreditScore'],
        title='Balance vs Estimated Salary (Churn Highlighted)',
        labels={'Balance': 'Account Balance (€)', 'EstimatedSalary': 'Estimated Salary (€)'}
    )
    fig_scatter.update_layout(height=380, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    bal_data = dff.groupby('BalanceSegment', observed=True)['Exited'].mean().reset_index()
    bal_data.columns = ['Segment','ChurnRate']
    bal_data['ChurnRate'] = bal_data['ChurnRate'] * 100
    fig_bal = px.bar(bal_data, x='Segment', y='ChurnRate',
                     color='ChurnRate',
                     color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'],
                     text=bal_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                     title='Churn Rate by Balance Segment')
    fig_bal.update_traces(textposition='outside')
    fig_bal.update_layout(height=380, coloraxis_showscale=False,
                           margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_bal, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ROW 5 — DEMOGRAPHIC ANALYSIS
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">👥 Comparative Demographic Analysis</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    gender_data = dff.groupby('Gender')['Exited'].mean().reset_index()
    gender_data.columns = ['Gender','ChurnRate']
    gender_data['ChurnRate'] = gender_data['ChurnRate'] * 100
    fig_gender = px.bar(gender_data, x='Gender', y='ChurnRate',
                        color='Gender',
                        color_discrete_map={'Female': COLORS['purple'], 'Male': COLORS['teal']},
                        text=gender_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                        title='Churn Rate by Gender')
    fig_gender.update_traces(textposition='outside')
    fig_gender.update_layout(height=320, showlegend=False,
                              margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_gender, use_container_width=True)

with col2:
    credit_data = dff.groupby('CreditBand', observed=True)['Exited'].mean().reset_index()
    credit_data.columns = ['CreditBand','ChurnRate']
    credit_data['ChurnRate'] = credit_data['ChurnRate'] * 100
    fig_credit = px.bar(credit_data, x='CreditBand', y='ChurnRate',
                        color='ChurnRate',
                        color_continuous_scale=['#FF4B4B','#FFA62B','#21C354'],
                        text=credit_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                        title='Churn Rate by Credit Score Band')
    fig_credit.update_traces(textposition='outside')
    fig_credit.update_layout(height=320, coloraxis_showscale=False,
                              margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_credit, use_container_width=True)

with col3:
    geo_age = dff.groupby(['Geography', 'AgeGroup'], observed=True)['Exited'].mean().reset_index()
    geo_age.columns = ['Geography','AgeGroup','ChurnRate']
    geo_age['ChurnRate'] = geo_age['ChurnRate'] * 100
    fig_heatmap = px.density_heatmap(
        geo_age, x='AgeGroup', y='Geography', z='ChurnRate',
        color_continuous_scale='RdYlGn_r',
        title='Geography × Age Churn Heatmap (%)'
    )
    fig_heatmap.update_layout(height=320, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ROW 6 — ENGAGEMENT DROP INDICATOR
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Engagement Drop Indicator</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    prod_data = dff.groupby('NumOfProducts')['Exited'].mean().reset_index()
    prod_data.columns = ['NumProducts','ChurnRate']
    prod_data['ChurnRate'] = prod_data['ChurnRate'] * 100
    fig_prod = px.bar(prod_data, x='NumProducts', y='ChurnRate',
                      color='ChurnRate',
                      color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'],
                      text=prod_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                      title='Churn Rate by Number of Products',
                      labels={'NumProducts': 'Number of Products', 'ChurnRate': 'Churn Rate (%)'})
    fig_prod.update_traces(textposition='outside')
    fig_prod.update_layout(height=350, coloraxis_showscale=False,
                            margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_prod, use_container_width=True)

with col2:
    active_data = dff.groupby('IsActiveMember')['Exited'].mean().reset_index()
    active_data.columns = ['IsActive','ChurnRate']
    active_data['ChurnRate'] = active_data['ChurnRate'] * 100
    active_data['Status'] = active_data['IsActive'].map({0:'Inactive', 1:'Active'})
    fig_active = px.bar(active_data, x='Status', y='ChurnRate',
                        color='Status',
                        color_discrete_map={'Inactive': COLORS['churn'], 'Active': COLORS['retain']},
                        text=active_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                        title='Churn Rate: Active vs Inactive Members')
    fig_active.update_traces(textposition='outside')
    fig_active.update_layout(height=350, showlegend=False,
                              margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig_active, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# ROW 7 — SEGMENT DRILLDOWN TABLE
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔎 Segment Drill-Down Table</div>', unsafe_allow_html=True)

seg_table = dff.groupby(['Geography', 'AgeGroup', 'Gender'], observed=True).agg(
    Total=('Exited', 'count'),
    Churned=('Exited', 'sum'),
    AvgBalance=('Balance', 'mean'),
    AvgCreditScore=('CreditScore', 'mean')
).reset_index()
seg_table['ChurnRate%'] = (seg_table['Churned'] / seg_table['Total'] * 100).round(1)
seg_table['AvgBalance'] = seg_table['AvgBalance'].round(0).astype(int)
seg_table['AvgCreditScore'] = seg_table['AvgCreditScore'].round(0).astype(int)
seg_table = seg_table.sort_values('ChurnRate%', ascending=False)

def color_churn(val):
    if val >= 40:   return 'background-color: #ff4b4b33; color: #ff4b4b'
    elif val >= 25: return 'background-color: #ffa62b33; color: #ffa62b'
    else:           return 'background-color: #21c35433; color: #21c354'

st.dataframe(
    seg_table.style.applymap(color_churn, subset=['ChurnRate%']),
    use_container_width=True,
    height=400
)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#555; font-size:13px;'>"
    "🏦 Customer Segmentation & Churn Pattern Analytics in European Banking &nbsp;|&nbsp; "
    "Unified Mentor Internship Project &nbsp;|&nbsp; Dataset: European Central Bank"
    "</div>",
    unsafe_allow_html=True
)
