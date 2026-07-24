import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="European Bank Churn Analytics", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    div[data-testid="metric-container"] { background: linear-gradient(135deg, #1e2130, #252840); border: 1px solid #3a3f5c; border-radius: 12px; padding: 16px 20px; }
    div[data-testid="metric-container"] label { color: #a0a8c8 !important; font-size: 13px !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 28px !important; font-weight: 700 !important; }
    .section-header { font-size: 20px; font-weight: 700; color: #e0e6ff; margin: 24px 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #3a3f5c; }
    .insight-box { background: #1e2130; border-left: 4px solid #4B7BFF; border-radius: 6px; padding: 10px 16px; margin: 8px 0 16px 0; color: #c8d0f0; font-size: 14px; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1d2e, #0f1117); }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("European_Bank.csv")
    df = df.drop(columns=['CustomerId', 'Surname', 'Year'], errors='ignore')
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,30,45,60,100], labels=['<30','30-45','46-60','60+'])
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[-1,2,5,10], labels=['New (0-2)','Mid (3-5)','Long (6+)'])
    df['CreditBand'] = pd.cut(df['CreditScore'], bins=[0,580,669,850], labels=['Low','Medium','High'])
    df['BalanceSegment'] = pd.cut(df['Balance'], bins=[-1,0,50000,1e9], labels=['Zero-balance','Low-balance','High-balance'])
    df['ChurnLabel'] = df['Exited'].map({0:'Retained', 1:'Churned'})
    return df

df = load_data()

PLOT_THEME = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#c8d0f0'), xaxis=dict(gridcolor='#2a2d40'), yaxis=dict(gridcolor='#2a2d40'), legend=dict(bgcolor='rgba(0,0,0,0)'))
COLORS = {'churn':'#FF4B4B','retain':'#21C354','purple':'#9B59B6','teal':'#1ABC9C'}

with st.sidebar:
    st.markdown("## 🏦 Churn Analytics")
    st.markdown("---")
    selected_geo     = st.selectbox("🌍 Geography",       ['All'] + sorted(df['Geography'].unique().tolist()))
    selected_gender  = st.selectbox("👤 Gender",          ['All'] + sorted(df['Gender'].unique().tolist()))
    selected_age     = st.selectbox("📅 Age Group",       ['All'] + list(df['AgeGroup'].cat.categories))
    selected_credit  = st.selectbox("💳 Credit Band",     ['All'] + list(df['CreditBand'].cat.categories))
    selected_balance = st.selectbox("💰 Balance Segment", ['All'] + list(df['BalanceSegment'].cat.categories))
    st.markdown("---")
    st.info("**Project:** Customer Segmentation & Churn Analytics\n\n**Source:** Unified Mentor Internship")

dff = df.copy()
if selected_geo     != 'All': dff = dff[dff['Geography']      == selected_geo]
if selected_gender  != 'All': dff = dff[dff['Gender']          == selected_gender]
if selected_age     != 'All': dff = dff[dff['AgeGroup']        == selected_age]
if selected_credit  != 'All': dff = dff[dff['CreditBand']      == selected_credit]
if selected_balance != 'All': dff = dff[dff['BalanceSegment']  == selected_balance]

st.markdown("# 🏦 European Bank Churn Analytics")
st.markdown("Customer Segmentation & Churn Pattern Dashboard — Unified Mentor Internship Project")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🌍 Geography", "👥 Demographics", "💎 High-Value", "📈 Engagement"])

with tab1:
    total = len(dff)
    churned = int(dff['Exited'].sum())
    churn_rate = churned / total * 100 if total > 0 else 0
    hv = dff[dff['BalanceSegment'] == 'High-balance']
    hv_churn = hv['Exited'].mean() * 100 if len(hv) > 0 else 0
    rev_risk = hv[hv['Exited'] == 1]['Balance'].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👥 Total Customers", f"{total:,}")
    col2.metric("🔴 Churned", f"{churned:,}")
    col3.metric("📉 Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_rate-20:.1f}% vs 20% bench", delta_color="inverse")
    col4.metric("💎 High-Value Churn", f"{hv_churn:.1f}%")
    col5.metric("💸 Revenue at Risk", f"€{rev_risk/1e6:.1f}M")

    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Overall churn rate is 20.4% — Germany drives the highest risk at 32.4%, nearly double France and Spain. The 46–60 age group churns at 56%+.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Pie(labels=['Churned','Retained'], values=[churned, total-churned], hole=0.65, marker_colors=[COLORS['churn'], COLORS['retain']], textinfo='percent'))
        fig.add_annotation(text=f"<b>{churn_rate:.1f}%</b><br>Churn", x=0.5, y=0.5, showarrow=False, font=dict(size=18, color='white'))
        fig.update_layout(title="Churn Distribution", height=320, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        geo = dff.groupby('Geography')['Exited'].agg(['sum','count']).reset_index()
        geo.columns = ['Geography','Churned','Total']
        geo['ChurnRate'] = geo['Churned'] / geo['Total'] * 100
        fig2 = px.bar(geo, x='Geography', y='ChurnRate', color='ChurnRate', color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'], text=geo['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn Rate by Country')
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Germany\'s 32.4% churn rate is nearly double France and Spain. Country-specific retention strategies are essential.</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    for col, country in zip([col1,col2,col3], ['Germany','France','Spain']):
        col.metric(f"🏳️ {country}", f"{df[df['Geography']==country]['Exited'].mean()*100:.1f}%")
    col1, col2 = st.columns(2)
    with col1:
        gd = dff.groupby('Geography').agg(Total=('Exited','count'), Churned=('Exited','sum')).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Retained', x=gd['Geography'], y=gd['Total']-gd['Churned'], marker_color=COLORS['retain']))
        fig.add_trace(go.Bar(name='Churned', x=gd['Geography'], y=gd['Churned'], marker_color=COLORS['churn']))
        fig.update_layout(barmode='stack', title='Volume by Country', height=350, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        ga = dff.groupby(['Geography','AgeGroup'], observed=True)['Exited'].mean().reset_index()
        ga.columns = ['Geography','AgeGroup','ChurnRate']
        ga['ChurnRate'] = ga['ChurnRate'] * 100
        fig2 = px.density_heatmap(ga, x='AgeGroup', y='Geography', z='ChurnRate', color_continuous_scale='RdYlGn_r', title='Geography × Age Heatmap (%)')
        fig2.update_layout(height=350, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Female customers churn 52% more than males (25.1% vs 16.5%). The 46–60 age group churns at 56%+ — more than 2.7x the overall average.</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        gd = dff.groupby('Gender', observed=True)['Exited'].mean().reset_index()
        gd['ChurnRate'] = gd['Exited'] * 100
        fig = px.bar(gd, x='Gender', y='ChurnRate', color='Gender', color_discrete_map={'Female':COLORS['purple'],'Male':COLORS['teal']}, text=gd['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn by Gender')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=320, showlegend=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        ad = dff.groupby('AgeGroup', observed=True)['Exited'].mean().reset_index()
        ad['ChurnRate'] = ad['Exited'] * 100
        fig2 = px.bar(ad, x='AgeGroup', y='ChurnRate', color='ChurnRate', color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'], text=ad['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn by Age Group')
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        cd = dff.groupby('CreditBand', observed=True)['Exited'].mean().reset_index()
        cd['ChurnRate'] = cd['Exited'] * 100
        fig3 = px.bar(cd, x='CreditBand', y='ChurnRate', color='ChurnRate', color_continuous_scale=['#FF4B4B','#FFA62B','#21C354'], text=cd['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn by Credit Band')
        fig3.update_traces(textposition='outside')
        fig3.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig3, use_container_width=True)
    td = dff.groupby('TenureGroup', observed=True)['Exited'].mean().reset_index()
    td['ChurnRate'] = td['Exited'] * 100
    fig4 = px.bar(td, x='TenureGroup', y='ChurnRate', color='ChurnRate', color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'], text=td['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn by Tenure Group')
    fig4.update_traces(textposition='outside')
    fig4.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> High-balance customers (>€50K) churn at 22.6%, placing ~€91.5M in deposits at risk.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])
    with col1:
        sample = dff.sample(min(3000, len(dff)), random_state=42)
        fig = px.scatter(sample, x='Balance', y='EstimatedSalary', color='ChurnLabel', color_discrete_map={'Churned':COLORS['churn'],'Retained':COLORS['retain']}, opacity=0.5, hover_data=['Age','Geography','CreditScore'], title='Balance vs Salary', labels={'Balance':'Balance (€)','EstimatedSalary':'Salary (€)'})
        fig.update_layout(height=380, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        bd = dff.groupby('BalanceSegment', observed=True)['Exited'].mean().reset_index()
        bd['ChurnRate'] = bd['Exited'] * 100
        fig2 = px.bar(bd, x='BalanceSegment', y='ChurnRate', color='ChurnRate', color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'], text=bd['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn by Balance Segment')
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)

with tab5:
    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Inactive members churn at 26.9% vs 14.3% for active. 2-product customers have lowest churn (7.6%), while 3–4 product holders churn at 82–100%.</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        act = dff.groupby('IsActiveMember')['Exited'].mean().reset_index()
        act['Status'] = act['IsActiveMember'].map({0:'Inactive',1:'Active'})
        act['ChurnRate'] = act['Exited'] * 100
        fig = px.bar(act, x='Status', y='ChurnRate', color='Status', color_discrete_map={'Inactive':COLORS['churn'],'Active':COLORS['retain']}, text=act['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Active vs Inactive Churn')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, showlegend=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        pd_ = dff.groupby('NumOfProducts')['Exited'].mean().reset_index()
        pd_['ChurnRate'] = pd_['Exited'] * 100
        fig2 = px.bar(pd_, x='NumOfProducts', y='ChurnRate', color='ChurnRate', color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'], text=pd_['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn by Number of Products', labels={'NumOfProducts':'Products'})
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=350, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">🔎 Segment Drill-Down Table</div>', unsafe_allow_html=True)
    seg = dff.groupby(['Geography','AgeGroup','Gender'], observed=True).agg(Total=('Exited','count'), Churned=('Exited','sum'), AvgBalance=('Balance','mean'), AvgCreditScore=('CreditScore','mean')).reset_index()
    seg['ChurnRate%'] = (seg['Churned'] / seg['Total'] * 100).round(1)
    seg['AvgBalance'] = seg['AvgBalance'].round(0).astype(int)
    seg['AvgCreditScore'] = seg['AvgCreditScore'].round(0).astype(int)
    st.dataframe(seg.sort_values('ChurnRate%', ascending=False), use_container_width=True, height=400)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#555;font-size:13px;'>🏦 European Bank Churn Analytics | Unified Mentor Internship | ML results in Research Paper & Colab Notebook</div>", unsafe_allow_html=True)
