# ============================================================
# Customer Segmentation & Churn Pattern Analytics
# Streamlit Dashboard — European Banking
# Project: Unified Mentor Internship
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="European Bank Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label { color: #a0a8c8 !important; font-size: 13px !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 28px !important; font-weight: 700 !important; }
    .section-header { font-size: 20px; font-weight: 700; color: #e0e6ff; margin: 24px 0 12px 0; padding-bottom: 8px; border-bottom: 2px solid #3a3f5c; }
    .insight-box { background: #1e2130; border-left: 4px solid #4B7BFF; border-radius: 6px; padding: 10px 16px; margin: 8px 0 16px 0; color: #c8d0f0; font-size: 14px; }
    .risk-high { background:#ff4b4b22; color:#ff4b4b; border:1px solid #ff4b4b55; border-radius:8px; padding:12px 20px; font-size:20px; font-weight:700; text-align:center; }
    .risk-low  { background:#21c35422; color:#21c354; border:1px solid #21c35455; border-radius:8px; padding:12px 20px; font-size:20px; font-weight:700; text-align:center; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1d2e, #0f1117); border-right: 1px solid #2a2d40; }
    .dashboard-title { font-size: 32px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }
    .dashboard-subtitle { font-size: 15px; color: #7a85aa; margin-top: -8px; margin-bottom: 24px; }
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

@st.cache_resource
def load_model():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        df_train = pd.read_csv("European_Bank.csv")
        df_train = df_train.drop(columns=['CustomerId','Surname','Year'], errors='ignore')
        le = LabelEncoder()
        df_train['Geography_enc'] = le.fit_transform(df_train['Geography'])
        df_train['Gender_enc']    = le.fit_transform(df_train['Gender'])
        features = ['CreditScore','Geography_enc','Gender_enc','Age','Tenure',
                    'Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']
        X = df_train[features]
        y = df_train['Exited']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        return model, scaler
    except Exception:
        return None, None

df = load_data()

PLOT_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c8d0f0', family='Inter, sans-serif'),
    xaxis=dict(gridcolor='#2a2d40', linecolor='#3a3f5c'),
    yaxis=dict(gridcolor='#2a2d40', linecolor='#3a3f5c'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#3a3f5c')
)
COLORS = {'churn':'#FF4B4B','retain':'#21C354','blue':'#4B7BFF','orange':'#FF9F40','purple':'#9B59B6','teal':'#1ABC9C'}

with st.sidebar:
    st.markdown("## 🏦 Churn Analytics")
    st.markdown("---")
    st.markdown("### 🔍 Segment Filters")
    selected_geo     = st.selectbox("🌍 Geography",       ['All'] + sorted(df['Geography'].unique().tolist()))
    selected_gender  = st.selectbox("👤 Gender",          ['All'] + sorted(df['Gender'].unique().tolist()))
    selected_age     = st.selectbox("📅 Age Group",       ['All'] + list(df['AgeGroup'].cat.categories))
    selected_credit  = st.selectbox("💳 Credit Band",     ['All'] + list(df['CreditBand'].cat.categories))
    selected_balance = st.selectbox("💰 Balance Segment", ['All'] + list(df['BalanceSegment'].cat.categories))
    st.markdown("---")
    st.info("**Project:** Customer Segmentation & Churn Analytics\n\n**Model:** Logistic Regression (80.5% accuracy)\n\n**Source:** Unified Mentor Internship")

dff = df.copy()
if selected_geo     != 'All': dff = dff[dff['Geography']      == selected_geo]
if selected_gender  != 'All': dff = dff[dff['Gender']          == selected_gender]
if selected_age     != 'All': dff = dff[dff['AgeGroup']        == selected_age]
if selected_credit  != 'All': dff = dff[dff['CreditBand']      == selected_credit]
if selected_balance != 'All': dff = dff[dff['BalanceSegment']  == selected_balance]

st.markdown('<div class="dashboard-title">🏦 European Bank Churn Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Customer Segmentation & Churn Pattern Dashboard — Unified Mentor Internship Project</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🌍 Geography", "👥 Demographics",
    "💎 High-Value", "📈 Engagement", "🤖 Churn Predictor"
])

with tab1:
    total      = len(dff)
    churned    = dff['Exited'].sum()
    churn_rate = churned / total * 100 if total > 0 else 0
    hv         = dff[dff['BalanceSegment'] == 'High-balance']
    hv_churn   = hv['Exited'].mean() * 100 if len(hv) > 0 else 0
    rev_risk   = hv[hv['Exited'] == 1]['Balance'].sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👥 Total Customers",    f"{total:,}")
    col2.metric("🔴 Churned",            f"{churned:,}")
    col3.metric("📉 Churn Rate",         f"{churn_rate:.1f}%", delta=f"{churn_rate-20:.1f}% vs 20% bench", delta_color="inverse")
    col4.metric("💎 High-Value Churn",   f"{hv_churn:.1f}%")
    col5.metric("💸 Revenue at Risk",    f"€{rev_risk/1e6:.1f}M")

    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Overall churn rate is 20.4% — Germany drives the highest risk at 32.4%, nearly double France and Spain. The 46–60 age group churns at 56%+, making it the single highest-risk demographic segment.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        fig_donut = go.Figure(go.Pie(
            labels=['Churned','Retained'], values=[churned, total-churned],
            hole=0.65, marker_colors=[COLORS['churn'], COLORS['retain']],
            textinfo='percent'
        ))
        fig_donut.add_annotation(text=f"<b>{churn_rate:.1f}%</b><br>Churn", x=0.5, y=0.5, showarrow=False, font=dict(size=18, color='white'))
        fig_donut.update_layout(title="Churn Distribution", height=320, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        seg_summary = dff.groupby('Geography')['Exited'].agg(['sum','count']).reset_index()
        seg_summary.columns = ['Geography','Churned','Total']
        seg_summary['ChurnRate'] = seg_summary['Churned'] / seg_summary['Total'] * 100
        fig_geo = px.bar(seg_summary, x='Geography', y='ChurnRate', color='ChurnRate',
                         color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'],
                         text=seg_summary['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                         title='Churn Rate by Country')
        fig_geo.update_traces(textposition='outside')
        fig_geo.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig_geo, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">🌍 Geographic Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Germany\'s 32.4% churn rate is nearly double France and Spain. Country-specific retention strategies are essential — a single blanket approach will under-invest in Germany while wasting resources in lower-risk markets.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, country in zip([col1, col2, col3], ['Germany', 'France', 'Spain']):
        c_data = df[df['Geography'] == country]
        rate = c_data['Exited'].mean() * 100
        col.metric(f"🏳️ {country} Churn Rate", f"{rate:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        geo_detail = dff.groupby('Geography').agg(Total=('Exited','count'), Churned=('Exited','sum')).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Retained', x=geo_detail['Geography'], y=geo_detail['Total']-geo_detail['Churned'], marker_color=COLORS['retain']))
        fig.add_trace(go.Bar(name='Churned',  x=geo_detail['Geography'], y=geo_detail['Churned'], marker_color=COLORS['churn']))
        fig.update_layout(barmode='stack', title='Customer Volume by Country', height=350, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        geo_age = dff.groupby(['Geography','AgeGroup'], observed=True)['Exited'].mean().reset_index()
        geo_age.columns = ['Geography','AgeGroup','ChurnRate']
        geo_age['ChurnRate'] = geo_age['ChurnRate'] * 100
        fig2 = px.density_heatmap(geo_age, x='AgeGroup', y='Geography', z='ChurnRate',
                                   color_continuous_scale='RdYlGn_r', title='Geography × Age Churn Heatmap (%)')
        fig2.update_layout(height=350, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">👥 Demographic Churn Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Female customers churn 52% more than males (25.1% vs 16.5%). The 46–60 age group is the highest-risk demographic at 56%+ churn — more than 2.7x the overall average.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        gender_data = dff.groupby('Gender', observed=True)['Exited'].mean().reset_index()
    gender_data.columns = ['Gender', 'Exited']
    gender_data['ChurnRate'] = gender_data['Exited'] * 100
        gender_data = dff.groupby('Gender')['Exited'].mean().reset_index()
        gender_data['ChurnRate'] = gender_data['Exited'] * 100
        fig = px.bar(gender_data, x='Gender', y='ChurnRate', color='Gender',
                     color_discrete_map={'Female': COLORS['purple'], 'Male': COLORS['teal']},
                     text=gender_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn Rate by Gender')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=320, showlegend=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
       gender_data = dff.groupby('Gender', observed=True)['Exited'].mean().reset_index()
    gender_data.columns = ['Gender', 'Exited']
    gender_data['ChurnRate'] = gender_data['Exited'] * 100
        fig2 = px.bar(age_data, x='AgeGroup', y='ChurnRate', color='ChurnRate',
                      color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'],
                      text=age_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn Rate by Age Group')
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        gender_data = dff.groupby('Gender', observed=True)['Exited'].mean().reset_index()
    gender_data.columns = ['Gender', 'Exited']
    gender_data['ChurnRate'] = gender_data['Exited'] * 100
        fig3 = px.bar(credit_data, x='CreditBand', y='ChurnRate', color='ChurnRate',
                      color_continuous_scale=['#FF4B4B','#FFA62B','#21C354'],
                      text=credit_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn by Credit Band')
        fig3.update_traces(textposition='outside')
        fig3.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig3, use_container_width=True)

    tenure_data = dff.groupby('TenureGroup', observed=True)['Exited'].mean().reset_index()
    tenure_data['ChurnRate'] = tenure_data['Exited'] * 100
    fig4 = px.bar(tenure_data, x='TenureGroup', y='ChurnRate', color='ChurnRate',
                  color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'],
                  text=tenure_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'), title='Churn Rate by Tenure Group')
    fig4.update_traces(textposition='outside')
    fig4.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
    st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">💎 High-Value Customer Churn Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> High-balance customers (>€50K) churn at 22.6%, placing ~€91.5M in deposits at risk. Churn spans all salary levels — service quality and engagement drive exits more than financial stress.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        sample = dff.sample(min(3000, len(dff)), random_state=42)
        fig = px.scatter(sample, x='Balance', y='EstimatedSalary', color='ChurnLabel',
                         color_discrete_map={'Churned': COLORS['churn'], 'Retained': COLORS['retain']},
                         opacity=0.5, hover_data=['Age','Geography','CreditScore'],
                         title='Balance vs Estimated Salary (Churn Highlighted)',
                         labels={'Balance':'Account Balance (€)','EstimatedSalary':'Estimated Salary (€)'})
        fig.update_layout(height=380, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bal_data = dff.groupby('BalanceSegment', observed=True)['Exited'].mean().reset_index()
        bal_data['ChurnRate'] = bal_data['Exited'] * 100
        fig2 = px.bar(bal_data, x='BalanceSegment', y='ChurnRate', color='ChurnRate',
                      color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'],
                      text=bal_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                      title='Churn Rate by Balance Segment')
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)

with tab5:
    st.markdown('<div class="section-header">📈 Engagement Drop Indicator</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Inactive members churn at 26.9% vs 14.3% for active members. Customers with 2 products have the lowest churn (7.6%), while 3–4 product holders churn at 82–100% — product overload is a churn trigger.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        active_data = dff.groupby('IsActiveMember')['Exited'].mean().reset_index()
        active_data['Status'] = active_data['IsActiveMember'].map({0:'Inactive', 1:'Active'})
        active_data['ChurnRate'] = active_data['Exited'] * 100
        fig = px.bar(active_data, x='Status', y='ChurnRate', color='Status',
                     color_discrete_map={'Inactive': COLORS['churn'], 'Active': COLORS['retain']},
                     text=active_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                     title='Active vs Inactive Member Churn Rate')
        fig.update_traces(textposition='outside')
        fig.update_layout(height=350, showlegend=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        prod_data = dff.groupby('NumOfProducts')['Exited'].mean().reset_index()
        prod_data['ChurnRate'] = prod_data['Exited'] * 100
        fig2 = px.bar(prod_data, x='NumOfProducts', y='ChurnRate', color='ChurnRate',
                      color_continuous_scale=['#21C354','#FFA62B','#FF4B4B'],
                      text=prod_data['ChurnRate'].apply(lambda x: f'{x:.1f}%'),
                      title='Churn Rate by Number of Products',
                      labels={'NumOfProducts':'Number of Products'})
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=350, coloraxis_showscale=False, margin=dict(t=40,b=0,l=0,r=0), **PLOT_THEME)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">🔎 Segment Drill-Down Table</div>', unsafe_allow_html=True)
    seg_table = dff.groupby(['Geography','AgeGroup','Gender'], observed=True).agg(
        Total=('Exited','count'), Churned=('Exited','sum'),
        AvgBalance=('Balance','mean'), AvgCreditScore=('CreditScore','mean')
    ).reset_index()
    seg_table['ChurnRate%'] = (seg_table['Churned'] / seg_table['Total'] * 100).round(1)
    seg_table['AvgBalance'] = seg_table['AvgBalance'].round(0).astype(int)
    seg_table['AvgCreditScore'] = seg_table['AvgCreditScore'].round(0).astype(int)
    seg_table = seg_table.sort_values('ChurnRate%', ascending=False)
    st.dataframe(seg_table, use_container_width=True, height=400)

with tab6:
    model, scaler = load_model()
    st.markdown('<div class="section-header">🤖 AI-Powered Churn Predictor</div>', unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ ML model is currently unavailable on this server. The full model results (86.8% accuracy, ROC-AUC 86.7%) are documented in the research paper and Colab notebook.")
        st.info("📊 Model training and evaluation was performed in Google Colab. See the attached research paper for full ML results including confusion matrix, feature importance, and classification report.")
    else:
        st.markdown("Enter a customer's details below to instantly predict their churn risk using our **ML classification model**.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📋 Personal Details**")
            age          = st.slider("Age", 18, 92, 40)
            gender       = st.selectbox("Gender", ["Male", "Female"])
            geography    = st.selectbox("Country", ["France", "Germany", "Spain"])
            credit_score = st.slider("Credit Score", 350, 850, 650)

        with col2:
            st.markdown("**💰 Financial Profile**")
            balance          = st.number_input("Account Balance (€)", 0, 300000, 50000, step=1000)
            estimated_salary = st.number_input("Estimated Salary (€)", 0, 300000, 80000, step=1000)
            num_products     = st.selectbox("Number of Products", [1, 2, 3, 4])
            has_cr_card      = st.selectbox("Has Credit Card?", ["Yes", "No"])

        with col3:
            st.markdown("**🏦 Engagement**")
            tenure    = st.slider("Tenure (years)", 0, 10, 5)
            is_active = st.selectbox("Active Member?", ["Yes", "No"])
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🔮 Predict Churn Risk", use_container_width=True):
                geo_map    = {'France': 0, 'Germany': 1, 'Spain': 2}
                gender_map = {'Female': 0, 'Male': 1}
                input_data = np.array([[
                    credit_score, geo_map[geography], gender_map[gender],
                    age, tenure, balance, num_products,
                    1 if has_cr_card == "Yes" else 0,
                    1 if is_active   == "Yes" else 0,
                    estimated_salary
                ]])
                input_scaled = scaler.transform(input_data)
                prob         = model.predict_proba(input_scaled)[0][1] * 100
                prediction   = model.predict(input_scaled)[0]

                st.markdown("---")
                st.markdown("### 📊 Prediction Result")
                col_a, col_b = st.columns(2)
                with col_a:
                    if prediction == 1:
                        st.markdown(f'<div class="risk-high">⚠️ HIGH CHURN RISK<br><span style="font-size:32px">{prob:.1f}%</span><br>probability of churning</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-low">✅ LOW CHURN RISK<br><span style="font-size:32px">{prob:.1f}%</span><br>probability of churning</div>', unsafe_allow_html=True)

                with col_b:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob,
                        number={'suffix': '%', 'font': {'size': 28, 'color': 'white'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#c8d0f0'},
                            'bar': {'color': '#FF4B4B' if prob > 50 else '#21C354'},
                            'steps': [
                                {'range': [0, 30],   'color': '#21c35422'},
                                {'range': [30, 60],  'color': '#ffa62b22'},
                                {'range': [60, 100], 'color': '#ff4b4b22'},
                            ],
                            'threshold': {'line': {'color': 'white', 'width': 2}, 'value': 50}
                        }
                    ))
                    fig_gauge.update_layout(height=220, margin=dict(t=20,b=0,l=20,r=20), **PLOT_THEME)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("### 💡 Retention Recommendations")
                tips = []
                if is_active == "No":
                    tips.append("🔔 **Activate membership** — inactive members churn at 26.9% vs 14.3% for active members")
                if num_products >= 3:
                    tips.append("📦 **Review product holdings** — customers with 3+ products churn at 82–100%")
                if geography == "Germany":
                    tips.append("🇩🇪 **Germany high-risk market** — apply country-specific retention campaign")
                if age >= 46 and age <= 60:
                    tips.append("👤 **Age 46–60 high-risk group** — offer premium loyalty benefits")
                if balance > 50000:
                    tips.append("💎 **High-value customer** — assign dedicated relationship manager")
                if not tips:
                    tips.append("✅ This customer shows low churn risk — maintain regular engagement")
                for tip in tips:
                    st.markdown(f"- {tip}")

st.markdown("---")
st.markdown("<div style='text-align:center; color:#555; font-size:13px;'>🏦 Customer Segmentation & Churn Pattern Analytics in European Banking &nbsp;|&nbsp; Unified Mentor Internship &nbsp;|&nbsp; AI-Powered Churn Predictor</div>", unsafe_allow_html=True)
