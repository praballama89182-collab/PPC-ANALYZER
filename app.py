import streamlit as st
import pandas as pd
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Ecommerce Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #232f3e; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

def normalize_match_type(val):
    if pd.isna(val): return 'UNKNOWN'
    val = str(val).upper()
    if 'EXACT' in val: return 'EXACT'
    if 'PHRASE' in val: return 'PHRASE'
    if 'BROAD' in val: return 'BROAD'
    return 'AUTO/OTHER'

def determine_winner(group, improvement_thresh, min_orders):
    max_sales_idx = group['sales_val'].idxmax()
    sales_leader = group.loc[max_sales_idx]
    
    max_roas_idx = group['calculated_roas'].idxmax()
    roas_leader = group.loc[max_roas_idx]
    
    if max_sales_idx == max_roas_idx:
        return max_sales_idx, "Best Sales & ROAS"
    
    roas_sales = sales_leader['calculated_roas']
    roas_challenger = roas_leader['calculated_roas']
    
    improvement = (roas_challenger - roas_sales) / roas_sales if roas_sales > 0 else 999
    
    if (improvement >= (improvement_thresh / 100.0)) and (roas_leader['orders_val'] >= min_orders):
        return max_roas_idx, f"Efficient (ROAS +{improvement:.0%})"
    else:
        return max_sales_idx, "Volume Leader"

def to_excel(dfs):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in dfs.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False) 
    return output.getvalue()

# --- MAIN APP ---

def main():
    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🛒 Prabal Ecommerce Analyzer")
        st.markdown("---")
        
        uploaded_file = st.file_uploader("Upload Search Term Report", type=["csv", "xlsx"])
        
        df = None
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded_file)
                else:
                    df_raw = pd.read_excel(uploaded_file)
                
                df_raw.columns = df_raw.columns.str.strip()
                
                port_col = next((c for c in df_raw.columns if 'Portfolio' in c), None)
                
                if port_col:
                    st.subheader("2. Filter Data")
                    all_portfolios = df_raw[port_col].dropna().unique().tolist()
                    selected_ports = st.multiselect("Select Portfolios", options=all_portfolios, default=all_portfolios)
                    
                    if selected_ports:
                        df = df_raw[df_raw[port_col].isin(selected_ports)].copy()
                    else:
                        df = df_raw.copy()
                else:
                    df = df_raw.copy()
                    st.info("No 'Portfolio' column found.")

            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        if df is not None:
            st.markdown("---")
            st.subheader("3. Analysis Thresholds")
            
            with st.expander("⚔️ Cannibalization Rules", expanded=False):
                roas_threshold = st.slider("Better ROAS Threshold (%)", 30, 200, 100, 10)
                min_orders_cannibal = st.number_input("Min Orders to Win", 1, 10, 2)
                
            with st.expander("🌾 Harvesting Rules", expanded=True):
                min_orders_harvest = st.number_input("Min Orders", 1, 10, 2)
                min_roas_harvest = st.number_input("Min ROAS", 0.1, 10.0, 1.0, 0.1)
                
            with st.expander("💰 CPC Analyzer", expanded=False):
                top_n_terms = st.slider("Analyze Top N Terms", 10, 100, 50)
                bad_roas_limit = st.number_input("Low ROAS Flag (<)", 0.1, 5.0, 1.0, 0.1)

    # --- MAIN CONTENT AREA ---
    if df is not None:
        try:
            col_map = {
                'date': next((c for c in df.columns if 'Date' in c), None),
                'term': next((c for c in df.columns if 'Matched product' in c or 'Customer Search Term' in c), None),
                'camp': next((c for c in df.columns if 'Campaign Name' in c), None),
                'adg': next((c for c in df.columns if 'Ad Group Name' in c), None),
                'match': next((c for c in df.columns if 'Match Type' in c), None),
                'orders': next((c for c in df.columns if 'Orders' in c or 'Units' in c), None),
                'sales': next((c for c in df.columns if 'Sales' in c), None),
                'spend': next((c for c in df.columns if 'Spend' in c), None),
                'clicks': next((c for c in df.columns if 'Clicks' in c), None),
                'impressions': next((c for c in df.columns if 'Impressions' in c), None)
            }

            if any(v is None for v in ['term', 'camp', 'adg', 'spend', 'sales']):
                st.error(f"Missing essential columns. Found: {col_map}")
            else:
                for c in ['orders', 'sales', 'spend', 'clicks', 'impressions']:
                    if col_map[c]:
                        df[col_map[c]] = pd.to_numeric(df[col_map[c]], errors='coerce').fillna(0)
                
                df['norm_match'] = df[col_map['match']].apply(normalize_match_type)
                if col_map['date']:
                    df['Date'] = pd.to_datetime(df[col_map['date']], errors='coerce')

                # --- AGGREGATION ---
                agg_cols = [col_map['term'], col_map['camp'], col_map['adg'], 'norm_match']
                df_agg = df.groupby(agg_cols, as_index=False).agg({
                    col_map['spend']: 'sum',
                    col_map['sales']: 'sum',
                    col_map['orders']: 'sum',
                    col_map['clicks']: 'sum',
                    col_map['impressions']: 'sum'
                })
                
                df_agg.rename(columns={
                    col_map['term']: 'Search Term', col_map['camp']: 'Campaign', col_map['adg']: 'Ad Group',
                    col_map['orders']: 'Orders', col_map['sales']: 'Sales', col_map['spend']: 'Spend',
                    col_map['clicks']: 'Clicks', col_map['impressions']: 'Impressions'
                }, inplace=True)
                
                # --- CALCULATED METRICS (CTR & CVR ADDED) ---
                df_agg['ROAS'] = df_agg.apply(lambda x: x['Sales']/x['Spend'] if x['Spend'] > 0 else 0, axis=1)
                df_agg['CPC'] = df_agg.apply(lambda x: x['Spend']/x['Clicks'] if x['Clicks'] > 0 else 0, axis=1)
                df_agg['ACOS'] = df_agg.apply(lambda x: (x['Spend']/x['Sales'])*100 if x['Sales'] > 0 else 0, axis=1)
                df_agg['CTR'] = df_agg.apply(lambda x: (x['Clicks']/x['Impressions'])*100 if x['Impressions'] > 0 else 0, axis=1)
                df_agg['CVR'] = df_agg.apply(lambda x: (x['Orders']/x['Clicks'])*100 if x['Clicks'] > 0 else 0, axis=1)

                for col in ['Spend', 'Sales', 'ROAS', 'CPC', 'ACOS', 'CTR', 'CVR']:
                    df_agg[col] = df_agg[col].round(1)

                existing_exact = set(df_agg[df_agg['norm_match'] == 'EXACT']['Search Term'].str.lower().unique())

                st.title("Prabal Ecommerce Analyzer")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Spend", f"₹{df_agg['Spend'].sum():,.1f}")
                c2.metric("Total Sales", f"₹{df_agg['Sales'].sum():,.1f}")
                acc_roas = df_agg['Sales'].sum() / df_agg['Spend'].sum() if df_agg['Spend'].sum() > 0 else 0
                c3.metric("Account ROAS", f"{acc_roas:.1f}")
                c4.metric("Unique Search Terms", f"{df_agg['Search Term'].nunique():,}")

                tabs = st.tabs(["⚔️ Cannibalization", "🌾 Harvesting", "💰 CPC Analyzer", "📅 Best Days", "💸 Wasted Spend"])

                # TAB 1: CANNIBALIZATION
                with tabs[0]:
                    st.subheader("Detect & Fix Self-Competition")
                    sales_df = df_agg[df_agg['Orders'] > 0].copy()
                    dupe_counts = sales_df.groupby('Search Term').size()
                    cannibal_list = dupe_counts[dupe_counts > 1].index.tolist()
                    
                    cannibal_results = []
                    if cannibal_list:
                        for term in cannibal_list:
                            subset = sales_df[sales_df['Search Term'] == term].rename(columns={'Sales': 'sales_val', 'Spend': 'spend_val', 'ROAS': 'calculated_roas', 'Orders': 'orders_val'}).copy()
                            win_idx, _ = determine_winner(subset, roas_threshold, min_orders_cannibal)
                            
                            for idx, row in subset.iterrows():
                                is_winner = (idx == win_idx)
                                cannibal_results.append({
                                    'Search Term': term, 'Campaign': row['Campaign'], 'Ad Group': row['Ad Group'],
                                    'CTR %': row['CTR'], 'CVR %': row['CVR'], 'CPC': row['CPC'], 
                                    'Spend': row['spend_val'], 'Sales': row['sales_val'], 'Orders': row['orders_val'],
                                    'ROAS': row['calculated_roas'], 'Action': "✅ KEEP" if is_winner else "⛔ NEGATE"
                                })
                        
                        df_cannibal = pd.DataFrame(cannibal_results)
                        st.dataframe(
                            df_cannibal.style.apply(lambda x: ['background-color: #ffebee' if 'NEGATE' in str(v) else '' for v in x], axis=1)
                            .format({'CTR %': '{:.1f}%', 'CVR %': '{:.1f}%'}), 
                            use_container_width=True
                        )
                    else:
                        st.success("No cannibalization found.")
                        df_cannibal = pd.DataFrame()

                # TAB 2: HARVESTING
                with tabs[1]:
                    st.subheader("Strict Growth Opportunities")
                    candidates = df_agg[(df_agg['norm_match'] != 'EXACT') & (df_agg['Orders'] >= min_orders_harvest) & (df_agg['ROAS'] >= min_roas_harvest)].copy()
                    harvest_results = []
                    for idx, row in candidates.iterrows():
                        if row['Search Term'].lower() not in existing_exact:
                            harvest_results.append({
                                'Search Term': row['Search Term'], 'Rec': '🚀 NEW EXACT', 
                                'Source Camp': row['Campaign'], 'Orders': row['Orders'], 
                                'Sales': row['Sales'], 'ROAS': row['ROAS'], 'CVR %': row['CVR']
                            })
                    df_harvest = pd.DataFrame(harvest_results)
                    st.dataframe(df_harvest, use_container_width=True)

                # TAB 3: CPC ANALYZER
                with tabs[2]:
                    st.subheader(f"Top {top_n_terms} Search Terms")
                    top_terms = df_agg.groupby('Search Term')['Spend'].sum().nlargest(top_n_terms).index.tolist()
                    df_top = df_agg[df_agg['Search Term'].isin(top_terms)].copy()
                    cpc_results = []
                    for term in top_terms:
                        subset = df_top[df_top['Search Term'] == term]
                        avg_cpc = subset['CPC'].mean()
                        for idx, row in subset.iterrows():
                            rec = "✅ Healthy"
                            if row['ROAS'] < bad_roas_limit: rec = "⚠️ Low ROAS"
                            elif row['CPC'] > (avg_cpc * 1.3): rec = "⚠️ High CPC"
                            cpc_results.append({'Search Term': term, 'Campaign': row['Campaign'], 'CPC': row['CPC'], 'ROAS': row['ROAS'], 'Rec': rec})
                    st.dataframe(pd.DataFrame(cpc_results), use_container_width=True)

                # TAB 4: BEST DAYS
                with tabs[3]:
                    st.subheader("📅 Day Parting")
                    if col_map['date']:
                        day_agg = df.groupby(df['Date'].dt.day_name()).agg({col_map['spend']: 'sum', col_map['sales']: 'sum', col_map['orders']: 'sum'}).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                        st.bar_chart(day_agg[col_map['spend']])
                        st.dataframe(day_agg, use_container_width=True)

                # TAB 5: WASTED SPEND
                with tabs[4]:
                    st.subheader("💸 Wasted Spend")
                    waste_thresh = st.slider("Min Spend", 50, 1000, 200)
                    df_waste = df_agg[(df_agg['Orders'] == 0) & (df_agg['Spend'] >= waste_thresh)].sort_values(by='Spend', ascending=False)
                    st.dataframe(df_waste[['Search Term', 'Campaign', 'Spend', 'Clicks', 'CPC']], use_container_width=True)

                # EXPORT
                st.markdown("---")
                export_data = {'Cannibalization': df_cannibal, 'Harvesting': df_harvest, 'Wasted_Spend': df_waste}
                st.download_button("📥 Download Master Report", data=to_excel(export_data), file_name="Prabal_PPC_Report.xlsx")

        except Exception as e:
            st.error(f"Analysis Error: {e}")

if __name__ == "__main__":
    main()
