# app2.py

# 1. IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# --- NEW IMPORTS for APRIORI ---
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    st.error("Apriori module not found. Please run 'pip install mlxtend'")
    # Stop the app if it can't load
    st.stop()


# 2. DATA LOADING (IMPROVED & FIXED)
# This function loads data from your 'data/' folder
@st.cache_data
def load_data():
    try:
        # Load all 14 files with na_values='\\N' to handle missing data
        results = pd.read_csv('data/results.csv', na_values='\\N')
        races = pd.read_csv('data/races.csv', na_values='\\N')
        status = pd.read_csv('data/status.csv', na_values='\\N')
        drivers = pd.read_csv('data/drivers.csv', na_values='\\N')
        constructors = pd.read_csv('data/constructors.csv', na_values='\\N')
        circuits = pd.read_csv('data/circuits.csv', na_values='\\N')
        
        # Load other files (even if not merged, this is robust)
        pd.read_csv('data/sprint_results.csv', na_values='\\N')
        pd.read_csv('data/driver_standings.csv', na_values='\\N')
        pd.read_csv('data/pit_stops.csv', na_values='\\N')
        pd.read_csv('data/lap_times.csv', na_values='\\N')
        pd.read_csv('data/qualifying.csv', na_values='\\N')
        pd.read_csv('data/constructor_standings.csv', na_values='\\N')
        pd.read_csv('data/constructor_results.csv', na_values='\\N')
        pd.read_csv('data/seasons.csv', na_values='\\N')

        # --- Data Pre-processing & Merging ---
        drivers.rename(columns={
            'url': 'driver_url', 
            'nationality': 'driver_nationality'
        }, inplace=True)
        drivers['driver_name'] = drivers['forename'] + ' ' + drivers['surname']

        constructors.rename(columns={
            'name': 'constructor_name', 
            'url': 'constructor_url',
            'nationality': 'constructor_nationality'
        }, inplace=True)

        races.rename(columns={
            'name': 'race_name', 
            'url': 'race_url',
            'date': 'race_date',
            'time': 'race_time'
        }, inplace=True)

        circuits.rename(columns={
            'name': 'circuit_name', 
            'location': 'circuit_location',
            'country': 'circuit_country',
            'url': 'circuit_url'
        }, inplace=True)

        # --- Create Master DataFrame ---
        df = results.merge(races, on='raceId', how='left')
        df = df.merge(drivers, on='driverId', how='left')
        df = df.merge(constructors, on='constructorId', how='left')
        df = df.merge(circuits, on='circuitId', how='left')
        df = df.merge(status, on='statusId', how='left')

        # --- Feature Engineering ---
        df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
        df['position'] = pd.to_numeric(df['position'], errors='coerce')
        df['race_date'] = pd.to_datetime(df['race_date'])
        
        df['is_win'] = (df['position'] == 1).astype(int)
        df['is_podium'] = df['position'].isin([1, 2, 3]).astype(int)
        df['is_dnf'] = (df['statusId'] != 1).astype(int) # Status 1 is 'Finished'

        df = df.sort_values(by=['year', 'driverId', 'race_date'])
        df['season_points'] = df.groupby(['year', 'driverId'])['points'].cumsum()

        # Return the data
        return df, drivers, constructors, circuits, races

    except FileNotFoundError as e:
        st.exception(e)
        st.error("Error: A CSV file is missing from the 'data/' folder.")
        return None, None, None, None, None

# 3. PAGE CONFIGURATION
st.set_page_config(
    page_title="F1 Data Analytics Dashboard",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏎️ F1 Data Analytics Dashboard")
st.markdown("An interactive dashboard for exploring Formula 1 race data")

# Load data
df, drivers, constructors, circuits, races = load_data()

# 4. SIDEBAR FILTERS
if df is not None:
    st.sidebar.header("Data Filters")
    
    years = sorted(df['year'].unique())
    min_year, max_year = int(min(years)), int(max(years))
    
    selected_year_range = st.sidebar.slider(
        "Select Year Range", 
        min_value=min_year, 
        max_value=max_year, 
        value=(max_year, max_year)
    )
    
    df_filtered = df[(df['year'] >= selected_year_range[0]) & (df['year'] <= selected_year_range[1])].copy()
    
    driver_list = sorted(df_filtered['driver_name'].dropna().unique())
    selected_drivers = st.sidebar.multiselect("Select Drivers (optional)", driver_list)
    
    constructor_list = sorted(df_filtered['constructor_name'].dropna().unique())
    selected_constructors = st.sidebar.multiselect("Select Constructors (optional)", constructor_list)
    
    if selected_drivers:
        df_filtered = df_filtered[df_filtered['driver_name'].isin(selected_drivers)]
    if selected_constructors:
        df_filtered = df_filtered[df_filtered['constructor_name'].isin(selected_constructors)]
    
    year_display = f"{selected_year_range[0]}-{selected_year_range[1]}" if selected_year_range[0] != selected_year_range[1] else str(selected_year_range[0])
        
    if df_filtered.empty:
        st.warning(f"No data found for the selected filters in {year_display}.")
    else:
        # 5. MAIN PAGE - ORGANIZED WITH 9 TABS 
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "📊 Statistics & KPIs", 
            "👥 Driver Comparison",
            "🏆 Win Gauge",
            "📖 Association Rules",  
            "🤖 PCA & Clustering", 
            "📈 Regression Model",
            "⚠ Outlier Detection",
            "🔍 Correlation",
            "🌍 Circuit Map"
        ])

        # --- FEATURE 1: Statistics of Data & KPIs ---
        with tab1:
            st.header(f"Season Overview: {year_display}")
            
            total_races = df_filtered['raceId'].nunique()
            total_drivers = df_filtered['driverId'].nunique()
            total_constructors = df_filtered['constructorId'].nunique()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Races in Season", total_races)
            col2.metric("Total Drivers", total_drivers)
            col3.metric("Total Constructors", total_constructors)

            st.markdown("---")
            st.subheader("Performance Leaders (Full Season)")

            col1, col2 = st.columns(2)

            with col1:
                wins = df_filtered[df_filtered['is_win'] == 1]['driver_name'].value_counts().reset_index()
                wins.columns = ['Driver', 'Wins']
                fig_wins = px.bar(wins.head(10), x='Driver', y='Wins', title="Top Drivers by Wins",
                                    color='Wins', color_continuous_scale='reds')
                st.plotly_chart(fig_wins, use_container_width=True)

            with col2:
                podiums = df_filtered[df_filtered['is_podium'] == 1]['constructor_name'].value_counts().reset_index()
                podiums.columns = ['Constructor', 'Podiums']
                fig_podiums = px.bar(podiums.head(10), x='Constructor', y='Podiums', title="Top Constructors by Podiums",
                                        color='Podiums', color_continuous_scale='blues')
                st.plotly_chart(fig_podiums, use_container_width=True)
            
            
            # --- PIE CHART SECTION ---
            st.markdown("---")
            st.subheader("Race Outcomes Analysis")

            status_counts = df_filtered['status'].value_counts()
            
            if len(status_counts) > 7:
                top_statuses = status_counts.nlargest(7)
                other_count = status_counts.nsmallest(len(status_counts) - 7).sum()
                if other_count > 0:
                    try:
                        top_statuses.loc['Other'] = other_count
                    except:
                        top_statuses._set_value('Other', other_count)

                status_data = top_statuses.reset_index()
            else:
                status_data = status_counts.reset_index()
            
            status_data.columns = ['Status', 'Count']

            fig_pie = px.pie(status_data, 
                             names='Status', 
                             values='Count', 
                             title=f"Finishing Status Distribution ({year_display})",
                             )
            fig_pie.update_traces(textinfo='percent+label', pull=[0.05 if i == 0 else 0 for i in range(len(status_data))])
            st.plotly_chart(fig_pie, use_container_width=True)


            if selected_drivers:
                st.markdown("---")
                st.subheader("Driver Points Progression")
                fig_points = px.line(df_filtered.sort_values('race_date'), 
                                        x='race_name', y='season_points', color='driver_name',
                                        title="Championship Points Over Season",
                                        labels={'race_name': 'Race', 'season_points': 'Cumulative Points'})
                st.plotly_chart(fig_points, use_container_width=True)


        # --- FEATURE 2: Driver Comparison ---
        with tab2:
            st.header("Driver Head-to-Head")
            st.write(f"Compare two specific drivers from the {year_display} season(s).")

            all_drivers_comp = sorted(df_filtered['driver_name'].dropna().unique())
            
            if len(all_drivers_comp) < 2:
                st.warning("Not enough drivers in the filtered data to compare.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    d1_name = st.selectbox("Driver 1", all_drivers_comp, index=0)
                with c2:
                    d2_name = st.selectbox("Driver 2", all_drivers_comp, index=1)
                
                d1_df = df_filtered[df_filtered['driver_name'] == d1_name]
                d2_df = df_filtered[df_filtered['driver_name'] == d2_name]

                def get_stats(df):
                    if df.empty:
                        return {"Wins": 0, "Podiums": 0, "Avg Finish": np.nan, "DNF %": np.nan, "Total Points": 0}
                    return {
                        "Wins": int(df['is_win'].sum()),
                        "Podiums": int(df['is_podium'].sum()),
                        "Avg Finish": df['position'].mean(),
                        "DNF %": df['is_dnf'].mean() * 100,
                        "Total Points": int(df['points'].sum())
                    }

                s1, s2 = get_stats(d1_df), get_stats(d2_df)
                
                st.markdown(f"### {d1_name} vs {d2_name}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader(d1_name)
                    st.metric("Total Points", f"{s1['Total Points']:.0f}")
                    st.metric("Wins", s1["Wins"])
                    st.metric("Podiums", s1["Podiums"])
                    st.metric("Avg Finish", f"{s1['Avg Finish']:.2f}")
                    st.metric("DNF %", f"{s1['DNF %']:.2f}%")
                
                with c2:
                    st.subheader(d2_name)
                    st.metric("Total Points", f"{s2['Total Points']:.0f}")
                    st.metric("Wins", s2["Wins"])
                    st.metric("Podiums", s2["Podiums"])
                    st.metric("Avg Finish", f"{s2['Avg Finish']:.2f}")
                    st.metric("DNF %", f"{s2['DNF %']:.2f}%")

            st.markdown("---")
            
            st.header("Overall Finishing Consistency")
            st.write(f"Compares the distribution of finishing positions for all drivers in the filtered data.")
            
            df_plot = df_filtered.dropna(subset=['position'])
            
            if not df_plot.empty:
                df_plot.loc[:, 'position'] = pd.to_numeric(df_plot['position'])
            
            if df_plot.empty:
                st.warning("No finishing position data to display for the selected filters.")
            else:
                fig_box = px.box(df_plot,
                    x='driver_name',
                    y='position',
                    color='constructor_name',
                    title=f"Finishing Position Distribution ({year_display})",
                    labels={'position': 'Finishing Position', 'driver_name': 'Driver'}
                )
                fig_box.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_box, use_container_width=True)

        # --- FEATURE 3: Win Probability Gauge ---
        with tab3:
            st.header("🏆 Win Rate Gauge")
            st.write(f"Win rate for a driver in {year_display}.")

            all_drivers_gauge = sorted(df_filtered['driver_name'].dropna().unique())
            
            if not all_drivers_gauge:
                st.warning("No driver data to display.")
            else:
                driver_gauge = st.selectbox("Select Driver", all_drivers_gauge, key="gauge_driver")
                d_gauge = df_filtered[df_filtered['driver_name'] == driver_gauge]
                
                if d_gauge.empty:
                    win_rate = 0.0
                    total_races = 0
                else:
                    win_rate = (d_gauge['is_win'] == 1).mean() * 100
                    total_races = d_gauge['raceId'].nunique()

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=win_rate,
                    title={'text': f"Win % (from {total_races} races)"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "#2ca02c"}, 
                           'steps': [
                               {'range': [0, 10], 'color': "#d62728"}, 
                               {'range': [10, 30], 'color': "#ff7f0e"}, 
                               {'range': [30, 100], 'color': "#1f77b4"}, 
                           ]}
                ))
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, use_container_width=True)

        
        # --- FEATURE 4: Association Rules (MOVED) ---
        with tab4:
            st.header("📖 Association Rule Mining (Apriori)")
            st.write("Find hidden rules and associations in race data.")
            st.info("Example Rule: IF a driver is in a `{Red Bull}` AND `{Starts in Top 5}`, THEN they are 85% likely to `{Finish on the Podium}`.")

            try:
                # --- 1. Pre-process Data for Transactions ---
                @st.cache_data
                def create_transactions(df_in):
                    transactions = []
                    
                    df_trans = df_in[['grid', 'position', 'points', 'is_dnf', 'is_win', 'constructor_name', 'circuit_country']].copy()

                    # Bin Grid Position
                    df_trans['grid_bin'] = pd.cut(df_trans['grid'], 
                                                  bins=[0, 5, 10, 25], 
                                                  labels=['Grid: P1-P5', 'Grid: P6-P10', 'Grid: P11+'], 
                                                  right=True)

                    # Bin Outcome
                    def get_outcome(row):
                        if row['is_dnf'] == 1:
                            return 'Outcome: DNF'
                        if row['is_win'] == 1:
                            return 'Outcome: Win'
                        if row['position'] in [2, 3]:
                            return 'Outcome: Podium (2-3)'
                        if row['points'] > 0:
                            return 'Outcome: Points'
                        return 'Outcome: No Points'

                    df_trans['outcome'] = df_trans.apply(get_outcome, axis=1)

                    # Create transactions
                    for _, row in df_trans.iterrows():
                        transaction = []
                        if pd.notna(row['constructor_name']):
                            transaction.append(f"Team: {row['constructor_name']}")
                        if pd.notna(row['grid_bin']):
                            transaction.append(str(row['grid_bin']))
                        if pd.notna(row['outcome']):
                            transaction.append(str(row['outcome']))
                        if pd.notna(row['circuit_country']):
                            transaction.append(f"Country: {row['circuit_country']}")
                        transactions.append(transaction)
                    return transactions

                transactions = create_transactions(df_filtered)

                if not transactions:
                    st.warning("No transactions to analyze for the selected filters.")
                else:
                    # --- 2. Encode Transactions ---
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

                    # --- 3. User Controls ---
                    st.subheader("Rule Generation Controls")
                    col1, col2 = st.columns(2)
                    with col1:
                        min_support = st.slider("Minimum Support (Itemset Frequency)", 0.01, 0.5, 0.05, 0.01)
                    with col2:
                        min_confidence = st.slider("Minimum Confidence (Rule Strength)", 0.1, 1.0, 0.7, 0.1)

                    # --- 4. Run Apriori ---
                    frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)

                    if frequent_itemsets.empty:
                        st.warning("No frequent itemsets found with the current support setting. Try lowering the 'Minimum Support' slider.")
                    else:
                        # --- 5. Generate and Display Rules ---
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                        
                        if rules.empty:
                            st.warning("No rules generated with the current confidence setting. Try lowering the 'Minimum Confidence' slider.")
                        else:
                            st.subheader("Generated Association Rules")
                            
                            rules_display = rules.copy()
                            rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                            rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
                            
                            st.dataframe(rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']].sort_values(by='lift', ascending=False))
                            
                            # st.markdown("---")
                            # st.subheader("How to Read These Rules:")
                            # st.markdown("- **antecedents (IF)**: The item(s) on the left side of the rule.")
                            # st.markdown("- **consequents (THEN)**: The item(s) on the right side.")
                            # st.markdown("- **support**: How often this combination appears in the *entire dataset*.")
                            # st.markdown("- **confidence**: The probability of seeing {THEN} *when* {IF} is present. (e.g., 0.8 = 80% chance).")
                            # st.markdown("- **lift**: How much more likely {THEN} is, given {IF}. (Lift > 1 means the rule is interesting).")

            except Exception as e:
                st.error(f"An error occurred during Apriori analysis: {e}")
                st.exception(e)
                
                
        # --- FEATURE 5: PCA & Clustering (MOVED) ---
        with tab5:
            st.header("Driver Performance Clustering")
            st.write("Using PCA to visualize driver profiles and cluster algorithms to find groups.")
            
            # --- 1. Data Prep ---
            pca_features = ['grid', 'position', 'points', 'is_win', 'is_podium', 'is_dnf']
            driver_stats = df_filtered.groupby('driver_name')[pca_features].mean().reset_index()
            
            if driver_stats.shape[0] < 3:
                st.warning("Not enough driver data to perform PCA/Clustering. Select more drivers or a different year range.")
            else:
                driver_stats_clean = driver_stats.dropna()
                
                if driver_stats_clean.shape[0] < 3:
                        st.warning("Not enough complete driver data for PCA/Clustering. (Removed NaNs)")
                else:
                    driver_names = driver_stats_clean['driver_name'].values
                    X = driver_stats_clean[pca_features].values
                    X_scaled = StandardScaler().fit_transform(X)
                    
                    # --- 2. PCA (Always Show) ---
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    driver_stats_clean['PC1'] = X_pca[:, 0]
                    driver_stats_clean['PC2'] = X_pca[:, 1]

                    st.subheader("PCA: 2D Performance Profile")
                    fig_pca = px.scatter(driver_stats_clean, 
                                            x='PC1', y='PC2', 
                                            hover_name='driver_name', 
                                            title="Driver Performance PCA Plot",
                                            text='driver_name')
                    fig_pca.update_traces(textposition='top center')
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # --- 3. Clustering Algorithm Selection ---
                    st.subheader("Clustering Algorithms")
                    st.write("Apply different algorithms to the scaled performance data to find groups.")
                    
                    cluster_method = st.selectbox(
                        "Choose Clustering Method", 
                        ["K-Means", "Hierarchical (Dendrogram)", "DBSCAN"]
                    )

                    # --- K-Means ---
                    if cluster_method == "K-Means":
                        st.subheader("K-Means Clustering")
                        max_clusters = min(driver_stats_clean.shape[0], 10)
                        n_clusters = st.slider("Select number of clusters (K)", 2, max_clusters, 3)

                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        driver_stats_clean['Cluster'] = kmeans.fit_predict(X_scaled)
                        driver_stats_clean['Cluster'] = driver_stats_clean['Cluster'].astype(str)
                        
                        fig_kmeans = px.scatter(driver_stats_clean, 
                                                x='PC1', y='PC2', 
                                                color='Cluster',
                                                hover_name='driver_name', 
                                                title=f"Driver Clusters (K={n_clusters})")
                        st.plotly_chart(fig_kmeans, use_container_width=True)

                    # --- Hierarchical ---
                    elif cluster_method == "Hierarchical (Dendrogram)":
                        st.subheader("Hierarchical Clustering Dendrogram")
                        st.write("This chart shows how clusters are merged. A longer vertical line means the clusters are more different.")
                        
                        linkage_data = shc.linkage(X_scaled, method='ward')

                        fig, ax = plt.subplots(figsize=(10, 7))
                        shc.dendrogram(linkage_data, labels=driver_names, leaf_rotation=90, ax=ax)
                        plt.title("Driver Performance Dendrogram")
                        plt.ylabel("Cluster Distance")
                        st.pyplot(fig)
                        
                    # --- DBSCAN ---
                    elif cluster_method == "DBSCAN":
                        st.subheader("DBSCAN Clustering")
                        st.write("This density-based model finds clusters and identifies 'noise' points (outliers).")
                        st.info("Cluster -1 represents 'Noise' - drivers who don't fit in any group.")

                        eps = st.slider("Select Epsilon (eps) - neighborhood distance", 0.1, 5.0, 1.5, 0.1)
                        min_samples = st.slider("Select Minimum Samples - core point density", 1, 10, 2)

                        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
                        driver_stats_clean['Cluster'] = db.labels_
                        driver_stats_clean['Cluster'] = driver_stats_clean['Cluster'].astype(str)
                        
                        n_clusters_found = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                        n_noise = list(db.labels_).count(-1)
                        st.metric("Estimated Clusters Found", n_clusters_found)
                        st.metric("Outlier (Noise) Drivers Found", n_noise)

                        fig_dbscan = px.scatter(driver_stats_clean, 
                                                x='PC1', y='PC2', 
                                                color='Cluster',
                                                hover_name='driver_name', 
                                                title=f"DBSCAN Clusters (eps={eps}, min_samples={min_samples})")
                        st.plotly_chart(fig_dbscan, use_container_width=True)


        # --- FEATURE 6: Predictive Model (MOVED & MODIFIED) ---
        with tab6:
            st.header("📈 Predictive Model: Race Points")

            # --- NEW SECTION: Point System ---
            st.subheader("Official F1 Points System")
            st.markdown("This is the standard points system the model is trying to predict.")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P1", "25 pts")
            c2.metric("P2", "18 pts")
            c3.metric("P3", "15 pts")
            c4.metric("P4", "12 pts")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("P5", "10 pts")
            c6.metric("P6", "8 pts")
            c7.metric("P7", "6 pts")
            c8.metric("P8", "4 pts")
            
            c9, c10, c11, c12 = st.columns(4)
            c9.metric("P9", "2 pts")
            c10.metric("P10", "1 pt")
            c11.metric("P11+", "0 pts")
            # --- END NEW SECTION ---

            st.markdown("---") # Add a separator
            
            st.write("A simple Linear Regression model to predict points based on grid position.")
            
            model_data = df_filtered[['grid', 'points']].dropna()
            
            if model_data.shape[0] < 10:
                st.warning("Not enough data to build a reliable regression model.")
            else:
                X = model_data[['grid']]
                y = model_data['points']
                
                model = LinearRegression()
                model.fit(X, y)
                
                coef = model.coef_[0]
                intercept = model.intercept_
                
                st.write(f"**Model Formula:** `Predicted Points = {coef:.2f} * (Grid Position) + {intercept:.2f}`")
                st.write(f"This model suggests that, on average, each position further back on the grid costs **{-coef:.2f} points**.")

                fig_reg = px.scatter(model_data, x='grid', y='points', 
                                    title="Grid Position vs. Points Scored",
                                    opacity=0.3, 
                                    trendline='ols',
                                    trendline_color_override='red')
                st.plotly_chart(fig_reg, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Make a Prediction")
                grid_input = st.slider("Select a Grid Position to predict points:", 1, 24, 10)
                
                predicted_points = model.predict([[grid_input]])[0]
                predicted_points = max(0, predicted_points)
                
                st.metric(f"Predicted Points for P{grid_input}", f"{predicted_points:.2f}")

        # --- FEATURE 7: Outlier Detection ---
        with tab7:
            st.header("⚠ Outlier Detection (Performance Deviations)")
            st.write(f"Find races in {year_display} where a driver's performance was significantly different from their average.")

            all_drivers_outlier = sorted(df_filtered['driver_name'].dropna().unique())
            
            if not all_drivers_outlier:
                st.warning("No driver data to analyze.")
            else:
                driver_outlier = st.selectbox("Select Driver for Outlier Analysis", all_drivers_outlier, key="outlier_driver")
                df_d = df_filtered[df_filtered['driver_name'] == driver_outlier].copy()
                
                df_d['position'] = pd.to_numeric(df_d['position'], errors='coerce')
                df_d = df_d.dropna(subset=['position'])

                if df_d.shape[0] < 3: 
                    st.warning(f"Not enough race data for {driver_outlier} to calculate outliers.")
                else:
                    mean_pos = df_d['position'].mean()
                    std_pos = df_d['position'].std()
                    
                    if std_pos == 0:
                        st.success(f"{driver_outlier} was perfectly consistent! (Average finish: {mean_pos:.0f})")
                    else:
                        threshold = 1.5 * std_pos
                        
                        outliers = df_d[
                            (df_d['position'] > (mean_pos + threshold)) | 
                            (df_d['position'] < (mean_pos - threshold))
                        ]
                        
                        st.write(f"**Analysis for {driver_outlier} in {year_display}:**")
                        st.metric(f"Average Finishing Position", f"{mean_pos:.2f}")
                        st.metric(f"Standard Deviation", f"{std_pos:.2f}")
                        st.info(f"An 'outlier' is a finish worse than {mean_pos + threshold:.2f} or better than {mean_pos - threshold:.2f}.")

                        st.subheader("Outlier Races Found")
                        if outliers.empty:
                            st.success("No significant outliers found. This driver was very consistent!")
                        else:
                            st.dataframe(outliers[['race_name', 'grid', 'position', 'status', 'points']])

        # --- FEATURE 8: Correlation ---
        with tab8:
            st.header("🔍 Correlation Analysis")
            st.write("Find relationships between different numeric features.")
            
            corr_features = ['grid', 'position', 'points', 'laps', 
                             'milliseconds', 'fastestLapSpeed', 'is_win', 'is_podium', 'is_dnf']
            
            existing_corr_features = [col for col in corr_features if col in df_filtered.columns]
            
            if len(existing_corr_features) > 1:
                corr_matrix = df_filtered[existing_corr_features].corr()
                
                fig_heatmap = px.imshow(corr_matrix,
                                        text_auto=True, 
                                        aspect="auto",
                                        color_continuous_scale='RdBu_r', 
                                        zmin=-1, zmax=1,
                                        title="Feature Correlation Heatmap")
                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.warning("Not enough numeric columns found to generate a heatmap.")

            st.markdown("---")
            st.header("Raw Data Viewer")
            st.write(f"Displaying data for **{year_display}** with current filters.")
            st.dataframe(df_filtered)

        # --- FEATURE 9: Circuit Map (MOVED) ---
        with tab9:
            st.header(f"Race Circuits Map ({year_display})")
            
            circuits_in_season = df_filtered[['circuitId', 'circuit_name', 'circuit_country']].drop_duplicates()
            
            map_data = circuits.merge(circuits_in_season, on='circuitId')

            if map_data.empty:
                st.warning("No circuit data to display on the map.")
            else:
                fig_map = px.scatter_geo(map_data,
                    lat='lat',
                    lon='lng',  
                    hover_name='circuit_name_x', 
                    hover_data={'circuit_country_x': True, 'lat': False, 'lng': False},
                    text='circuit_name_x',
                    projection="natural earth",
                    title=f"Race Circuits for {year_display}"
                )
                
                fig_map.update_traces(textfont_color='black', textposition='top right')
                
                st.plotly_chart(fig_map, use_container_width=True)

# Initial message if data loading fails
elif df is None:
    st.header("Dashboard could not be loaded.")
    st.write("Please check the `data/` folder and ensure all 14 required CSV files are present.")
