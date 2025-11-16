# 🏎️ F1 Data Analytics Dashboard

An interactive Streamlit dashboard for exploring Formula 1 race data with advanced analytics and machine learning models.

## Features

-  **Statistics & KPIs** - Season overviews, performance leaders, and race outcome analysis
-  **Driver Comparison** - Head-to-head driver statistics and finishing consistency
-  **Win Gauge** - Visual win rate indicators for drivers
-  **Association Rules** - Apriori algorithm for discovering patterns in race data
-  **PCA & Clustering** - K-Means, Hierarchical, and DBSCAN clustering of driver performance
-  **Regression Model** - Linear regression for predicting race points
-  **Outlier Detection** - Identify unusual driver performances
-  **Correlation Analysis** - Feature correlation heatmaps
-  **Circuit Map** - Geographic visualization of race circuits

## Prerequisites

- Python 
- pip (Python package installer)

## Setup Instructions

### 1. Clone or Download the Project

### 2. Create a Virtual Environment

### 3. Activate the Virtual Environment

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

or just

```bash
pip install streamlit pandas numpy plotly scikit-learn mlxtend scipy matplotlib
```

### 5. Prepare Data

Ensure you have a `data/` folder in the project directory containing the following CSV files:

- `results.csv`
- `races.csv`
- `status.csv`
- `drivers.csv`
- `constructors.csv`
- `circuits.csv`
- `sprint_results.csv`
- `driver_standings.csv`
- `pit_stops.csv`
- `lap_times.csv`
- `qualifying.csv`
- `constructor_standings.csv`
- `constructor_results.csv`
- `seasons.csv`


### 6. Run the Application

```bash
streamlit run app2.py
```

## Dependencies

- **streamlit** - Web application framework
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **plotly** - Interactive visualizations
- **scikit-learn** - Machine learning algorithms
- **mlxtend** - Apriori association rule mining
- **scipy** - Scientific computing
- **matplotlib** - Additional plotting capabilities

