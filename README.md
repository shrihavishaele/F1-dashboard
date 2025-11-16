# 🏎️ F1 Data Analytics Dashboard

An interactive Streamlit dashboard for exploring Formula 1 race data with advanced analytics and machine learning models.

## Features

- 📊 **Statistics & KPIs** - Season overviews, performance leaders, and race outcome analysis
- 👥 **Driver Comparison** - Head-to-head driver statistics and finishing consistency
- 🏆 **Win Gauge** - Visual win rate indicators for drivers
- 📖 **Association Rules** - Apriori algorithm for discovering patterns in race data
- 🤖 **PCA & Clustering** - K-Means, Hierarchical, and DBSCAN clustering of driver performance
- 📈 **Regression Model** - Linear regression for predicting race points
- ⚠️ **Outlier Detection** - Identify unusual driver performances
- 🔍 **Correlation Analysis** - Feature correlation heatmaps
- 🌍 **Circuit Map** - Geographic visualization of race circuits

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Setup Instructions

### 1. Clone or Download the Project

```bash
cd "d:\f1 streamlit"
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
```

**macOS/Linux:**
```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install packages manually:

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

**Data Source:** [Ergast F1 API Database](http://ergast.com/mrd/db/)

### 6. Run the Application

```bash
streamlit run app2.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Deactivating the Virtual Environment

When you're done working on the project:

```bash
deactivate
```

## Project Structure

```
d:\f1 streamlit\
│
├── app2.py                 # Main application file
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── data/                  # CSV data files
│   ├── results.csv
│   ├── races.csv
│   └── ...
└── venv/                  # Virtual environment (created after setup)
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

## Troubleshooting

### Virtual Environment Not Activating

- Ensure you're in the correct directory
- Check Python installation with `python --version`
- On Windows, you may need to allow script execution:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### Missing Data Files

- Verify all 14 CSV files are in the `data/` folder
- Check file names match exactly (case-sensitive)

### Import Errors

- Ensure virtual environment is activated
- Reinstall packages: `pip install -r requirements.txt --upgrade`

## Usage Tips

1. Use the sidebar filters to select year ranges, drivers, and constructors
2. Navigate between tabs to explore different analytical features
3. Adjust algorithm parameters (clusters, confidence, support) to see different insights
4. Hover over charts for detailed information

## License

This project uses publicly available F1 data from the Ergast API.

## Contact

For issues or questions, please create an issue in the project repository.
