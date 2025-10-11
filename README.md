#  Stock Market Analysis Using Data Mining and AI Techniques

##  Overview
This project presents a **comprehensive stock market analysis framework** that integrates **machine learning**, **deep learning**, and **reinforcement learning** to enhance investment strategies and trading decisions.  
It combines predictive modeling, anomaly detection, and portfolio optimization techniques using historical market data for **30 major stocks** over **1.5 years (Sep 2022 – Feb 2024)**.

The system bridges the gap between historical trend analysis and real-time decision support, enabling investors to make **data-driven, adaptive, and risk-optimized** trading decisions.  



##  Objectives
- Develop robust models for **stock price prediction** and **market behavior analysis**.  
- Apply **data mining** and **AI-driven models** to enhance forecasting reliability.  
- Identify **hidden correlations** between stocks to improve portfolio diversification.  
- Detect **abnormal trading behaviors** using statistical and AI-based anomaly detection.  
- Design a **reinforcement learning trading agent** to suggest real-time buy/sell/hold actions.

---

##  Project Structure
```
.
├── data/
│   ├── raw/                     # Original Yahoo Finance data
│   ├── processed/               # Cleaned & feature-engineered data
├── notebooks/
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Modeling_LSTM_RL.ipynb
│   └── 04_Visualization_Insights.ipynb
├── src/
│   ├── data_prep.py             # Cleaning, scaling, sequence creation
│   ├── feature_engineering.py   # RSI, MACD, EMA, volatility, returns
│   ├── portfolio_opt.py         # Modern Portfolio Theory (MPT)
│   ├── anomaly_detect.py        # Z-Score & Isolation Forest methods
│   ├── lstm_predictor.py        # LSTM next-day price prediction
│   ├── correlation_analysis.py  # Heatmaps & hierarchical clustering
│   ├── rl_trader.py             # Deep Q-Learning trading agent
│   └── utils.py
├── models/                      # Saved models (.h5, .pkl)
├── reports/                     # Plots, metrics, results
├── train_lstm.py                # Runnable LSTM demo script
├── requirements.txt
└── README.md
```



## Dataset

**Source:**  
- Yahoo Finance API (via `yfinance`) for 30 major stocks (AAPL, NVDA, META, AMZN, COST, etc.).  
- SEC company listings for ticker verification.  

**Features:**  
- Date, Open, High, Low, Close, Volume  
- Engineered Indicators: RSI, MACD, EMA, Daily Returns, Volatility  
- Period Covered: *September 2022 – February 2024*  

**Preprocessing Steps:**  
- Handled missing values with **forward-fill**.  
- Removed duplicates and anomalies.  
- Applied **Min-Max normalization** and **time-windowing** for LSTM models.  



##  Methodology

### 1️⃣ Data Collection & Cleaning
- Fetched Yahoo Finance historical data using API calls.  
- Cleaned inconsistencies and normalized values for comparability.  

### 2️⃣ Feature Engineering
- Calculated **technical indicators**: RSI, MACD, EMA, volatility, daily returns.  
- Constructed rolling statistics for anomaly detection.  

### 3️⃣ Modeling Approaches
| Model | Purpose | Technique |
|--------|----------|-----------|
| **MPT Portfolio Optimizer** | Optimal allocation to maximize returns/minimize risk | Modern Portfolio Theory |
| **Anomaly Detector** | Detect suspicious trading spikes | Rolling Z-Score & Volume Spike |
| **LSTM Predictor** | Next-day price prediction | Deep Learning (Time-series) |
| **Correlation Mapper** | Group similar stocks | Pearson correlation + Hierarchical clustering |
| **RL Action Model** | Real-time Buy/Sell/Hold decisions | Deep Q-Learning Reinforcement Learning |

### 4️⃣ Evaluation Metrics
- **RMSE** – for price prediction accuracy  
- **Sharpe Ratio** – for portfolio risk-return balance  
- **Z-Score thresholds** – for anomaly detection sensitivity  
- **Backtesting cumulative profit** – for RL performance  



##  Results Summary
| Model | Key Findings |
|--------|---------------|
| **MPT Portfolio Optimization** | Optimal portfolio: NVDA, NVO, COST, META, SAP achieved high Sharpe Ratio. |
| **Anomaly Detection** | Detected abnormal volume spikes (e.g., META & TSLA) indicating possible insider activity. |
| **LSTM Forecasting** | Achieved RMSE < $1.50 for 20+ stocks; strong accuracy on stable equities. |
| **Hidden Correlation Discovery** | Clustered stocks by sector; e.g., AAPL–MSFT (high correlation), XOM–LLY (low). |
| **RL Agent (Deep Q-Learning)** | Generated adaptive Buy/Sell recommendations and profitable backtests. |



##  Tools & Technologies
- **Language:** Python 3.10  
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `yfinance`, `tensorflow`, `keras`, `gym`, `scipy`  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Modeling Frameworks:** Scikit-learn, TensorFlow/Keras, Stable-Baselines3  



##  How to Run
```bash
# Clone the repository
git clone https://github.com/your-username/stock-market-ai.git
cd stock-market-ai

# (Optional) Create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run notebooks or scripts
jupyter notebook notebooks/01_Data_Exploration.ipynb
```



##  Insights
- LSTM models capture **short-term temporal dependencies** effectively.  
- Portfolio optimization balances **risk vs. reward** better than manual selection.  
- Reinforcement learning introduces **adaptive, self-improving** trade strategies.  
- Correlation discovery enhances **diversification and sector exposure management**.  



##  Future Enhancements
- Integrate **news & sentiment analysis** (NLP).  
- Implement **real-time dashboards** for traders.  
- Extend reinforcement learning to **PPO/Actor-Critic** architectures.  
- Incorporate **Graph Neural Networks (GNNs)** for inter-stock dependency learning.  
- Address **fairness, transparency, and explainability** in AI-based trading.  



##  Acknowledgements
Special thanks to  
**Prof. Sumona Mondal and Prof. Naveen Reddy**  
for their continuous mentorship, feedback, and guidance throughout the project.  



## 📚 References
1. Chen Z. (2024). *Portfolio Optimization Model Based on Machine Learning*.  
2. Du S., Shen H. (2024). *Reinforcement Learning-Based Multimodal Model for Portfolio Management*.  
3. Sen J. et al. (2021). *Stock Portfolio Optimization Using LSTM Models*.  
4. Fischer T., Krauss C. (2018). *Deep Learning with LSTM for Financial Market Prediction*.  
5. Jiang Z. et al. (2017). *Deep Reinforcement Learning for Portfolio Management*.  
