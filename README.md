
# Donchian Channel Breakout Strategy

This project implements and evaluates a long-only **Donchian Channel breakout strategy** on SPY over the last **1,000 trading days**.

## 📌 Strategy Overview

- **Entry Rule**: Buy when price breaks above the 20-day high.
- **Exit Rule**: Exit when price drops below the 10-day low.
- **Risk Management**:
  - Evaluates drawdowns, Sharpe, Sortino, and Calmar ratios
  - Includes transaction cost simulation
  - Benchmarks performance against SPY (Buy & Hold)

## 📊 Results Summary

| Metric            | Donchian Strategy | Buy & Hold SPY |
|------------------|-------------------|----------------|
| Annual Return     | 8.54%             | 10.98%         |
| Annual Volatility | 9.67%             | 18.15%         |
| Sharpe Ratio      | 0.88              | 0.60           |
| Max Drawdown      | -8.31%            | -24.50%        |

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 🧠 Author

**Tapesh Awasthi**  
MS Quantitative Finance, Fordham University  
[LinkedIn](https://www.linkedin.com/in/tapeshawasthi/)  
[Email](mailto:ta61@fordham.edu)
