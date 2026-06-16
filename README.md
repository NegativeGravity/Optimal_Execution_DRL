# 📈 Institutional-Grade RL Execution Agent (Smart TWAP Manager)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-orange.svg)
![Stable Baselines 3](https://img.shields.io/badge/RL_Library-Stable_Baselines3-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)

> An advanced Deep Reinforcement Learning (DRL) system designed to execute large block trades optimally. Instead of replacing traditional algorithms, this AI acts as a **Smart Manager for TWAP**, dynamically adjusting execution speed to minimize market impact (Slippage) while managing inventory risk.

---

## 💡 The Problem: Execution Dilemma
When liquidating a large institutional position (e.g., 100 BTC) within a fixed time horizon (e.g., 300 seconds), traders face two conflicting enemies:
1. **Market Impact (Slippage):** Selling too fast sweeps the Order Book, destroying the execution price.
2. **Inventory Risk:** Selling too slow exposes the portfolio to sudden adverse market volatility.

## 🚀 The Solution: AI as a Strategy Manager
Traditional RL models fail because they try to guess raw execution percentages, leading to erratic behavior (Overtrading). This project introduces a paradigm shift:
**The agent does not predict volume directly. It predicts a multiplier `[-1.0, 1.0]` that modulates a base TWAP strategy.**

* Action `-1.0` ➡️ Slow down TWAP by 50% (Brake during low liquidity).
* Action `0.0` ➡️ Execute pure TWAP (Cruise control).
* Action `+1.0` ➡️ Speed up TWAP by 50% (Accelerate during high liquidity).

---

## 🧠 Core Architecture & Innovations

### 1. Market Physics Simulator (`BaseExecutionEnv`)
* **Walking the Book Approximation:** Combines top-of-book (`L0`) and deep-book (`L1`) liquidity to realistically simulate price slippage when large orders eat through the bids.
* **Continuous State Space:** Normalizes limit order book (LOB) data, time decay, and inventory tracking into a stable `np.float32` vector.

### 2. Almgren-Chriss Exponential Reward (`RewardWrapper`)
Amateur models use linear penalties. We use mathematical models standard in Wall Street:
* **Step Cost:** Penalized purely by basis points (bps) lost to spread and impact.
* **Exponential Inventory Risk:** Penalty scales quadratically `(inventory ** 2) * 5.0` to force institutional urgency.
* **Terminal Penalty:** A massive `(inventory ** 2) * 5000.0` penalty at $T_{end}$ to guarantee a 100% completion rate.

### 3. Anti-Overfitting Regime (`RandomEpisodeWrapper`)
RL agents memorize 5-minute charts. Our wrapper dynamically samples random 300-second regimes from the entire historical dataset at the start of every episode, forcing the agent to learn true market dynamics, not just memorize history.

---

## 📊 Performance Benchmarks (Validation Set)

Evaluated under strict unseen validation conditions, comparing traditional algorithms against our trained **PPO** and **SAC** continuous agents.

| Method | Mean Cost (bps) | Cost Std Dev (Risk) | Completion | Smoothness | Persona / Style |
|:---|:---|:---|:---|:---|:---|
| **DQN** | 93.74 ❌ | High | 100% | Erratic | *The Nervous Novice* (Too discrete) |
| **TWAP** | 4.29 ⏱️ | 10.51 | 100% | Baseline | *The Train* (Blind but steady) |
| **PPO** | 4.83 🛡️ | **10.04** | 100% | 0.082 | *The Off-Road Driver* (Safe & Reactive) |
| **SAC** | **4.15** 🏆 | **9.80** | 100% | **0.021** | *The VIP Chauffeur* (Smooth & Optimal)|

> **Winner:** `SAC (Soft Actor-Critic)` utilizes off-policy experience replay and maximum entropy learning to beat TWAP in raw cost while significantly reducing execution variance (risk).

---

## 📈 Visualizing the Intelligence (Risk Manager Dashboard)
The project includes a 3-panel institutional dashboard for post-trade analysis:
1. **Execution vs. Midpoint:** Scatter plots showing exactly where the AI chose to strike.
2. **Inventory Decay Race:** Visualizing the AI's inventory curve wrapping around the linear TWAP baseline.
3. **Action Multiplier:** A clear bar chart showing where the AI applied the brakes or pressed the gas.

*(Add your dashboard screenshot here: `![Execution Dashboard](assets/dashboard.png)`)*

---

### Prerequisites
* Python 3.10+
* Stable-Baselines3
* Gymnasium
* Pandas, NumPy, Matplotlib
