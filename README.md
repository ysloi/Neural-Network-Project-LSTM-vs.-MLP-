
# Neural Network Project: LSTM vs. MLP for European Call Option Pricing

Jasmine Charize Carpiz · Alyssa Siew Ching Yi · Bhakti Soni · Tiffany Vooi · Yu See Loi

---

## Overview

This project compares two neural networks, a **Multi-Layer Perceptron (MLP)** and a **Long Short-Term Memory (LSTM)** network, on the task of pricing European call options. Both models are trained on simulated call prices generated with the Black-Scholes formula, and their hyperparameters are tuned with Bayesian optimisation via [Optuna](https://optuna.org/).

**Key finding:** the MLP outperforms the LSTM by roughly 90% on test MAE and RMSE. Call prices here come from a deterministic, memoryless formula with static inputs, which is exactly the input-output mapping an MLP is built for. The LSTM's sequential machinery adds complexity without benefit on this kind of data.

## Repository Structure

| File | Description |
|---|---|
| `Call price simulation.py` | Simulates stock paths and computes Black-Scholes call prices |
| `call_price_results.csv` | Simulated call option dataset |
| `MLP.py` | MLP model with Optuna hyperparameter optimisation |
| `LSTM.py` | LSTM model with Optuna hyperparameter optimisation |
| `README.md` | Project overview (this file) |

## Getting Started

Install the dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow optuna
```

Then run the scripts in order:

```bash
python "Call price simulation.py"   # generate the dataset
python MLP.py                       # train and evaluate the MLP
python LSTM.py                      # train and evaluate the LSTM
```

> **Note:** the scripts were originally written in Google Colab. If a script loads data from a Google Drive path (e.g. `/content/drive/...`), change it to `call_price_results.csv` to run locally.

---

## Methodology

### 1. Simulating the Financial Data

- **27 million rows** of European call option prices
- 9 strike prices, expiry dates spanning 5 years, and 50,000 scenarios simulated across 12 months
- **Volatility:** modelled from the VIX, fitted to a t-distribution
- **Risk-free rate:** 3.64% (6-month average of the 1-year Treasury rate)
- **Stock prices:** simulated with Geometric Brownian Motion
- **Call prices:** calculated with the Black-Scholes formula

### 2. Standardised Setup

To keep the comparison fair, both networks share the same setup:

| Setting | Value |
|---|---|
| Train / validation / test split | 80 / 10 / 10 |
| Max epochs (tuning) | 30 |
| Max epochs (final model) | 100 |
| Optimisation trials | 50 |
| Early stopping patience | 15 |
| Optimiser | Adam |

### 3. Bayesian Optimisation via Optuna

Grid search exhausts every hyperparameter combination, but on a dataset this large it scales poorly and run times become impractical, especially for the computationally heavier LSTM. Bayesian optimisation explores the search space more efficiently, giving better error metrics with much shorter run times. It was used to tune the **number of neurons**, **learning rate** and **batch size**.

### 4. Multi-Layer Perceptron (MLP)

- ReLU activations throughout, trained with supervised learning to minimise the error between predicted and actual call prices
- Inputs and target scaled with a **RobustScaler** (centres on the median, scales by the IQR), which handles outliers common in financial data better than standard normalisation. The scaler is fitted on the training set only, to prevent data leakage.
- A random **15% sample (~4 million rows)** is used to keep processing time manageable, split into two non-overlapping pools:
  - **Hyperparameter search set (30%)**
  - **Retraining set (70%)**

  Each pool is split 80/10/10 independently. The best configuration is retrained from scratch on the retraining set, so the final evaluation reflects performance on unseen data.

### 5. Long Short-Term Memory (LSTM)

An LSTM is a type of recurrent neural network that uses forget, input and output gates to handle long-term dependencies in sequential data.

- **Inputs:** asset price (S), strike price (K), time to maturity (T), volatility (σ)
- **Output:** call price
- Inputs and target scaled to [0, 1] with a **MinMaxScaler**, which prevents gradient saturation and stops large-magnitude features (like S) overwhelming small ones (like σ)
- Trained with MSE loss and the Adam optimiser

---

## Results

### MLP

| Metric | Bayesian Optimisation |
|---|---|
| Val loss | 0.000004 |
| Test R² | 0.999998 |
| Test MAE | 0.0150 |
| Test RMSE | 0.0283 |
| Test MAPE (%) | 0.1946 |
| Hyperparameter search time (s) | 2,584 |

The MLP achieves a near-perfect R² with very low validation loss, MAE and RMSE, so its predictions deviate only minimally from the true prices and it generalises well to unseen data.

### LSTM

| Metric | Bayesian Optimisation |
|---|---|
| Val loss | 0.000149 |
| Test R² | 0.987278 |
| Test MAE | 0.134590 |
| Test RMSE | 0.209137 |
| Test MAPE (%) | 1.4486 |
| Hyperparameter search time (s) | 12,228 |

The LSTM is still a strong proxy for the Black-Scholes pricing surface. MAE and RMSE are close together, so it isn't suffering from large outlier errors, and on average predictions are off by about 0.13. The low MAPE shows the error is also small relative to the prices being predicted.

### Comparison

| Metric | MLP | LSTM |
|---|---|---|
| MAE | **0.0150** | 0.1346 |
| RMSE | **0.0283** | 0.2091 |
| R² | **0.9999** | 0.9873 |
| Search time (s) | **2,584** | 12,228 |

Both models reach an R² close to 1, but the MLP reduces MAE and RMSE by around 90% and tunes almost 5× faster.

**Why?** Simulated European call prices are not time-series data. They come from a deterministic formula with independent, static inputs. The MLP is designed for exactly this kind of memoryless input-output mapping, whereas the LSTM is optimised for sequential dependencies that aren't present here.

---

## Limitations

- **Limited tuning scope:** only neurons, learning rate and batch size were tuned. Patience, max epochs, activation function, early stopping, optimiser choice, depth and width were fixed. More robust methods such as **BOHB** could be used instead of vanilla Bayesian optimisation.
- **Partial data:** results are based on 15% of the full dataset, so performance on all 27 million rows is unknown.
- **Simulated data:** real markets have microstructure effects, jumps and volatility clustering that the simulation doesn't capture.
- **No cross-validation:** a single train/test split was used. Repeated runs gave slightly varying results (±0.02 for MAE and RMSE), so cross-validation would help confirm stability.

## Alternative Neural Networks

- **Gated Recurrent Unit (GRU):** a simplified LSTM with only update and reset gates. It addresses vanishing gradients and performs comparably to LSTMs on time-series tasks, with shorter run times, but lacks a separate memory cell. Like the LSTM, it is better suited to sequential data than to this problem.
- **Radial Basis Function Network (RBF):** similar to an MLP but with radial basis activation functions. Well suited to function approximation tasks like option pricing.
- **CNNs and GNNs:** typically used for images and graph-structured data respectively, so less suitable here.

## Conclusion

For predicting call prices from static inputs, the **MLP performs significantly better than the LSTM**. The LSTM isn't suited to this memoryless dataset, but it would likely perform better on real time-series market data, as research on datasets such as TSLA has shown.

---

## Code

- MLP (Colab): https://colab.research.google.com/drive/1Gd8wu44kfpLRCDaPdKxi1SHMsHZeDBM2?usp=sharing
- LSTM (Colab): https://colab.research.google.com/drive/1Pzo7Wukdr9wcfrOCr9FAokuE0ZMnk7n8?usp=sharing

## References

1. K. D. Brūdere, "Bayesian hyperparameter optimisation for neural networks," 2025. https://www.theseus.fi/handle/10024/890082
2. GeeksforGeeks, "Hyperparameter tuning," 2025. https://www.geeksforgeeks.org/machine-learning/hyperparameter-tuning/
3. G. Tandberg, "Financial forecasting using LSTM networks," 2015. https://wiki.ubc.ca/Course:CPSC522/Financial_Forecasting_using_LSTM_Networks
4. K. Azar, "Stock price prediction LSTM hyperparameter tuning," 2021. https://www.kaggle.com/code/kamyarazar/stock-price-prediction-lstm-hyperparameter-tuning
5. S. Abolmakarem, F. Abdi, K. Khalili-Damghani and H. Didehkhani, "A multi-stage machine learning approach for stock price prediction: Engineered and derivative indices," 2024. https://www.sciencedirect.com/science/article/pii/S2667305324001236
6. J. Starmer, "Neural networks / deep learning," 2023. https://www.youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1
7. V. Deswal and D. Kumar, "Hyperparameter tuning of the LSTM model for stock price prediction," *International Journal of Intelligent Systems and Applications in Engineering*, 2024.
8. R. Chaudhary and A. Kumar, "Advanced stock market prediction using long short-term memory networks: A comprehensive deep learning framework," 2025. https://arxiv.org/html/2505.05325v1
9. F. Hutter, L. Kotthoff and J. Vanschoren, *Automated Machine Learning: Methods, Systems, Challenges*, 2019.
10. K. E. Hoque and H. Aljamaan, "Impact of hyperparameter tuning on machine learning models in stock price forecasting," 2021. https://ieeexplore.ieee.org/abstract/document/9643011
11. J. Chung, C. Gulcehre, K. Cho and Y. Bengio, "Empirical evaluation of gated recurrent neural networks on sequence modeling," 2014. https://arxiv.org/abs/1412.3555
12. Y. Wu, H. Wang, B. Zhang and K. Du, "Using radial basis function networks for function approximation and classification," 2012. https://www.researchgate.net/publication/258403761
