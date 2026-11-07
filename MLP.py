import optuna
import itertools
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import random

SEED = 123

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 1. Loading & preparing the simulated data
df_full = results

  # Using 15% of total data for the model training
df_total  = df_full.sample(frac=0.15, random_state=SEED)
df_search = df_total.sample(frac=0.3, random_state=SEED)
df_full10 = df_total.drop(df_search.index)

features = ["S", "K", "T", "sigma"]
target   = "call_price"

def make_splits(df):
    X = df[features].values
    y = df[target].values.reshape(-1, 1)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_v,  X_te, y_v,  y_te  = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED)
    sx, sy = RobustScaler(), RobustScaler()
    X_tr = sx.fit_transform(X_tr);  X_v = sx.transform(X_v);  X_te = sx.transform(X_te)
    y_tr = sy.fit_transform(y_tr);  y_v = sy.transform(y_v);  y_te = sy.transform(y_te)
    def tt(X, y):
        return TensorDataset(torch.tensor(X, dtype=torch.float32),
                             torch.tensor(y, dtype=torch.float32))
    return tt(X_tr,y_tr), tt(X_v,y_v), tt(X_te,y_te), sx, sy

# Spliting into training, testing and validation sets
tr_s, va_s, te_s, sx_s, sy_s = make_splits(df_search)
tr_f, va_f, te_f, sx_f, sy_f = make_splits(df_full10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Building the model
class OptionPriceMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[256, 128]):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Training function for the model
# Training uses max_epochs = 30, patience = 15
# Final retraining uses max_epochs = 100, patience = 15.
def train_eval(hidden_dims, lr, batch_size, max_norm,
               train_ds, val_ds,
               max_epochs=30, patience=15, trial=None, save_path=None):
    model     = OptionPriceMLP(input_dim=len(features), hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn   = nn.MSELoss()
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(val_ds,   batch_size=batch_size)

    best_val, wait = float("inf"), 0
    tr_losses, va_losses = [], []

    for epoch in range(max_epochs):
        model.train()
        ep_loss = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        tr_losses.append(ep_loss / len(train_ds))

        model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += loss_fn(model(xb), yb).item() * len(xb)
        vl /= len(val_ds)
        va_losses.append(vl); scheduler.step(vl)

        if vl < best_val:
            best_val, wait = vl, 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= patience:
                break

        if trial is not None:
            trial.report(vl, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if save_path is not None:
        model.load_state_dict(torch.load(save_path))

    return best_val, model, tr_losses, va_losses

# 4. Bayesian optimisation (Optuna)
def objective(trial):
    lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1024, 2048, 4096])
    max_norm   = trial.suggest_float("max_norm", 0.5, 2.0)
    n_layers   = trial.suggest_int("n_layers", 1, 3)
    dims       = [trial.suggest_categorical(f"d{i}", [64, 128, 256, 512])
                  for i in range(n_layers)]
    val_loss, _, _, _ = train_eval(dims, lr, batch_size, max_norm,
                                tr_s, va_s,
                                max_epochs=30, patience=15,
                                trial=trial)
    return val_loss

print("=" * 60)
print("Bayesian Optimisation (Optuna, 50 trials)")
print("=" * 60)
t0 = time.time()
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
)
study.optimize(objective, n_trials=50, show_progress_bar=True)
bayes_time = time.time() - t0

bp        = study.best_params
n_best    = bp["n_layers"]
bayes_cfg = dict(
    hidden_dims = [bp[f"d{i}"] for i in range(n_best)],
    lr          = bp["lr"],
    batch_size  = bp["batch_size"],
    max_norm    = bp["max_norm"],
)
print(f"\nBayesian best val loss : {study.best_value:.6f}  ({bayes_time:.0f}s)")
print(f"  {bayes_cfg}")

# 5. Retrain on optimised hyperparameters on the full sample dataset
print("\n" + "=" * 60)
print("Retraining both optimised hyperparameters on full sample dataset …")

def retrain(cfg, save_path):
    _, model, tl, vl = train_eval(
        **cfg, train_ds=tr_f, val_ds=va_f,
        max_epochs=100, patience=15, save_path=save_path
    )
    return model, tl, vl

def test_metrics(model, scaler_y):
    model.eval()
    X_te_np = te_f.tensors[0].numpy()
    y_te_np = te_f.tensors[1].numpy()
    with torch.no_grad():
        preds = model(torch.tensor(X_te_np).to(device)).cpu().numpy()
    preds   = scaler_y.inverse_transform(preds)
    actuals = scaler_y.inverse_transform(y_te_np)
    mae  = np.mean(np.abs(preds - actuals))
    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    r2   = r2_score(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / np.where(actuals < 0.01, np.nan, actuals))) * 100
    return mae, rmse, r2, mape, preds, actuals

model, tl, vl = retrain(bayes_cfg, "best_bayes.pt")
mae, rmse, r2, mape, preds, actuals = test_metrics(model, sy_f)

# 6. Results
print("\n" + "=" * 60)
print(f"{'Metric':<25} {'Bayesian':>15}")
print("-" * 40)
print(f"{'Val Loss (search)':<25} {study.best_value:>15.6f}")
print(f"{'Test R²':<25} {r2:>15.6f}")
print(f"{'Test MAE':<25} {mae:>15.4f}")
print(f"{'Test RMSE':<25} {rmse:>15.4f}")
print(f"{'Test MAPE (%)':<25} {mape:>15.4f}")
print(f"{'Search Time (s)':<25} {bayes_time:>15.1f}")
print(f"{'Configs evaluated':<25} {'50 (adaptive)':>15}")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curves
axes[0].plot(tl, label="Train")
axes[0].plot(vl, label="Validation")
axes[0].set(xlabel="Epoch", ylabel="MSE Loss", title="Training and Validation Loss")
axes[0].legend()


axes[1].scatter(actuals, preds, alpha=0.3, s=10)
axes[1].plot([0, actuals.max()], [0, actuals.max()], "r--")
axes[1].set_xlim(left=0)
axes[1].set_ylim(bottom=0)
axes[1].set(xlabel="Actual Call Price", ylabel="Predicted Call Price",
            title=f"Predicted vs Actual")

metrics_labels = ["MAE", "RMSE", "R²"]
values = [mae, rmse, r2]
axes[2].bar(metrics_labels, values, color="steelblue")
axes[2].set(title="Performance Metrics")

plt.tight_layout()
plt.savefig("bayes_results.png", dpi=150)
plt.show()

try:
    from optuna.visualization import plot_optimization_history, plot_param_importances
    plot_optimization_history(study).show()
    plot_param_importances(study).show()
except ImportError:
    print("Install plotly for Optuna visualizations: pip install plotly")