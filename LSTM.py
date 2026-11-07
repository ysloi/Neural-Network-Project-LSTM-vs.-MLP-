import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import time
import warnings
import gc
warnings.filterwarnings('ignore')

# Suppress Optuna logs (optional)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)

print("=" * 80)
print("LSTM MODEL FOR CALL PRICE PREDICTION")
print("Bayesian Optimization with Optuna")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA FROM CSV IN CHUNKS
# ============================================================================

print("\n[STEP 1] Loading Data from CSV")
print("-" * 80)

print("Loading call_price_results.csv in chunks...")

target_sample_size = 300000
chunks_to_sample = []

# Read CSV in chunks
chunk_iterator = pd.read_csv('/content/drive/MyDrive/call_price_results.csv', chunksize=50000)

for i, chunk in enumerate(chunk_iterator):
    sampled_chunk = chunk.sample(frac=0.01, random_state=42)
    chunks_to_sample.append(sampled_chunk)

    if (i + 1) % 10 == 0:
        current_total = sum(len(c) for c in chunks_to_sample)
        print(f"  Processed {(i+1)*50000:,} rows → Sampled {current_total:,}")

    if sum(len(c) for c in chunks_to_sample) >= target_sample_size:
        break

data = pd.concat(chunks_to_sample, ignore_index=True)
data = data.sample(n=min(target_sample_size, len(data)), random_state=42)

print(f"\n✓ Loaded {len(data):,} rows")
print(f"✓ Memory usage: {data.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

del chunks_to_sample
gc.collect()

# ============================================================================
# STEP 2: DATA PREPARATION
# ============================================================================

print("[STEP 2] Data Preparation")
print("-" * 80)

feature_columns = ['S', 'K', 'T', 'sigma']
target_column = 'call_price'

X = data[feature_columns].values
y = data[target_column].values

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Features: {feature_columns}\n")

# ============================================================================
# STEP 3: DATA SCALING
# ============================================================================

print("[STEP 3] Data Scaling")
print("-" * 80)

feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X)

target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

print("Data scaling completed")
print(f"Feature range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
print(f"Target range: [{y_scaled.min():.4f}, {y_scaled.max():.4f}]\n")

del X, y, data
gc.collect()

# ============================================================================
# STEP 4: TRAIN/VAL/TEST SPLIT (80/10/10)
# ============================================================================

print("[STEP 4] Data Splitting (80/10/10)")
print("-" * 80)

X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.1, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=1/9, random_state=42
)

print(f"Training set:   {X_train.shape[0]:,} samples (80%)")
print(f"Validation set: {X_val.shape[0]:,} samples (10%)")
print(f"Test set:       {X_test.shape[0]:,} samples (10%)")

# Reshape for LSTM: (samples, timesteps, features)
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm   = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_lstm  = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"\nReshaped for LSTM: {X_train_lstm.shape}\n")

# FIX: Only delete the unreshaped X arrays — keep y_train, y_val, y_test
# as they are needed inside the objective function and final evaluation
del X_temp, X_scaled, X_train, X_val, X_test
gc.collect()

# ============================================================================
# STEP 5: DEFINE OBJECTIVE FUNCTION FOR OPTUNA
# ============================================================================

print("[STEP 5] Setting Up Bayesian Optimization")
print("-" * 80)

def create_lstm_model(neurons, learning_rate, dropout_rate, input_shape):
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(neurons, input_shape=input_shape, return_sequences=False),
        Dropout(dropout_rate),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def evaluate_model(y_true, y_pred, scaler):
    """Calculate evaluation metrics (inverse-scaled)"""
    y_true_orig = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    mse  = mean_squared_error(y_true_orig, y_pred_orig)
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    r2   = r2_score(y_true_orig, y_pred_orig)
    return mse, mae, r2

all_trials_results = []

def objective(trial):
    """Objective function for Optuna optimization"""

    neurons       = trial.suggest_categorical('neurons',       [64, 128, 256, 512])
    batch_size    = trial.suggest_categorical('batch_size',    [128, 256, 512])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout_rate  = trial.suggest_float('dropout_rate', 0.05, 0.3)

    print(f"\n[Trial {trial.number + 1}] Testing:")
    print(f"  Neurons: {neurons}, Batch: {batch_size}, "
          f"LR: {learning_rate:.6f}, Dropout: {dropout_rate:.3f}")

    model = create_lstm_model(
        neurons=neurons,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        X_train_lstm, y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=30,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    y_val_pred  = model.predict(X_val_lstm,  verbose=0).flatten()
    y_test_pred = model.predict(X_test_lstm, verbose=0).flatten()

    val_mse,  val_mae,  val_r2  = evaluate_model(y_val,  y_val_pred,  target_scaler)
    test_mse, test_mae, test_r2 = evaluate_model(y_test, y_test_pred, target_scaler)

    trial_result = {
        'trial':         trial.number + 1,
        'neurons':       neurons,
        'batch_size':    batch_size,
        'learning_rate': learning_rate,
        'dropout_rate':  dropout_rate,
        'val_mse':       val_mse,
        'val_mae':       val_mae,
        'val_r2':        val_r2,
        'test_mse':      test_mse,
        'test_mae':      test_mae,
        'test_r2':       test_r2,
        'epochs':        len(history.history['loss'])
    }
    all_trials_results.append(trial_result)

    print(f"  → Val  R²: {val_r2:.6f}, MSE: {val_mse:.6e}, MAE: {val_mae:.6f}")
    print(f"  → Test R²: {test_r2:.6f}, Epochs: {len(history.history['loss'])}")

    pd.DataFrame(all_trials_results).to_csv('/content/drive/MyDrive/bayesian_optimization_progress.csv', index=False)

    keras.backend.clear_session()
    gc.collect()

    return val_r2  # Optuna maximises this

# ============================================================================
# STEP 6: RUN BAYESIAN OPTIMIZATION
# ============================================================================

print("[STEP 6] Running Bayesian Optimization")
print("-" * 80)

N_TRIALS = 50

print(f"\nOptuna will intelligently search {N_TRIALS} hyperparameter combinations")
print("Using TPE (Tree-structured Parzen Estimator) sampler")
print("This is much more efficient than grid search!\n")

study = optuna.create_study(
    study_name='lstm_call_price_optimization',
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
)

print("Starting optimization...")
print("=" * 80)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ============================================================================
# STEP 7: BEST TRIAL RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION COMPLETE")
print("=" * 80)

best_trial = study.best_trial
print(f"\nBest Trial: #{best_trial.number + 1}")
print(f"  Best Val R²: {study.best_value:.6f}")
print(f"  Best Params:")
for k, v in best_trial.params.items():
    print(f"    {k}: {v}")

# ============================================================================
# STEP 8: RETRAIN BEST MODEL ON FULL TRAIN+VAL DATA
# ============================================================================

print("\n" + "=" * 80)
print("Retraining best configuration on full training data")
print("=" * 80)

best_params = best_trial.params

# Combine train + val for final training
X_trainval_lstm = np.concatenate([X_train_lstm, X_val_lstm], axis=0)
y_trainval      = np.concatenate([y_train,      y_val],      axis=0)

best_model = create_lstm_model(
    neurons=best_params['neurons'],
    learning_rate=best_params['learning_rate'],
    dropout_rate=best_params['dropout_rate'],
    input_shape=(X_trainval_lstm.shape[1], X_trainval_lstm.shape[2])
)

early_stop_final = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Use 10% of trainval as a small hold-out just for early stopping
X_ft, X_fv, y_ft, y_fv = train_test_split(
    X_trainval_lstm, y_trainval, test_size=0.1, random_state=42
)

history_final = best_model.fit(
    X_ft, y_ft,
    validation_data=(X_fv, y_fv),
    epochs=100,
    batch_size=best_params['batch_size'],
    callbacks=[early_stop_final],
    verbose=1
)

best_model.save('/content/drive/MyDrive/best_lstm_model.keras')
print("\n✓ Best model saved as 'best_lstm_model.keras'")

# ============================================================================
# STEP 9: FINAL EVALUATION ON TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("Final Evaluation on Test Set")
print("=" * 80)

y_test_pred_final = best_model.predict(X_test_lstm, verbose=0).flatten()
test_mse, test_mae, test_r2 = evaluate_model(y_test, y_test_pred_final, target_scaler)

# Also compute RMSE on original scale
y_test_orig      = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_test_pred_orig = target_scaler.inverse_transform(y_test_pred_final.reshape(-1, 1)).flatten()
test_rmse = np.sqrt(test_mse)

print(f"\n  Test MSE:  {test_mse:.6e}")
print(f"  Test RMSE: {test_rmse:.6f}")
print(f"  Test MAE:  {test_mae:.6f}")
print(f"  Test R²:   {test_r2:.6f}")

# ============================================================================
# STEP 10: SAVE FULL RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("Saving Results")
print("=" * 80)

results_df = pd.DataFrame(all_trials_results)
results_df.to_csv('bayesian_optimization_progress.csv', index=False)
print("✓ All trial results saved to 'bayesian_optimization_progress.csv'")

summary = {
    'best_trial':      best_trial.number + 1,
    'best_val_r2':     study.best_value,
    'final_test_mse':  test_mse,
    'final_test_rmse': test_rmse,
    'final_test_mae':  test_mae,
    'final_test_r2':   test_r2,
    **best_params
}
pd.DataFrame([summary]).to_csv('best_model_summary.csv', index=False)
print("✓ Best model summary saved to 'best_model_summary.csv'")

print("\n" + "=" * 80)
print("ALL DONE")
print("=" * 80)


import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# RESULTS & VISUALISATION
# ============================================================================

# Get predictions on test set (inverse scaled)
y_test_pred_plot = best_model.predict(X_test_lstm, verbose=0).flatten()
actuals = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
preds   = target_scaler.inverse_transform(y_test_pred_plot.reshape(-1, 1)).flatten()

# Metrics
mae  = np.mean(np.abs(preds - actuals))
rmse = np.sqrt(np.mean((preds - actuals) ** 2))
print(f"Test MAE:  {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

y_val_pred = best_model.predict(X_val_lstm, verbose=0).flatten()
val_loss = np.mean((y_val - y_val_pred) ** 2)
print(f"Final Validation Loss (MSE): {val_loss:.6e}")

y_test_pred = best_model.predict(X_test_lstm, verbose=0).flatten()
y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_orig = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

test_r2 = r2_score(y_test_orig, y_pred_orig)
print(f"Final Test R²:              {test_r2:.6f}")

mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100
print(f"Final Test MAPE: {mape:.4f}%")

t0 = time.time()
study.optimize(objective, n_trials=50, show_progress_bar=True)
search_time = time.time() - t0

print(f"\nSearch time: {search_time/60:.1f} minutes")
print("\n" + "=" * 80)
print("BEST HYPERPARAMETERS")
print("=" * 80)
print(f"  Best Val R2: {study.best_value:.6f}")
for k, v in study.best_trial.params.items():
    print(f"  {k}: {v}")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Training & Validation Loss
axes[0].plot(history_final.history['loss'],     label='Train')
axes[0].plot(history_final.history['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()

# Plot 2: Predicted vs Actual
axes[1].scatter(actuals, preds, alpha=0.3, s=10)
axes[1].plot([actuals.min(), actuals.max()],
             [actuals.min(), actuals.max()], 'r--')
axes[1].set_xlabel('Actual Call Price')
axes[1].set_ylabel('Predicted Call Price')
axes[1].set_title('Predicted vs Actual')

plt.tight_layout()
plt.savefig('results_lstm_optuna.png', dpi=150)
plt.show()

# Optuna visualisations
try:
    from optuna.visualization import plot_optimization_history, plot_param_importances
    fig1 = plot_optimization_history(study)
    fig1.show()
    fig2 = plot_param_importances(study)
    fig2.show()
except ImportError:
    print("Install plotly for Optuna visualizations: pip install plotly")