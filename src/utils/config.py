"""
Central configuration file for all hyperparameters, paths, and constants.
"""
from pathlib import Path
import torch

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Dataset-specific paths
BATTERYLIFE_DIR = RAW_DIR / "batterylife"
STANFORD_DIR = RAW_DIR / "stanford"
NASA_DIR = RAW_DIR / "nasa"
CALCE_DIR = RAW_DIR / "calce"

# ============================================================
# DEVICE CONFIGURATION (Apple M4 Pro)
# ============================================================
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ============================================================
# RANDOM SEEDS (Reproducibility)
# ============================================================
RANDOM_SEED = 42

# ============================================================
# DATA PREPROCESSING
# ============================================================
TEST_SIZE = 0.2
MISSING_THRESHOLD = 0.2  # Drop battery if >20% values missing
OUTLIER_ZSCORE = 3.0  # Z-score threshold for outlier flagging

# Voltage range (chemistry-dependent, general bounds)
VOLTAGE_MIN = 2.0  # V
VOLTAGE_MAX = 4.5  # V
TEMPERATURE_MIN = -10  # Celsius
TEMPERATURE_MAX = 60  # Celsius

# ============================================================
# GRADE CLASSIFICATION THRESHOLDS
# ============================================================
GRADE_THRESHOLDS = {
    "A": 80,  # SoH > 80% → Grid-scale ESS
    "B": 60,  # 60% < SoH <= 80% → Home/Small ESS
    "C": 40,  # 40% < SoH <= 60% → Component reuse
    "D": 0,   # SoH <= 40% → Direct recycling
}

GRADE_ROUTES = {
    "A": "Grid-scale ESS (Second-life)",
    "B": "Home/Distributed ESS",
    "C": "Component/Module Reuse",
    "D": "Direct Recycling",
}

# ============================================================
# VAE HYPERPARAMETERS
# ============================================================
VAE_CONFIG = {
    "latent_dim": 12,
    "hidden_dims": [128, 64, 32],
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,
    "beta": 1.0,  # KL divergence weight
    "anomaly_percentile": 95,  # Top 5% reconstruction error = anomaly
}

# ============================================================
# XGBOOST HYPERPARAMETERS (Initial — Optuna will tune)
# ============================================================
XGBOOST_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",  # Optimal for Apple Silicon
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Optuna tuning
OPTUNA_N_TRIALS = 100
OPTUNA_CV_FOLDS = 5

# ============================================================
# LSTM HYPERPARAMETERS
# Iteration 2: regularization recipe anchored in literature.
#   hidden_size 128 → 64        (Chen 2023 Energy ablation: 64 best at our scale)
#   dropout     0.2 → 0.35      (Tang 2023 RESS, with recurrent component)
#   Adam → AdamW(weight_decay=1e-4) (Hong 2024 Applied Energy)
#   patience    15  → 8         (Severson 2019 = 8, Hong 2024 = 10)
#   seq_len     30  → 60        (Hong 2024 ablation: 100 best, 30 worst by 30 % RMSE)
# ============================================================
LSTM_CONFIG = {
    "input_size": None,         # set dynamically based on feature count
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.35,
    "weight_decay": 1e-4,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 200,
    "patience": 8,
    "sequence_length": 60,
    "scheduler": "cosine",
}

# ============================================================
# TCN HYPERPARAMETERS (Iter-3 §3.13 secondary DL baseline)
# Bai et al. 2018 (arXiv:1803.01271) reference architecture
# - num_channels [32,32,64,64] gives 4 blocks, dilations 1,2,4,8
# - kernel=3 + dilation 8 → receptive field = 1 + 2*(3-1)*(2^4-1) = 61 cycles
#   (just covers the 60-cycle context window from LSTM_CONFIG)
# - dropout 0.3 (Tang 2023 RESS regularization recipe; matches LSTM-Iter-2)
# - AdamW + weight_decay 1e-4 (same recipe as LSTM Iter-2)
# - patience 8 (Severson 2019 / Hong 2024 short-patience)
# ============================================================
TCN_CONFIG = {
    "input_size": None,                  # set dynamically based on feature count
    "num_channels": [32, 32, 64, 64],    # 4 TemporalBlocks
    "kernel_size": 3,
    "dropout": 0.3,
    "weight_decay": 1e-4,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 200,
    "patience": 8,
    "sequence_length": 60,               # match LSTM context window
    "scheduler": "cosine",
}

# ============================================================
# MCDM CONFIGURATION
# ============================================================
MCDM_CRITERIA = [
    "Technical Feasibility",
    "Economic Viability",
    "Environmental Impact",
    "BWMR Compliance",
    "Safety Risk",
]

MCDM_CRITERIA_TYPES = [
    "benefit",  # Technical Feasibility
    "benefit",  # Economic Viability
    "benefit",  # Environmental Impact
    "benefit",  # BWMR Compliance
    "cost",     # Safety Risk (lower is better)
]

MCDM_ALTERNATIVES = [
    "Grid-scale ESS",
    "Home/Distributed ESS",
    "Component Reuse",
    "Direct Recycling",
]

# Sensitivity analysis scenarios
# Sensitivity scenarios for the canonical 6 MCDM criteria, ordered as
# (SoH, Value, Carbon, Compliance, Safety, EPR Return). Carbon and Safety
# are cost-type in TOPSIS and inverted at scoring time, so the weights here
# are simply the *importance* attached to each criterion (sum to 1).
#
# BWMR-Heavy and EU-Heavy operationalise RQ2 (BWMR vs EU regulatory regimes).
SENSITIVITY_SCENARIOS = {
    "Equal Weights":     [1/6,  1/6,  1/6,  1/6,  1/6,  1/6],
    "Literature Mean":   None,                         # computed from fuzzy_bwm_input.csv
    "Technical-Heavy":   [0.30, 0.25, 0.10, 0.15, 0.10, 0.10],
    "BWMR-Heavy":        [0.05, 0.10, 0.10, 0.40, 0.10, 0.25],   # India: Compliance + EPR Return dominant
    "EU-Heavy":          [0.10, 0.10, 0.35, 0.30, 0.10, 0.05],   # EU 2023/1542: Carbon + Compliance dominant
}

# ============================================================
# EVALUATION TARGETS
# ============================================================
TARGETS = {
    "soh_r2": 0.95,
    "soh_rmse": 2.0,  # % SoH
    "soh_mae": 1.5,   # % SoH
    "rul_rmse": 2.0,   # % RUL
    "grade_accuracy": 0.90,
    "grade_f1": 0.85,
    "bwm_consistency": 0.1,
    "topsis_stability": 0.8,  # Spearman rank correlation
    "dpp_coverage": 0.80,
}

# ============================================================
# REGULATORY CONSTANTS (India BWMR + EU DPP)
# ============================================================
BWMR_RECOVERY_TARGETS = {
    "FY2024-25": 0.70,
    "FY2025-26": 0.80,
    "FY2026-27+": 0.90,
}

BWMR_RECYCLED_CONTENT = {
    "2027-28": 0.05,
    "2028-29": 0.10,
    "2029-30": 0.15,
    "2030-31+": 0.20,
}

EU_DPP_MANDATORY_DATE = "2027-02-18"
