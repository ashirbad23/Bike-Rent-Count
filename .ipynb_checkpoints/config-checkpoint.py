from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent

# =====================
# Data & preprocessing
# =====================
DATA_DIR = ROOT_DIR / "data"
TOOLS_DIR = ROOT_DIR / "tools"
WEIGHTS_DIR = ROOT_DIR / "weights"
LOGS_DIR = ROOT_DIR / "logs"

CSV_PATH = DATA_DIR / "hour.csv"
PREPROCESSOR_PATH = TOOLS_DIR / "preprocessor.pkl"
y_SCALER_PATH = TOOLS_DIR / "y_scaler.pkl"

# =====================
# Training hyperparams
# =====================
BATCH_SIZE = 64
EPOCHS = 50
WINDOW = 24
TEST_SIZE = 0.2
VAL_SIZE = 0.1
LR = 1e-3

# =====================
# Model saving
# =====================
WEIGHTS_PATH_MLP = WEIGHTS_DIR / "weights_mlp.pt"
WEIGHTS_PATH_LSTM = WEIGHTS_DIR / "weights_lstm.pt"
