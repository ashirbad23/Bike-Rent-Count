import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from tqdm import tqdm

from utils.dataset import BikeDataModule
from models.mlp import MLP
from models.lstm import LSTMModel
from config import (
    CSV_PATH,
    PREPROCESSOR_PATH,
    y_SCALER_PATH,
    LOGS_DIR,
    BATCH_SIZE,
    EPOCHS,
    WINDOW,
    LR,
    TEST_SIZE,
    VAL_SIZE,
    WEIGHTS_PATH_MLP,
    WEIGHTS_PATH_LSTM
)

# =====================
# Device
# =====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# Training / Eval loops
# =====================

def train_one_epoch(model, loader, optimizer, criterion, epoch, tag):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc=f"{tag} | Epoch {epoch}", leave=False)

    for X, y in pbar:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


# =====================
# MLP Training
# =====================

def train_mlp():
    print("\n===== Training MLP =====")

    dm = BikeDataModule(
        csv_path=CSV_PATH,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        batch_size=BATCH_SIZE,
        window=WINDOW,
        x_preprocessor_path=PREPROCESSOR_PATH,
        y_scaler_path=y_SCALER_PATH
    )

    train_loader = dm.mlp_train_loader()
    val_loader = dm.mlp_val_loader()

    input_dim = next(iter(train_loader))[0].shape[1]

    model = MLP(input_dim=input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, tag="MLP"
        )
        val_loss = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        with open(LOGS_DIR / "mlp_hist.pkl", "wb") as f:
            pickle.dump(history, f)

        print(
            f"[MLP] Epoch {epoch:03d} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    torch.save(model.state_dict(), WEIGHTS_PATH_MLP)
    print(f"MLP weights saved to {WEIGHTS_PATH_MLP}")


# =====================
# LSTM Training
# =====================

def train_lstm():
    print("\n===== Training LSTM =====")

    dm = BikeDataModule(
        csv_path=CSV_PATH,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        batch_size=BATCH_SIZE,
        window=WINDOW,
        x_preprocessor_path=PREPROCESSOR_PATH,
        y_scaler_path=y_SCALER_PATH
    )

    train_loader = dm.lstm_train_loader()
    val_loader = dm.lstm_val_loader()

    input_dim = next(iter(train_loader))[0].shape[2]

    model = LSTMModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=1
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, tag="LSTM"
        )
        val_loss = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        with open(LOGS_DIR / "lstm_hist.pkl", "wb") as f:
            pickle.dump(history, f)

        print(
            f"[LSTM] Epoch {epoch:03d} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    torch.save(model.state_dict(), WEIGHTS_PATH_LSTM)
    print(f"LSTM weights saved to {WEIGHTS_PATH_LSTM}")


# =====================
# Entry point
# =====================

if __name__ == "__main__":
    train_mlp()
    train_lstm()
