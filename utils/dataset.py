import numpy as np
import pandas as pd
import torch
import joblib

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# =========================
# Internal Torch Datasets
# =========================

class _BikeMLPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class _BikeLSTMDataset(Dataset):
    def __init__(self, X, y, window):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return len(self.X) - self.window

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.window]
        y_target = self.y[idx + self.window]
        return X_seq, y_target.view(1)


# =========================
# Public Data Module
# =========================

class BikeDataModule:
    def __init__(
        self,
        csv_path,
        test_size=0.15,
        val_size=0.15,
        batch_size=64,
        window=24,
        x_preprocessor_path="../tools/x_preprocessor.pkl",
        y_scaler_path="../tools/y_scaler.pkl"
    ):
        assert test_size + val_size < 1.0, "Train size must be > 0"

        self.csv_path = csv_path
        self.test_size = test_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.window = window
        self.x_preprocessor_path = x_preprocessor_path
        self.y_scaler_path = y_scaler_path

        self._load_data()
        self._add_cyclic_features()
        self._define_columns()
        self._build_preprocessors()
        self._split_and_preprocess()

    # ---------- Internal steps ----------

    def _load_data(self):
        self.df = pd.read_csv(self.csv_path)

    def _add_cyclic_features(self):
        df = self.df

        df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
        df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)

        df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

        df["mnth_sin"] = np.sin(2 * np.pi * (df["mnth"] - 1) / 12)
        df["mnth_cos"] = np.cos(2 * np.pi * (df["mnth"] - 1) / 12)

    def _define_columns(self):
        self.num_cols = [
            "temp", "atemp", "hum", "windspeed",
            "hr_sin", "hr_cos",
            "weekday_sin", "weekday_cos",
            "mnth_sin", "mnth_cos"
        ]
        self.cat_cols = ["season", "weathersit"]
        self.bin_cols = ["holiday", "workingday", "yr"]

    def _build_preprocessors(self):
        self.x_preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_cols),
                ("cat", OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore"
                ), self.cat_cols),
                ("bin", "passthrough", self.bin_cols)
            ]
        )

        self.y_scaler = StandardScaler()

    def _split_and_preprocess(self):
        X = self.df[self.num_cols + self.cat_cols + self.bin_cols]
        y = self.df["cnt"].values.reshape(-1, 1)

        n = len(X)
        train_end = int(n * (1 - self.val_size - self.test_size))
        val_end = int(n * (1 - self.test_size))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        # ----- X -----
        self.X_train = self.x_preprocessor.fit_transform(X_train)
        self.X_val = self.x_preprocessor.transform(X_val)
        self.X_test = self.x_preprocessor.transform(X_test)

        # ----- y -----
        self.y_train = self.y_scaler.fit_transform(y_train).ravel()
        self.y_val = self.y_scaler.transform(y_val).ravel()
        self.y_test = self.y_scaler.transform(y_test).ravel()

        joblib.dump(self.x_preprocessor, self.x_preprocessor_path)
        joblib.dump(self.y_scaler, self.y_scaler_path)

    # =====================
    # MLP loaders
    # =====================

    def mlp_train_loader(self, shuffle=True):
        return DataLoader(
            _BikeMLPDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=shuffle
        )

    def mlp_val_loader(self):
        return DataLoader(
            _BikeMLPDataset(self.X_val, self.y_val),
            batch_size=self.batch_size,
            shuffle=False
        )

    def mlp_test_loader(self):
        return DataLoader(
            _BikeMLPDataset(self.X_test, self.y_test),
            batch_size=self.batch_size,
            shuffle=False
        )

    # =====================
    # LSTM loaders
    # =====================

    def lstm_train_loader(self, shuffle=False):
        return DataLoader(
            _BikeLSTMDataset(self.X_train, self.y_train, self.window),
            batch_size=self.batch_size,
            shuffle=shuffle
        )

    def lstm_val_loader(self):
        X = np.vstack([self.X_train[-self.window:], self.X_val])
        y = np.concatenate([self.y_train[-self.window:], self.y_val])
        return DataLoader(
            _BikeLSTMDataset(X, y, self.window),
            batch_size=self.batch_size,
            shuffle=False
        )

    def lstm_test_loader(self):
        X = np.vstack([self.X_val[-self.window:], self.X_test])
        y = np.concatenate([self.y_val[-self.window:], self.y_test])
        return DataLoader(
            _BikeLSTMDataset(X, y, self.window),
            batch_size=self.batch_size,
            shuffle=False
        )
