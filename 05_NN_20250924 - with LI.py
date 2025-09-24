import pandas as pd
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ======== 裝置選擇開關 ========
def get_device(use_cpu_only: bool) -> torch.device:
    """
    開關邏輯：
    - use_cpu_only = True  -> 強制 CPU
    - use_cpu_only = False -> 有 CUDA 用 CUDA，否則 CPU
    """
    if use_cpu_only:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== 工具函式 ========
def ensure_datetime(s: pd.Series) -> pd.Series:                                                             #把Series轉datetime
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().all():
        raise ValueError("TIMETAG 不能轉成 datetime，請檢查格式。")
    return out

def temporal_split_idx_by_ratio_by_order(n: int, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:      #按時間切分
    if n <= 4:
        tr = np.arange(max(0, n-1))
        va = np.arange(max(0, n-1), n)
        return tr, va
    k = int(np.floor(n * ratio))
    return np.arange(k), np.arange(k, n)

def temporal_split_idx_by_ratio(n: int, ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:               #依比例切分
    """
    隨機切分索引：
    - ratio 是 train 的比例（0~1）
    - seed 可重現
    - 至少保留 1 筆做 valid（若 n>=2）
    """
    if n <= 1:
        # 0或1筆：全部當訓練，valid 空
        return np.arange(n), np.array([], dtype=int)

    rng = np.random
    idx = np.arange(n)
    rng.shuffle(idx)

    k = int(np.floor(n * ratio))
    k = min(max(k, 1), n - 1)  # 確保 train >=1 且 valid >=1
    tr = idx[:k]
    va = idx[k:]
    return np.sort(tr), np.sort(va)  # 回傳可不排序；若你想保持穩定性可排序

def zscore_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=0)
    sd = np.where((~np.isfinite(sd)) | (sd == 0), 1.0, sd)
    mu = np.where(~np.isfinite(mu), 0.0, mu)
    return mu, sd

def zscore_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd

def zscore_inv(xn: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return xn * sd + mu

def build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, pin_memory: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=pin_memory
    )

#time_stamp
def _now_ts() -> str:
    """回傳 'YYYYMMDD_HHMM' 的時間戳（本地時間）。"""
    return datetime.now().strftime("%Y%m%d_%H%M")
def _save_with_ts_csv(df: pd.DataFrame, out_dir: str, filename: str, ts: Optional[str] = None) -> str:
    """
    以時間戳前綴 '(YYYYMMDD_HHMM)' 存成 CSV，並回傳完整路徑。
    例如： (20250923_0110)metrics.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    if ts is None:
        ts = _now_ts()
    fname = f"({ts}){filename}"
    path = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)
    return path

# === 新增：Lautum Information === 
def _cov_centered(mat: torch.Tensor, eps: float) -> torch.Tensor:
    """
    mat: [B, D] -> 回傳 [D, D] 共變陣（/B，有偏估計），附對角 jitter。
    批次太小時回傳單位矩陣（避免數值炸掉）。
    """
    if mat.ndim != 2:
        raise ValueError("expect 2D [B, D]")
    B, D = mat.shape
    if B < 2:
        return torch.eye(D, device=mat.device, dtype=mat.dtype)
    x = mat - mat.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / float(B)
    return cov + eps * torch.eye(D, device=mat.device, dtype=mat.dtype)

def _safe_cholesky(A: torch.Tensor):
    try:
        return torch.linalg.cholesky(A)
    except RuntimeError:
        return None


def lautum_zx(z: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    """
    L(z; x) 的高斯閉式估計（batch 版）：
      P = Σ_x^{-1} Σ_xz Σ_z^{-1} Σ_zx
      L = log det(I - P) + 2 * tr( (I - P)^{-1} - I )
    回傳「越大越好」的純量 Tensor；不穩定時回 0。
    """
    if z.ndim != 2 or x.ndim != 2 or z.size(0) != x.size(0):
        raise ValueError("lautum_zx expects z:[B,Dz], x:[B,Dx] with same B.")
    B = z.size(0)
    if B < 2:
        return z.new_tensor(0.0)

    Sz = _cov_centered(z, eps)                 # [Dz, Dz]
    Sx = _cov_centered(x, eps)                 # [Dx, Dx]
    zc = z - z.mean(dim=0, keepdim=True)
    xc = x - x.mean(dim=0, keepdim=True)
    Szx = (zc.T @ xc) / float(B)               # [Dz, Dx]
    Sxz = Szx.T                                # [Dx, Dz]

    Lz = _safe_cholesky(Sz); Lx = _safe_cholesky(Sx)
    if (Lz is None) or (Lx is None):
        return z.new_tensor(0.0)

    invSz_Szx = torch.cholesky_solve(Szx, Lz)  # [Dz, Dx]
    invSx_Sxz = torch.cholesky_solve(Sxz, Lx)  # [Dx, Dz]
    P = invSx_Sxz @ invSz_Szx                  # [Dx, Dx]

    I = torch.eye(P.size(0), device=P.device, dtype=P.dtype)
    ImP = I - P
    sign, logdet = torch.linalg.slogdet(ImP)
    if sign <= 0:
        return z.new_tensor(0.0)
    try:
        Inv = torch.linalg.inv(ImP)
    except RuntimeError:
        return z.new_tensor(0.0)

    Lval = logdet + 2.0 * torch.trace(Inv - I)
    return Lval if torch.isfinite(Lval) else z.new_tensor(0.0)


def lautum_zy(z: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    """
    L(z; y) 的高斯 KL 閉式：
      設 Σ = Cov([z,y]), B = blockdiag(Cov(z), Cov(y)), k = Dz + Dy
      L = 0.5 * ( tr( Σ^{-1} B ) - k + logdet(Σ) - logdet(B) )
    """
    if y.ndim == 1:
        y = y[:, None]
    if z.size(0) != y.size(0):
        raise ValueError("Batch mismatch for z and y.")
    Bsz = z.size(0)
    if Bsz < 2:
        return z.new_tensor(0.0)

    cov_z = _cov_centered(z, eps)              # [Dz, Dz]
    cov_y = _cov_centered(y, eps)              # [Dy, Dy]
    Dz = cov_z.size(0); Dy = cov_y.size(0)

    B = torch.zeros((Dz+Dy, Dz+Dy), device=z.device, dtype=z.dtype)
    B[:Dz, :Dz] = cov_z
    B[Dz:, Dz:] = cov_y

    zy = torch.cat([z, y], dim=1)              # [B, Dz+Dy]
    Sigma = _cov_centered(zy, eps)             # [Dz+Dy, Dz+Dy]

    Lchol = _safe_cholesky(Sigma)
    if Lchol is None:
        return z.new_tensor(0.0)
    X = torch.cholesky_solve(B, Lchol)         # Σ^{-1} B
    trace_term = torch.trace(X)

    sign_S, logdet_S = torch.linalg.slogdet(Sigma)
    sign_B, logdet_B = torch.linalg.slogdet(B)
    if (sign_S <= 0) or (sign_B <= 0):
        return z.new_tensor(0.0)

    k = Dz + Dy
    Lval = 0.5 * (trace_term - k + (logdet_S - logdet_B))
    return Lval if torch.isfinite(Lval) else z.new_tensor(0.0)

 
# === 支援 sample weight 的 DataLoader ===
def build_weighted_loader(X: np.ndarray, y: np.ndarray, w: np.ndarray, batch_size: int, shuffle: bool, pin_memory: bool) -> DataLoader:
    if w.ndim == 1:
        w = w[:, None]
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
        torch.from_numpy(w).float()
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=pin_memory
    )

class ResBPNN_block(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        h = max(1, input_dim // 10)  # 防止 0 寬度
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, h),
            nn.BatchNorm1d(h),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(h, input_dim),
        )
        # 同維殘差
        self.outlayer = nn.Identity()

    def forward(self, x):
        res = x
        x = self.net(x)
        x = x + res
        x = self.outlayer(x)  # Identity
        return x

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden:int, output_dim: int = 1, num_blocks: int = 3, dropout: float = 0.0):
        super().__init__()
        blocks = [ResBPNN_block(hidden, dropout=dropout) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
        self.inputlayer = nn.Linear(input_dim, hidden)

    def forward(self, x, return_rep: bool = False):
        x = self.inputlayer(x)
        x = self.body(x)
        z = x                         # head 前的隱表示
        out = self.head(z)
        return (out, z) if return_rep else out


# === 支援 sample weight 的訓練迴圈 ===
def train_once(X_tr, y_tr, X_va, y_va, input_dim: int, cfg, w_tr=None, w_va=None):
    device = get_device(cfg.use_cpu_only)
    pin_mem = (device.type == "cuda")

    model = MLP(input_dim=input_dim, output_dim=1, dropout=cfg.dropout, num_blocks=cfg.num_blocks, hidden=cfg.hidden).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit = nn.L1Loss(reduction="none")  # 逐樣本 L1

    if w_tr is None: w_tr = np.ones((len(X_tr), 1), dtype=np.float32)
    if w_va is None: w_va = np.ones((len(X_va), 1), dtype=np.float32)

    train_loader = build_weighted_loader(X_tr, y_tr, w_tr, cfg.batch_size, shuffle=True,  pin_memory=pin_mem)
    valid_loader = build_weighted_loader(X_va, y_va, w_va, cfg.batch_size, shuffle=False, pin_memory=pin_mem)

    # EMA 緩衝（選用）
    ema_z = ema_x = ema_y = None

    best = np.inf
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss = 0.0; n_tr = 0
        mean_L = 0.0; L_count = 0

        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), wb.to(device, non_blocking=True)

            opt.zero_grad()
            # 前向：要 z
            pred, z = model(xb, return_rep=True)

            # 主要加權 L1
            per = crit(pred, yb)                  # [B,1]
            base_loss = (per * wb).mean()

            # ---- Lautum（可關）----
            L_val = pred.new_tensor(0.0)
            if cfg.lautum_strength > 0.0:
                # 可選 EMA：平滑 batch 噪音
                if cfg.lautum_use_ema:
                    a = cfg.lautum_ema_alpha
                    ema_z = (a * ema_z + (1-a) * z) if ema_z is not None else z.detach()
                    ema_x = (a * ema_x + (1-a) * xb) if ema_x is not None else xb.detach()
                    ema_y = (a * ema_y + (1-a) * yb) if ema_y is not None else yb.detach()
                    z_use, x_use, y_use = ema_z, ema_x, ema_y
                else:
                    z_use, x_use, y_use = z, xb, yb

                try:
                    if cfg.lautum_mode.lower() == "zx":
                        L_val = lautum_zx(z_use, x_use, cfg.lautum_eps)
                    elif cfg.lautum_mode.lower() == "zy":
                        L_val = lautum_zy(z_use, y_use, cfg.lautum_eps)
                    else:
                        raise ValueError("lautum_mode must be 'zx' or 'zy'")
                except RuntimeError:
                    L_val = pred.new_tensor(0.0)

            # 最小化 base - λ·L（= 最大化 L）
            total_loss = base_loss - cfg.lautum_strength * L_val
            total_loss.backward()
            opt.step()

            bs = xb.size(0)
            run_loss += base_loss.item() * bs   # 用 base_loss 報表，易比較
            n_tr += bs
            if torch.isfinite(L_val):
                mean_L += float(L_val.item()); L_count += 1

        tr_mae = run_loss / max(n_tr, 1)
        mean_L = (mean_L / max(L_count, 1)) if L_count else 0.0

        # ===== 驗證：只看 MAE，不加 Lautum =====
        model.eval(); va_loss = 0.0; n_va = 0
        with torch.no_grad():
            for xb, yb, wb in valid_loader:
                xb, yb, wb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True), wb.to(device, non_blocking=True)
                pred = model(xb)
                per = crit(pred, yb)
                loss = (per * wb).mean()
                bs = xb.size(0)
                va_loss += loss.item() * bs
                n_va += bs
        va_mae = va_loss / max(n_va, 1)

        history.append({"epoch": epoch, "train_mae": tr_mae, "valid_mae": va_mae, "lautum": mean_L})

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}] Train MAE={tr_mae:.6f}, Valid MAE={va_mae:.6f}, L≈{mean_L:.5f}")

        if va_mae + 1e-12 < best:
            best = va_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if cfg.patience > 0 and wait >= cfg.patience:
                print(f"Early stopping triggered at epoch {epoch}, best val MAE={best:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
   


def predict_numpy(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """
    依據模型當前所在的裝置推論，避免 CPU/GPU 混用錯誤。
    """
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).float().to(device, non_blocking=True)
            pr = model(xb).detach().cpu().numpy()
            preds.append(pr)
    return np.vstack(preds)




# ======== 參數 & 主流程 ========

class TrainConfig:
    def __init__(
        self,
        test_start="2024-04-01 00:00:00",
        separate_models=True,
        per_combo_normalizer=True,
        use_ohe_columns=True,
        target_col="AVERAGE",
        time_col="TIMETAG",
        id_cols=None,
        # impute
        impute_X_strategy="mean",   # "zero" 或 "mean"
        impute_fit_scope="pre_only",# "pre_only" 或 "all"
        # NN
        dropout=0.0,
        epochs=1000,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        patience=100,
        # 裝置
        use_cpu_only=False,
        # split
        pre_ratio=0.8,
        # filters
        min_pre=10,
        min_test=1,
        num_blocks=3,
        hidden=16,
        # ===== 新增：極端值處理 =====
        outlier_n: float = 6.0,      # UL/LL = mean ± n*std（std 以 IQR 濾後資料計算）
        
        # ==== Lautum 設定 ====
        lautum_strength: float = 0.01,   # λ：0 = 關閉
        lautum_mode: str = "zx",         # "zx" = L(z; x), "zy" = L(z; y)
        lautum_eps: float = 1e-6,        # 抖動（jitter），數值穩定用
        lautum_use_ema: bool = False,    # 是否用 EMA 平滑共變
        lautum_ema_alpha: float = 0.99,  # EMA 衰減（0.9/0.99/0.999）
    ):
        self.test_start = test_start
        self.separate_models = separate_models
        self.per_combo_normalizer = per_combo_normalizer
        self.use_ohe_columns = use_ohe_columns
        self.target_col = target_col
        self.time_col = time_col

        if id_cols is None:
            id_cols = ["CONTEXTID", "EQID", "CHAMBERID", "RECIPEID"]
        self.id_cols = id_cols

        self.impute_X_strategy = impute_X_strategy
        self.impute_fit_scope = impute_fit_scope

        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.use_cpu_only = use_cpu_only

        self.pre_ratio = pre_ratio
        self.min_pre = min_pre
        self.min_test = min_test
        self.num_blocks = num_blocks
        self.hidden = hidden
        self.outlier_n = float(outlier_n)

        #Lautum Information
        self.lautum_strength = float(lautum_strength)
        self.lautum_mode = str(lautum_mode)
        self.lautum_eps = float(lautum_eps)
        self.lautum_use_ema = bool(lautum_use_ema)
        self.lautum_ema_alpha = float(lautum_ema_alpha)

def train_on_data(data: pd.DataFrame, cfg: TrainConfig):
    ts = _now_ts()
    def _safe_ul_ll(s: pd.Series) -> Tuple[float, float]:
        # 只用來「計算統計量」的 IQR 篩內部點，不直接刪資料
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) == 0:
            return np.nan, np.nan
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        # 若 iqr 為 0，退而直接用全樣本
        if not np.isfinite(iqr) or iqr == 0:
            mu = s.mean()
            sd = s.std(ddof=0)
        else:
            mask = (s >= (q1 - 2.5 * iqr)) & (s <= (q3 + 2.5 * iqr))
            inliers = s[mask]
            if len(inliers) == 0:
                mu = s.mean()
                sd = s.std(ddof=0)
            else:
                mu = inliers.mean()
                sd = inliers.std(ddof=0)
        # 防止 sd=0 或 NaN
        if (not np.isfinite(sd)) or sd == 0:
            sd = 1e-12
        if not np.isfinite(mu):
            mu = 0.0
        UL = mu + n * (sd+1e-4)
        LL = mu - n * (sd+1e-4)
        return LL, UL

    df = data.copy()

    # 檢查欄位
    need = set(cfg.id_cols + [cfg.target_col, cfg.time_col])
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要欄位：{miss}")

    # 時間
    df[cfg.time_col] = ensure_datetime(df[cfg.time_col])
    df = df.sort_values(cfg.time_col).reset_index(drop=True)

    # 建立 Combination
    df["Combination"] = (
        df["EQID"].astype(str) + "_" +
        df["CHAMBERID"].astype(str) + "_" +
        df["RECIPEID"].astype(str)
    )

    # 特徵欄位
    exclude = set([cfg.target_col, cfg.time_col, "Combination"] + cfg.id_cols)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if c not in exclude]
    if not cfg.use_ohe_columns:
        feature_cols = [c for c in feature_cols if "ohe" not in c.lower()]
    if len(feature_cols) == 0:
        raise ValueError("沒有可用的數值特徵（可能全被排除了）。")

    # ===== X 的缺失補值 =====
    test_ts = pd.to_datetime(cfg.test_start)
    if cfg.impute_X_strategy not in ("zero", "mean"):
        raise ValueError("impute_X_strategy 只能為 'zero' 或 'mean'。")

    if cfg.impute_X_strategy == "zero":
        df[feature_cols] = df[feature_cols].fillna(0.0)
    else:
        if cfg.impute_fit_scope == "pre_only":
            fit_mask = df[cfg.time_col] < test_ts
        elif cfg.impute_fit_scope == "all":
            fit_mask = np.ones(len(df), dtype=bool)
        else:
            raise ValueError("impute_fit_scope 只能為 'pre_only' 或 'all'。")
        fit_means = df.loc[fit_mask, feature_cols].mean(axis=0, skipna=True).fillna(0.0)
        df[feature_cols] = df[feature_cols].fillna(fit_means)

    # ===== 極端值處理 =====
    n = cfg.outlier_n
    y_col = cfg.target_col
    drop_idx = np.zeros(len(df), dtype=bool)  # 要刪的列（僅 Y 超界）

    # y 缺失必刪
    df = df[~df[y_col].isna()].reset_index(drop=True)

    to_float = list(dict.fromkeys(feature_cols + [cfg.target_col]))
    df[to_float] = (
        df[to_float]
        .apply(lambda s: pd.to_numeric(s, errors="coerce"))
        .astype("float64", copy=False)
    )

    # 逐 RECIPEID 做 clip / 刪除
    for rid, gidx in df.groupby("RECIPEID").groups.items():
        idx = np.array(list(gidx))
        # 對 X 每一欄做 clip
        for c in feature_cols:
            LL, UL = _safe_ul_ll(df.loc[idx, c])
            if np.isfinite(LL) and np.isfinite(UL):
                df.loc[idx, c] = df.loc[idx, c].clip(lower=LL, upper=UL)

        # 對 y 做「超界刪除」
        LL_y, UL_y = _safe_ul_ll(df.loc[idx, y_col])
        if np.isfinite(LL_y) and np.isfinite(UL_y):
            y_vals = pd.to_numeric(df.loc[idx, y_col], errors="coerce")
            drop_idx[idx] |= (y_vals < LL_y) | (y_vals > UL_y)

    # 先把 Y 超界的列刪除
    if drop_idx.any():
        df = df.loc[~drop_idx].reset_index(drop=True)

    # （保持你原本的輸出存檔，若不需要可移除）
    data_clean_path = _save_with_ts_csv(df, "./Data_clean", "Data_for_BPNN.csv", ts)

    # ===== pre/test 切、建立 by_combo =====
    test_ts = pd.to_datetime(cfg.test_start)

    by_combo: Dict[str, dict] = {}
    for combo, g in df.groupby("Combination"):
        g_pre  = g[g[cfg.time_col] < test_ts]
        g_test = g[g[cfg.time_col] >= test_ts]

        if len(g_pre) < cfg.min_pre or len(g_test) < cfg.min_test:
            continue

        X_pre = g_pre[feature_cols].to_numpy()
        y_pre = g_pre[[y_col]].to_numpy()
        t_pre = g_pre[cfg.time_col].to_numpy()
        id_pre = g_pre["CONTEXTID"].to_numpy()

        X_te  = g_test[feature_cols].to_numpy()
        y_te  = g_test[[y_col]].to_numpy()
        t_te  = g_test[cfg.time_col].to_numpy()
        id_te = g_test["CONTEXTID"].to_numpy()

        tr_idx, va_idx = temporal_split_idx_by_ratio(len(X_pre), cfg.pre_ratio)
        X_tr, y_tr, t_tr, id_tr = X_pre[tr_idx], y_pre[tr_idx], t_pre[tr_idx], id_pre[tr_idx]
        X_va, y_va, t_va, id_va = X_pre[va_idx], y_pre[va_idx], t_pre[va_idx], id_pre[va_idx]

        by_combo[combo] = {
            "X_tr": X_tr, "y_tr": y_tr, "t_tr": t_tr, "id_tr": id_tr,
            "X_va": X_va, "y_va": y_va, "t_va": t_va, "id_va": id_va,
            "X_te": X_te, "y_te": y_te, "t_te": t_te, "id_te": id_te,
        }

    if not by_combo:
        raise RuntimeError("沒有任何 combination 同時滿足 pre/test 數量門檻，可調整 test_start 或門檻。")

    # === (重點) 依 combination 樣本量建權重：小樣本較大，且最大/最小 ≤ 10× ===
    # 基於 pre (train+valid) 樣本數，避免用到未來資訊
    combo_sizes = {c: (len(d["X_tr"]) + len(d["X_va"])) for c, d in by_combo.items()}
    raw_w = {c: 1.0 / max(1, n_) for c, n_ in combo_sizes.items()}  # 也可改成 1/sqrt(n)
    w_values = np.array(list(raw_w.values()), dtype=np.float64)
    w_max = float(np.max(w_values))
    # 夾到 10× 範圍內：最小不得小於 w_max/10
    w_clipped = {c: max(w_max / 5.0, w) for c, w in raw_w.items()}
    # 校正到「以樣本數加權的平均」= 1，避免改變整體 loss 尺度
    total_n = sum(combo_sizes.values())
    weighted_sum = sum(w_clipped[c] * combo_sizes[c] for c in combo_sizes)
    scale = (total_n / max(weighted_sum, 1e-12))
    combo_weight = {c: float(w_clipped[c] * scale) for c in combo_sizes}

    # 把權重展開到每一筆樣本（train/valid/test 都存；test 之後可用於加權評估）
    for combo, d in by_combo.items():
        w_tr = np.full((len(d["X_tr"]), 1), combo_weight[combo], dtype=np.float32)
        w_va = np.full((len(d["X_va"]), 1), combo_weight[combo], dtype=np.float32)
        w_te = np.full((len(d["X_te"]), 1), combo_weight[combo], dtype=np.float32)
        d["w_tr"] = w_tr
        d["w_va"] = w_va
        d["w_te"] = w_te

    # ---- 一律用「全球」X normalizer（用所有 pre：train+valid）----
    X_all_fit = np.vstack([np.vstack([d["X_tr"], d["X_va"]]) for d in by_combo.values()])
    x_mu_g, x_sd_g = zscore_fit(X_all_fit)

    # ---- y normalizer：沿用原本邏輯（可 per-combo 或 shared）----
    scalers: Dict[str, dict] = {"__x_global__": {"x_mu": x_mu_g, "x_sd": x_sd_g}}
    if cfg.per_combo_normalizer:
        for combo, d in by_combo.items():
            y_fit = np.vstack([d["y_tr"], d["y_va"]])
            y_mu, y_sd = zscore_fit(y_fit)
            scalers[combo] = {"y_mu": y_mu, "y_sd": y_sd}
    else:
        y_all_fit = np.vstack([np.vstack([d["y_tr"], d["y_va"]]) for d in by_combo.values()])
        y_mu_g, y_sd_g = zscore_fit(y_all_fit)
        scalers["__shared__"] = {"y_mu": y_mu_g, "y_sd": y_sd_g}

    # ---- 套用 normalizer：X 用全域、y 用 per-combo/shared ----
    for combo, d in by_combo.items():
        y_key = combo if cfg.per_combo_normalizer else "__shared__"
        y_mu, y_sd = scalers[y_key]["y_mu"], scalers[y_key]["y_sd"]

        # X 一律使用全域的 x_mu_g, x_sd_g
        d["X_tr_n"] = zscore_apply(d["X_tr"], x_mu_g, x_sd_g)
        d["X_va_n"] = zscore_apply(d["X_va"], x_mu_g, x_sd_g)
        d["X_te_n"] = zscore_apply(d["X_te"], x_mu_g, x_sd_g)

        # y 依設定
        d["y_tr_n"] = zscore_apply(d["y_tr"], y_mu, y_sd)
        d["y_va_n"] = zscore_apply(d["y_va"], y_mu, y_sd)

    input_dim = by_combo[next(iter(by_combo))]["X_tr_n"].shape[1]

    models: Dict[str, nn.Module] = {}
    historys = {}
    if cfg.separate_models:
        for combo, d in by_combo.items():
            # separate_models 下跨組不平衡較少影響；仍保留權重以維持一致接口
            model, history = train_once(
                d["X_tr_n"], d["y_tr_n"], d["X_va_n"], d["y_va_n"], input_dim, cfg
            )
            models[combo] = model
            historys[combo] = history
    else:
        X_tr_all = np.vstack([d["X_tr_n"] for d in by_combo.values()])
        y_tr_all = np.vstack([d["y_tr_n"] for d in by_combo.values()])
        w_tr_all = np.vstack([d["w_tr"]   for d in by_combo.values()])

        X_va_all = np.vstack([d["X_va_n"] for d in by_combo.values()])
        y_va_all = np.vstack([d["y_va_n"] for d in by_combo.values()])
        w_va_all = np.vstack([d["w_va"]   for d in by_combo.values()])

        model, history = train_once(
            X_tr_all, y_tr_all, X_va_all, y_va_all, input_dim, cfg,
            w_tr=w_tr_all, w_va=w_va_all
        )
        models["__shared__"] = model
        historys["__shared__"] = history

    rows, pred_rows = [], []
    for combo, d in by_combo.items():
        key_model  = combo if cfg.separate_models else "__shared__"
        key_scaler = combo if cfg.per_combo_normalizer else "__shared__"
        model = models[key_model]
        y_mu, y_sd = scalers[key_scaler]["y_mu"], scalers[key_scaler]["y_sd"]

        # 推論
        y_pred_n = predict_numpy(model, d["X_te_n"], batch_size=max(256, cfg.batch_size))
        y_pred   = zscore_inv(y_pred_n, y_mu, y_sd).ravel()
        y_true   = d["y_te"].ravel()

        # 各點誤差
        abs_errs = np.abs(y_pred - y_true)

        # === 新增的指標 ===
        test_mae   = float(abs_errs.mean())
        test_p95ae = float(np.percentile(abs_errs, 95))   # 絕對誤差 95 百分位
        test_y_std = float(np.std(y_true, ddof=0))        # test y 的 std（母體）

        rows.append({
            "Combination": combo,
            "train_n":    len(d["X_tr"]),
            "valid_n":    len(d["X_va"]),
            "test_n":     len(d["X_te"]),
            "test_mae":   test_mae,
            "test_p95ae": test_p95ae,
            "test_y_std": test_y_std,
        })

        # 保留逐點輸出（含 abs_err）
        for cid, tt, yt, yp, ae in zip(d["id_te"], d["t_te"], y_true, y_pred, abs_errs):
            pred_rows.append({
                "Combination": combo,
                "CONTEXTID": cid,
                "TIMETAG": pd.Timestamp(tt),
                "y_true": float(yt),
                "y_pred": float(yp),
                "abs_err": float(ae),
            })

    metrics = pd.DataFrame(rows).sort_values("Combination").reset_index(drop=True)
    pred_points = pd.DataFrame(pred_rows).sort_values(
        ["Combination","TIMETAG","CONTEXTID"]
    ).reset_index(drop=True)

    # === 把評估輸出加時間戳存檔（原始行為） ===
    metrics_path = _save_with_ts_csv(metrics, "./Outputs", "metrics.csv", ts)
    pred_points_path = _save_with_ts_csv(pred_points, "./Outputs", "pred_points.csv", ts)

    return {
        "feature_cols": feature_cols,
        "models": models,
        "historys": historys,
        "scalers": scalers,
        "metrics": metrics,
        "pred_points": pred_points,
        "used_config": cfg,
    }

#%%
if __name__ == "__main__":
    # 每次手動調整的 Lautum 係數
    LAUTUM_STRENGTH = 0.01

    cfg = TrainConfig(
        test_start="2024-04-01 00:00:00",
        separate_models=False,
        per_combo_normalizer=True,
        use_ohe_columns=False,
        target_col="AVERAGE",
        time_col="TIMETAG",
        id_cols=["CONTEXTID","EQID","CHAMBERID","RECIPEID"],
        # NN 超參數
        epochs=500, patience=20, batch_size=2048, lr=1e-4, weight_decay=1e-4,
        num_blocks=20, dropout=0.2, hidden=250,
        # 裝置 & 其他
        use_cpu_only=False, outlier_n=6,
        # Lautum（手動設定）
        lautum_strength=LAUTUM_STRENGTH,
        lautum_mode="zy",
        lautum_eps=1e-6,
        lautum_use_ema=False,      # 需要的話改 True
        lautum_ema_alpha=0.99,
    )

    data = pd.read_csv("./Data_clean/Indicator_Table_selected_encoded.csv")
    result = train_on_data(data, cfg)

    # 訓練完成後輸出每組 metrics（同時檔案已寫到 ./Outputs/）
    print(result["metrics"])

