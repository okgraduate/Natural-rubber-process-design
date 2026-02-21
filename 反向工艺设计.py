# -*- coding: utf-8 -*-
"""
天然橡胶初加工工艺设计系统

功能概述：
1. 解析 acid / enzyme / natural 三个 indicator 表，构建 base_df
2. 计算塑性指标 plasticity（作为 6 指标之一，和其他指标完全对等）
3. 标记 dry_type：
   - acid/natural 的 dry -> pellet_drying （Pellet drying，造粒烘干）
   - enzyme 的 dry        -> sheet_drying  （Sheet drying，挂片烘干）
4. 使用 NNI 对每个凝固方式做数据增强，得到 aug_df（约 3000 条）
5. 训练 BP 神经网络，完成 6 指标 -> 凝固方式 (acid/enzyme/natural) 三分类
6. 保存：
   - bp_model.pt
   - scaler.npz
   - label_map.npz
   - aug_data_with_drytype.pkl  （完整增强 DataFrame，用于反向推荐）
7. 交互模式（工艺阶段性预测 + 反向工艺推荐）：
   - 输入 6 项指标（氮含量 灰分含量 挥发分含量 杂质含量 门尼粘度 塑性初值）
   - 输出顺序：
       (1) 凝固方式概率
       (2) 推断原料路径：acid/enzyme -> latex；natural -> glue
       (3) 推荐凝固方式
       (4) 推荐破碎和烘干方式：Sheet drying / Pellet drying
   - 在对应凝固方式下，选“终点状态”（dry 或 package）的样本
   - 找到与目标最接近的 5 条终点样本
   - 输出推荐表：工艺阶段、烘干类型、6 指标值、RelErr（综合相对误差）
   - 给出一条示例完整工艺路径，方便论文说明
"""

import csv
from io import StringIO
import pickle
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# 1. 原始 CSV 文本
# =========================

RAW_CSV_NATURAL = r"""
process_step,Nitrigen content/%,Standard deviation,Ash content/ %,Standard deviation,volatiles content/ %,Standard deviation,impurity content/ %,Standard deviation,Monney viscosity,Standard deviation,P0,Standard deviation,PRI,Standard deviation,process_step,Nitrigen content/%,Standard deviation,Ash content/ %,Standard deviation,volatiles content/ %,Standard deviation,impurity content/ %,Standard deviation,Monney viscosity,Standard deviation,P0,Standard deviation,PRI,Standard deviation,process_step,Nitrigen content/%,Standard deviation,Ash content/ %,Standard deviation,volatiles content/ %,Standard deviation,impurity content/ %,Standard deviation,Monn ey viscosity,Standard deviation,P0,Standard deviation,PRI,Standard deviation
Glue,0.2208,0.004608,0.2,0.04,0.5,0.08,0.43,0.17,63.77,1.48492,64.5,1,82.94573643,1,Glue,0.2688,0.0384,0.41,0.01289,0.59,0.01,0.46,0.08,62.6,1.41421,66.5,1,79.69924812,1,Glue,0.3013,0.0528,0.41,0.05,0.55,0.12,1.13,0.95,65.35,1.7355,64.5,1,82.94573643,1
Broken,0.2144,0.003152,0.22,0.06,0.43,0.02,0.34,0.13,67.33,2.36548,64,1,78.125,1,Broken,0.2816,0.0016,0.34,0.05825,0.58,0.03,0.32,0.09,59.62,0.66468,63,1,83.33333333,1,Broken,0.3025,0.0316,0.34,0.07,0.53,0.06,0.38,0.07,63.54,1.8532,64,1,78.125,1
First granulation,0.2528,0.02992,0.17,0.02,0.41,0.02,0.26,0.03,52.91,2.68215,58.5,1,79.48717949,1,First granulation,0.312,0.0032,0.31,0.09001,0.64,0.11,0.31,0.09,55.78,4.5467,60,1,74.16666667,1,First granulation,0.2855,0.0254,0.31,0.09,0.42,0.09,0.15,0.05,57.35,2.5575,58.5,1,79.48717949,1
crepe,0.2416,0.009872,0.15,0.02,0.42,0.01,0.24,0.01,54.71,1.49907,60.5,1,78.51239669,1,crepe,0.312,0.0032,0.29,0.01765,0.49,0.01,0.34,0.06,65.94,2.59508,58.5,1,74.35897436,1,crepe,0.2836,0.0366,0.29,0.02,0.39,0.05,0.16,0.12,54.32,3.5342,59.5,1,79.83193277,1
Second granulation,0.2208,0.0005888,0.13,0.04,0.39,0.03,0.25,0.02,50.89,4.70933,60,1,81.66666667,1,Second granulation,0.2912,0.008,0.31,0.02809,0.51,0.03,0.29,0.02,52.82,3.09713,56,1,74.10714286,1,Second granulation,0.2652,0.0415,0.31,0.04,0.35,0.06,0.11,0.06,58.35,4.5389,55.5,1,78.37837838,1
dry,0.2064,0.02464,0.04,0,0.29,0.01,0.21,0.06,43.48,1.81019,47,1,82.9787234,1,dry,0.2608,0.0016,0.24,0.03121,0.31,0.02,0.31,0.03,44.63,2.58094,48,1,80.20833333,1,dry,0.2435,0.0152,0.24,0.03,0.23,0.01,0.12,0.01,49.32,3.5715,47,1,82.9787234,1
package,0.2032,0.005472,0.06,0.01,0.3,0.02,0.27,0.05,43.1,0.77782,48,1,80.20833333,1,package,0.2608,0.0032,0.23,0.04879,0.29,0.02,0.3,0.03,46,0.55861,47.5,1,78.94736842,1,package,0.2411,0.0037,0.23,0.02,0.23,0.02,0.12,0.03,48.35,1.2574,48,1,80.20833333,1
"""

RAW_CSV_ENZYME = r"""
process_step,Nitrigen content/%,Standard deviation,Ash content/ %,Standard deviation,volatiles content/ %,Standard deviation,impurity content/ %,Standard deviation,Monney viscosity,Standard deviation,P0,Standard deviation,PRI,Standard deviation,process_step,Nitrigen content/%,Standard deviation,Ash content/ %,Standard deviation,volatiles content/ %,Standard deviation,impurity content/ %,Standard deviation,Monney viscosity,Standard deviation,P0,Standard deviation,PRI,Standard deviation,process_step,Nitrigen content/%,Standard deviation,Ash content/ %,Standard deviation,volatiles content/ %,Standard deviation,impurity content/ %,Standard deviation,Monney viscosity,Standard deviation,P0,Standard deviation,PRI,Standard deviation
latex,0.8896,0.0248,2.27,0.15,1.42,0.21,0.19,0.04,59.7,1.8,52,1,83.65384615,1,latex,0.9552,0.0272,2.55,0.24,1.87,0.18,0.24,0.11,44.89,1.7,50.5,1,96.03960396,1,latex,0.9033,0.0305,1.97,0.2,1.33,0.21,0.1,0.05,48.5,1.6,52,1,90.38461538,1
coagulation,0.4288,0.00744,0.84,0.09,0.78,0.01,0.32,0.12,51.9,6.6,45.5,1,86.81318681,1,coagulation,0.4512,0.016,0.92,0.08,1.38,0.05,0.22,0.07,59.22,9.67,52,1,89.42307692,1,coagulation,0.425,0.0208,0.82,0.12,0.85,0.06,0.25,0.08,52.6,2.6,47.5,1,89.47368421,1
mature,0.3536,0.012656,0.72,0.08,0.78,0.08,0.35,0.03,55.6,4.2,52,1,75.96153846,1,mature,0.3744,0.0224,0.62,0.08,1.31,0.01,0.33,0.02,78.33,6.82,58.5,1,89.74358974,1,mature,0.3514,0.0102,0.73,0.09,0.82,0.04,0.13,0.03,64.7,3.2,53.5,1,88.78504673,1
thining,0.2512,0.03328,0.59,0.02,0.69,0.06,0.44,0.03,62.5,2.9,48,1,85.41666667,1,thining,0.3712,0.0016,0.56,0.06,1.16,0.01,0.37,0.09,73.65,0.08,60,1,91.66666667,1,thining,0.3425,0.0181,0.52,0.07,0.69,0.05,0.18,0.07,65.2,1.5,54,1,83.33333333,1
crepe,0.2704,0.009248,0.53,0.26,0.66,0.03,0.38,0.03,69,6.8,45.5,1,91.20879121,1,crepe,0.3328,0.0176,0.65,0.19,1.1,0.02,0.41,0.05,62.66,2.38,58.8,1,89.28571429,1,crepe,0.3208,0.0122,0.48,0.15,0.66,0.02,0.23,0.06,68.5,3.7,52,1,88.46153846,1
dry,0.2512,0.0032,0.34,0.07,0.45,0.01,0.34,0.04,42.9,1,43,1,87.20930233,1,dry,0.2976,0.0016,0.55,0.05,1.09,0.02,0.34,0.01,53.91,4.74,47.5,1,83.15789474,1,dry,0.272,0.0085,0.42,0.07,0.52,0.03,0.17,0.06,56.9,1.7,43.5,1,87.35632184,1
package,0.2544,0.005312,0.31,0.02,0.43,0.03,0.35,0.01,45.7,2.6,42.5,1,87.05882353,1,package,0.2944,0.0018,0.49,0.04,0.99,0.04,0.36,0.03,55.12,0.44,48,1,79.16666667,1,package,0.2681,0.0058,0.41,0.06,0.5,0.02,0.18,0.11,56.3,1.2,42,1,89.28571429,1
"""

RAW_CSV_ACID = r"""
process_step,Nitrigen content/%,Standard deviation,Ash content/ %,Standard deviation,volatiles content/ %,Standard deviation,impurity content/ %,Standard deviation,Monney viscosity,Standard deviation,P0,Standard deviation,PRI,Standard deviation,,process_step,Nitrigen content/%,Standard deviation,Ash content/ %,Standard deviation,volatiles content/ %,Standard deviation,impurity content/ %,Standard deviation,Monney viscosity,Standard deviation,P0,Standard deviation,PRI,Standard deviation,process_step,Nitrigen content/%,Standard deviation,Ash content/ %,Standard deviation,volatiles content/ %,Standard deviation,impurity content/ %,Standard deviation,Monney viscosity,Standard deviation,P0,Standard deviation,PRI,Standard deviation
latex,0.7856,0.03264,2.9,0.135,3.01,0.19,0.14,0.01,70.44,1.75,65,1,93.07692308,1,,latex,0.7584,0.0224,2,0,1.99,0.12,0.26,0.03,61.5,3.3,51.5,1,93.2038835,1,latex,0.7522,0.0358,1.98,0.157,1.58,0.09,0.12,0.07,55.6,1.6,50,1,93,1
coagulation,0.4528,0.0002032,0.37,0.0347,0.74,0.19,0.23,0.02,64.7,3.36,47.5,1,90.52631579,1,,coagulation,0.5504,0.0288,0.71,0.1,0.73,0.02,0.1,0.09,47.8,5.9,43.5,1,90.8045977,1,coagulation,0.5342,0.0302,0.82,0.062,0.72,0.05,0.19,0.15,52.5,3.3,42,1,91.66666667,1
mature,0.4336,0.0416,0.293,0.0177,0.76,0.07,0.24,0.02,69.42,1.16,41,1,95.12195122,1,,mature,0.5136,0.0352,0.45,0.03,0.47,0.02,0.24,0.1,53.7,2,41.5,1,96.38554217,1,mature,0.5007,0.0281,0.49,0.103,0.58,0.08,0.2,0.08,51.2,1.7,41.5,1,86.74698795,1
thining,0.3952,0.03856,0.278,0.074,0.6,0.26,0.35,0.05,59.65,2.77,41,1,91.46341463,1,,thining,0.4688,0.0016,0.44,0.06,0.56,0.05,0.33,0.03,60.5,1.4,43,1,98.8372093,1,thining,0.4717,0.0125,0.42,0.092,0.43,0.01,0.25,0.02,57.5,1.3,40,1,96.25,1
crepe,0.4096,0.0002496,0.223,0.0437,0.69,0.06,0.4,0.03,61.24,1.63,39.5,1,82.27848101,1,,crepe,0.464,0.0032,0.51,0.07,0.56,0.07,0.36,0,61.2,0.5,42.5,1,94.11764706,1,crepe,0.4628,0.025,0.402,0.038,0.4,0.05,0.19,0.06,54.2,0.8,38.5,1,90.90909091,1
granulation,0.4112,0.01984,0.171,0.0158,0.46,0.11,0.35,0.03,62.14,2.52,42.5,1,83.52941176,1,,granulation,0.4288,0.008,0.4,0.01,0.4,0.01,0.32,0.05,63,0.9,42,1,98.80952381,1,granulation,0.4239,0.0152,0.292,0.031,0.38,0.02,0.26,0.08,59.7,1.7,40,1,85,1
dry,0.4112,0.01568,0.174,0.014,0.4,0.08,0.38,0.01,55.83,3.66,38.5,1,84.41558442,1,,dry,0.4336,0.0048,0.4,0.01,0.4,0.01,0.33,0.03,47.4,5.1,40,1,93.75,1,dry,0.4105,0.0058,0.288,0.025,0.18,0.01,0.28,0.07,47.4,1.5,35.5,1,91.54929577,1
package,0.4064,0.015648,0.224,0.0488,0.26,0.01,0.33,0.05,50.52,1.82,39,1,84.61538462,1,,package,0.4352,0.0096,0.41,0.02,0.36,0,0.31,0.03,49.3,0.9,41,1,91.46341463,1,package,0.4048,0.0175,0.283,0.001,0.15,0.02,0.2,0.05,53.3,0.9,35,1,91.42857143,1
"""

# =========================
# 2. CSV 解析
# =========================

def parse_horizontal_csv(raw_text):
    records = []
    reader = csv.reader(StringIO(raw_text.strip()))
    rows = list(reader)
    if not rows:
        return pd.DataFrame()
    data_rows = rows[1:]
    group_size = 15
    for row in data_rows:
        if not row or all(str(x).strip() == "" for x in row):
            continue
        start = 0
        while start < len(row):
            chunk = row[start:start + group_size]
            if all(str(x).strip() == "" for x in chunk):
                break
            if len(chunk) < group_size:
                chunk = chunk + [""] * (group_size - len(chunk))
            rec = {
                "process_step": chunk[0],
                "nitrogen": chunk[1],
                "nitrogen_sd": chunk[2],
                "ash": chunk[3],
                "ash_sd": chunk[4],
                "volatiles": chunk[5],
                "volatiles_sd": chunk[6],
                "impurity": chunk[7],
                "impurity_sd": chunk[8],
                "mooney": chunk[9],
                "mooney_sd": chunk[10],
                "p0": chunk[11],
                "p0_sd": chunk[12],
                "pri": chunk[13],
                "pri_sd": chunk[14],
            }
            records.append(rec)
            start += group_size
    df = pd.DataFrame(records)
    num_cols = [c for c in df.columns if c != "process_step"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna()

def parse_acid_csv(raw_text):
    records = []
    reader = csv.reader(StringIO(raw_text.strip()))
    rows = list(reader)
    if not rows:
        return pd.DataFrame()
    data_rows = rows[1:]
    group_size = 16
    for row in data_rows:
        if not row or all(str(x).strip() == "" for x in row):
            continue
        start = 0
        while start < len(row):
            chunk = row[start:start + group_size]
            if all(str(x).strip() == "" for x in chunk):
                break
            if len(chunk) < group_size:
                chunk = chunk + [""] * (group_size - len(chunk))
            rec = {
                "process_step": chunk[0],
                "nitrogen": chunk[1],
                "nitrogen_sd": chunk[2],
                "ash": chunk[3],
                "ash_sd": chunk[4],
                "volatiles": chunk[5],
                "volatiles_sd": chunk[6],
                "impurity": chunk[7],
                "impurity_sd": chunk[8],
                "mooney": chunk[9],
                "mooney_sd": chunk[10],
                "p0": chunk[11],
                "p0_sd": chunk[12],
                "pri": chunk[13],
                "pri_sd": chunk[14],
            }
            records.append(rec)
            start += group_size
    df = pd.DataFrame(records)
    num_cols = [c for c in df.columns if c != "process_step"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna()

# =========================
# 3. 构建 base_df + plasticity + dry_type
# =========================

def compute_plasticity(p0, pri):
    # 塑性指标：简单平均，可根据论文需要调整
    return 0.5 * p0 + 0.5 * pri

def build_master_dataframe():
    dfs = []

    df_nat = parse_horizontal_csv(RAW_CSV_NATURAL)
    df_nat["coag_type"] = "natural"
    dfs.append(df_nat)

    df_enzyme = parse_horizontal_csv(RAW_CSV_ENZYME)
    df_enzyme["coag_type"] = "enzyme"
    dfs.append(df_enzyme)

    df_acid = parse_acid_csv(RAW_CSV_ACID)
    df_acid["coag_type"] = "acid"
    dfs.append(df_acid)

    all_df = pd.concat(dfs, ignore_index=True)

    # 计算塑性
    all_df["plasticity"] = compute_plasticity(all_df["p0"], all_df["pri"])
    all_df["plasticity_sd"] = np.sqrt(
        (0.5 * all_df["p0_sd"]) ** 2 + (0.5 * all_df["pri_sd"]) ** 2
    )

    # 烘干类型 dry_type
    all_df["dry_type"] = "none"
    mask_pellet = (
        all_df["process_step"].str.lower() == "dry"
    ) & (all_df["coag_type"].isin(["acid", "natural"]))
    mask_sheet = (
        all_df["process_step"].str.lower() == "dry"
    ) & (all_df["coag_type"] == "enzyme")

    all_df.loc[mask_pellet, "dry_type"] = "pellet_drying"  # Pellet drying
    all_df.loc[mask_sheet, "dry_type"] = "sheet_drying"    # Sheet drying

    cols = [
        "coag_type",
        "process_step",
        "dry_type",
        "nitrogen", "nitrogen_sd",
        "ash", "ash_sd",
        "volatiles", "volatiles_sd",
        "impurity", "impurity_sd",
        "mooney", "mooney_sd",
        "plasticity", "plasticity_sd",
    ]
    return all_df[cols].copy()

# =========================
# 4. NNI 数据增强
# =========================

def nni_augmentation(df, target_per_class=1000, noise_scale=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    indicator_cols = ["nitrogen", "ash", "volatiles", "impurity", "mooney", "plasticity"]
    sd_cols = ["nitrogen_sd", "ash_sd", "volatiles_sd", "impurity_sd", "mooney_sd", "plasticity_sd"]
    dfs = []
    for ctype, grp in df.groupby("coag_type"):
        real = grp.copy().reset_index(drop=True)
        n_real = len(real)
        if n_real < 2:
            dfs.append(real)
            continue
        X = real[indicator_cols].values
        SD = real[sd_cols].values

        synth_records = list(real.to_dict(orient="records"))
        n_needed = target_per_class - n_real
        for _ in range(n_needed):
            i, j = rng.integers(0, n_real, size=2)
            t = rng.random()
            x_interp = X[i] + t * (X[j] - X[i])
            sd_mean = (SD[i] + SD[j]) / 2.0
            noise = rng.normal(0, sd_mean * noise_scale)
            x_aug = np.clip(x_interp + noise, 0, None)

            ps = real.loc[i, "process_step"] if rng.random() < 0.5 else real.loc[j, "process_step"]
            dry_type = real.loc[i, "dry_type"] if rng.random() < 0.5 else real.loc[j, "dry_type"]

            rec = {
                "coag_type": ctype,
                "process_step": ps,
                "dry_type": dry_type,
                "nitrogen": x_aug[0], "nitrogen_sd": sd_mean[0],
                "ash": x_aug[1], "ash_sd": sd_mean[1],
                "volatiles": x_aug[2], "volatiles_sd": sd_mean[2],
                "impurity": x_aug[3], "impurity_sd": sd_mean[3],
                "mooney": x_aug[4], "mooney_sd": sd_mean[4],
                "plasticity": x_aug[5], "plasticity_sd": sd_mean[5],
            }
            synth_records.append(rec)
        dfs.append(pd.DataFrame(synth_records))
    return pd.concat(dfs, ignore_index=True)

# =========================
# 5. BP 模型
# =========================

class BPClassifier(nn.Module):
    def __init__(self, input_dim=6, hidden1=32, hidden2=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_model(X_train, y_train, X_val, y_val, num_classes=3,
                epochs=200, lr=1e-3, device="cpu"):
    model = BPClassifier(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds_train = logits.argmax(dim=1)
            train_acc = (preds_train == y_train_t).float().mean().item()

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val_t)
            val_loss = criterion(logits_val, y_val_t).item()
            preds_val = logits_val.argmax(dim=1)
            val_acc = (preds_val == y_val_t).float().mean().item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}: "
                  f"Train Loss={loss.item():.4f}, Val Loss={val_loss:.4f}, "
                  f"Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

    return model, (train_losses, val_losses, train_accs, val_accs)

# =========================
# 6. 训练并保存
# =========================

def train_and_save():
    print("=== 构建基础数据 ===")
    base_df = build_master_dataframe()
    print("Original data size:", base_df.shape)
    print(base_df.groupby("coag_type")["process_step"].count())

    print("=== NNI 数据增强 ===")
    aug_df = nni_augmentation(base_df, target_per_class=1000, noise_scale=0.2)
    print("Augmented data size:", aug_df.shape)
    print(aug_df.groupby("coag_type")["process_step"].count())

    indicator_cols = ["nitrogen", "ash", "volatiles", "impurity", "mooney", "plasticity"]
    X = aug_df[indicator_cols].values
    coag_types = sorted(aug_df["coag_type"].unique())
    type_to_idx = {t: i for i, t in enumerate(coag_types)}
    y = aug_df["coag_type"].map(type_to_idx).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.1765, random_state=42, stratify=y_tmp
    )

    print("Train size:", X_train.shape[0],
          "Val size:", X_val.shape[0],
          "Test size:", X_test.shape[0])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model, _ = train_model(
        X_train, y_train, X_val, y_val,
        num_classes=len(coag_types),
        epochs=200,
        lr=1e-3,
        device=device
    )

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits_test = model(X_test_t)
        preds_test = logits_test.argmax(dim=1).cpu().numpy()
    test_acc = (preds_test == y_test).mean()
    print(f"Test Accuracy: {test_acc:.3f}")

    # 保存
    torch.save(model.state_dict(), "bp_model.pt")
    np.savez("scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    np.savez("label_map.npz", types=np.array(coag_types))
    with open("aug_data_with_drytype.pkl", "wb") as f:
        pickle.dump(aug_df, f)

    print("Saved: bp_model.pt, scaler.npz, label_map.npz, aug_data_with_drytype.pkl")

# =========================
# 7. 交互式推荐
# =========================

def load_model_and_data():
    with open("aug_data_with_drytype.pkl", "rb") as f:
        aug_df = pickle.load(f)
    scaler_data = np.load("scaler.npz")
    mean = scaler_data["mean"]
    scale = scaler_data["scale"]
    label_map = np.load("label_map.npz", allow_pickle=True)
    coag_types = label_map["types"].tolist()

    model = BPClassifier(input_dim=6, num_classes=len(coag_types))
    state_dict = torch.load("bp_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, mean, scale, coag_types, aug_df

def relative_error(x, x_target, eps=1e-6):
    # 综合相对误差：6 项指标相对误差的平均值
    x = np.asarray(x, dtype=float)
    t = np.asarray(x_target, dtype=float)
    denom = np.maximum(np.abs(t), eps)
    rel = np.abs(x - t) / denom
    return rel.mean()

def recommend_drying_strategy(best_type):
    """
    推荐破碎和烘干方式（英文描述）：
    - enzyme       -> Sheet drying   （挂片烘干）
    - acid/natural -> Pellet drying  （造粒烘干）
    """
    if best_type == "enzyme":
        return "Sheet drying"
    elif best_type in ["acid", "natural"]:
        return "Pellet drying"
    else:
        return "Unknown"

def show_example_route(best_type):
    # 根据凝固方式给出一条示例完整工艺路径（文字提示）
    if best_type == "natural":
        print("示例工艺路径：Glue → Broken → First granulation → crepe → Second granulation → dry (Pellet drying) → package")
    elif best_type == "acid":
        print("示例工艺路径：latex → coagulation → mature → thining → crepe → granulation → dry (Pellet drying) → package")
    elif best_type == "enzyme":
        print("示例工艺路径：latex → coagulation → mature → thining → crepe → dry (Sheet drying) → package")
    else:
        print("示例工艺路径：根据实际工艺另行确定。")

def interactive_recommend(top_k=5):
    if not (os.path.exists("bp_model.pt") and
            os.path.exists("scaler.npz") and
            os.path.exists("label_map.npz") and
            os.path.exists("aug_data_with_drytype.pkl")):
        print("未找到模型或数据文件，请先运行训练模式（选 1）。")
        return

    model, mean, scale, coag_types, aug_df = load_model_and_data()
    indicator_cols = ["nitrogen", "ash", "volatiles", "impurity", "mooney", "plasticity"]

    while True:
        print("\n请输入 6 项指标（氮含量 灰分含量 挥发分含量 杂质含量 门尼粘度 塑性初值），以空格分隔；输入 q 退出：")
        line = input().strip()
        if line.lower() in ["q", "quit", "exit"]:
            break
        parts = line.split()
        if len(parts) != 6:
            print("需要输入 6 个数，请重新输入。")
            continue

        try:
            x_target = np.array([float(p) for p in parts], dtype=float)
        except ValueError:
            print("输入中包含非数字，请重新输入。")
            continue

        # 标准化 + 预测
        x_scaled = (x_target - mean) / scale
        x_t = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(x_t)
            probs = torch.softmax(logits, dim=1).numpy().squeeze(0)

        # 1. 凝固方式概率
        print("\n工艺阶段性预测：")
        print("模型预测凝固方式概率：")
        for t, p in zip(coag_types, probs):
            print(f"  {t:8s}: {p:.3f}")
        best_idx = int(probs.argmax())
        best_type = coag_types[best_idx]

        # 2. 推断原料路径（方案 A）
        if best_type in ["acid", "enzyme"]:
            source_type = "latex"
        else:
            source_type = "glue"
        print(f"推断原料路径：{source_type}")

        # 3. 推荐凝固方式
        print(f"推荐凝固方式：{best_type}")

        # 4. 推荐破碎和烘干方式（英文）
        drying_strategy = recommend_drying_strategy(best_type)
        print(f"推荐破碎和烘干方式：{drying_strategy}")

        # 选择终点样本：dry 或 package
        mask_type = (aug_df["coag_type"] == best_type) & (
            aug_df["process_step"].str.lower().isin(["dry", "package"])
        )
        cand = aug_df.loc[mask_type].copy()
        if cand.empty:
            print("在该凝固方式下未找到 dry/package 终点样本。")
            continue

        X_cand = cand[indicator_cols].values
        dists = np.linalg.norm(X_cand - x_target[None, :], axis=1)
        order = np.argsort(dists)[:top_k]
        cand_top = cand.iloc[order].copy()
        cand_top["RelErr"] = [
            relative_error(row[indicator_cols].values, x_target) for _, row in cand_top.iterrows()
        ]

        print(f"\n与目标最接近的 {len(cand_top)} 条终点样本（{best_type}, {source_type} 路径）：")
        print(f"{'rank':<4} {'工艺阶段':<18} {'烘干类型':<15} "
              f"{'N':>6} {'Ash':>6} {'Vol':>6} {'Imp':>6} {'Mooney':>8} {'Plast':>8} {'RelErr':>8}")
        for rank, (_, row) in enumerate(cand_top.iterrows(), start=1):
            v = row[indicator_cols].values
            print(f"{rank:<4} {row['process_step']:<18} {row['dry_type']:<15} "
                  f"{v[0]:6.3f} {v[1]:6.3f} {v[2]:6.3f} {v[3]:6.3f} {v[4]:8.3f} {v[5]:8.3f} {row['RelErr']:8.3f}")

        print("\n提示：可以根据凝固方式 + 原料路径 + 烘干类型补全整条工艺路径。")
        show_example_route(best_type)

# =========================
# 8. 主入口
# =========================

if __name__ == "__main__":
    print("选择模式：")
    print("  1 = 训练模型并保存增强数据")
    print("  2 = 进入交互式反向工艺推荐")
    mode = input("请输入 1 或 2：").strip()
    if mode == "1":
        train_and_save()
    else:
        interactive_recommend(top_k=5)