"""
上海房价预测 —— 本地训练脚本
==============================
功能:
  1. 读取 train.csv，执行探索性数据分析 (EDA)
  2. 特征工程（与 predict.py 保持一致）
  3. 训练多种回归模型并对比
  4. 交叉验证评估
  5. 保存模型（可选，用于 predict.py 加载）

运行: python train_model.py
依赖: pandas, numpy, matplotlib, seaborn, scikit-learn
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import joblib

warnings.filterwarnings("ignore")

# ============================================================
# 0. 配置
# ============================================================
DATA_PATH = "train.csv"  # 从 Codabench 下载的公开训练集
MODEL_SAVE_PATH = "model.pkl"
RANDOM_STATE = 42


# ============================================================
# 1. 数据读取与探索
# ============================================================

def load_and_explore(path: str) -> pd.DataFrame:
    """加载数据并打印基本信息"""
    print("=" * 60)
    print("1. 数据加载与探索")
    print("=" * 60)

    df = pd.read_csv(path)
    print(f"\n数据集形状: {df.shape}")
    print(f"\n前 5 行:\n{df.head()}")
    print(f"\n字段信息:")
    print(df.info())
    print(f"\n基本统计量:\n{df.describe()}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")

    # 目标变量分布
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df["house_price"], bins=80, kde=True)
    plt.title("house_price 分布")
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df["house_price"])
    plt.title("house_price 箱线图")
    plt.tight_layout()
    plt.savefig("eda_target_distribution.png", dpi=150)
    plt.close()
    print("\n[已保存] eda_target_distribution.png")

    # 类别字段频次
    cat_cols = ["district", "floor_region", "orientation"]
    for col in cat_cols:
        print(f"\n{col} 前 10 类:\n{df[col].value_counts().head(10)}")

    # 数值相关性
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Id" in numeric_cols:
        numeric_cols.remove("Id")
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("数值特征相关性热力图")
    plt.tight_layout()
    plt.savefig("eda_correlation.png", dpi=150)
    plt.close()
    print("\n[已保存] eda_correlation.png")

    return df


# ============================================================
# 2. 特征工程（与 predict.py 保持一致）
# ============================================================

def extract_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 title 字段提取关键词特征。
    上海二手房标题中常见关键词及其对房价的影响：
      - 地铁: 近地铁房，通常有溢价
      - 精装/豪装: 装修品质高
      - 电梯: 有电梯 (影响楼层便捷性)
      - 学区: 学区房，通常有显著溢价
      - 拎包: 拎包入住，装修齐全
      - 满五/满二: 税费相关
    """
    title_series = df["title"].fillna("").astype(str).str.lower()

    keywords = {
        "has_subway": "地铁",
        "has_decorated": "精装|豪装",
        "has_elevator": "电梯",
        "has_school": "学区|学校",
        "has_movein": "拎包",
        "has_mature": "满五|满二|满",
        "has_garden": "花园|景观",
        "has_quiet": "安静|不临街",
        "has_corner": "边套|全明",
    }

    result = pd.DataFrame(index=df.index)
    for col_name, pattern in keywords.items():
        result[col_name] = title_series.str.contains(pattern, regex=True).astype(int)

    result["title_len"] = title_series.str.len()
    return result


def extract_orientation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 orientation 字段提取朝向特征。
    朝向价值: 南 > 南北 > 东/东南 > 西/西南 > 北
    """
    ori_series = df["orientation"].fillna("").astype(str).str.lower()

    result = pd.DataFrame(index=df.index)
    result["has_south"] = ori_series.str.contains("南").astype(int)
    result["has_north"] = ori_series.str.contains("北").astype(int)
    result["has_east"] = ori_series.str.contains("东").astype(int)
    result["has_west"] = ori_series.str.contains("西").astype(int)
    result["is_cross_vent"] = (
        ori_series.str.contains("南") & ori_series.str.contains("北")
    ).astype(int)
    return result


def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算比例和派生数值特征。
    """
    result = pd.DataFrame(index=df.index)

    bedrooms = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(2)
    livingrooms = pd.to_numeric(df["livingrooms"], errors="coerce").fillna(1)
    area = pd.to_numeric(df["area_sqm"], errors="coerce").fillna(
        df["area_sqm"].median()
    )
    total_floor = pd.to_numeric(df["total_floor"], errors="coerce").fillna(
        df["total_floor"].median()
    )
    build_year = pd.to_numeric(df["build_year"], errors="coerce").fillna(2000)

    result["total_rooms"] = bedrooms + livingrooms
    result["rooms_per_sqm"] = (bedrooms + livingrooms) / (area + 1e-6)
    result["area_per_room"] = area / (bedrooms + livingrooms + 1e-6)
    result["bedroom_ratio"] = bedrooms / (bedrooms + livingrooms + 1e-6)
    house_age = 2024 - build_year
    result["house_age"] = house_age.clip(0, 100)
    result["is_new_building"] = (house_age <= 5).astype(int)
    result["is_old_building"] = (house_age > 20).astype(int)
    floor_map = {"低": 0, "中": 1, "高": 2, "低区": 0, "中区": 1, "高区": 2}
    result["floor_region_ord"] = (
        df["floor_region"].astype(str).str.strip().map(floor_map).fillna(1)
    )
    result["floor_height_ratio"] = (result["floor_region_ord"] + 1) / (
        total_floor + 1
    )
    return result


def frequency_encode(train, test, col):
    """频率编码"""
    freq = train[col].value_counts().to_dict()
    train_enc = train[col].map(freq).fillna(1)
    test_enc = test[col].map(freq).fillna(1)
    return train_enc.values, test_enc.values


def target_encode(train, test, col, target):
    """目标编码（含平滑）"""
    global_mean = target.mean()
    agg = train.groupby(col)[target.name].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean) / (agg["count"] + 1)
    encoding_map = smoothed.to_dict()
    train_enc = train[col].map(encoding_map).fillna(global_mean)
    test_enc = test[col].map(encoding_map).fillna(global_mean)
    return train_enc.values, test_enc.values


def build_features(df, target=None, is_train=True, ref_freq=None, ref_target=None):
    """
    构建完整特征矩阵。
    如果是训练集 (is_train=True)，需要 target 来进行目标编码。
    如果是测试集 (is_train=False)，需要传入 ref_freq 和 ref_target 引用。
    """
    title_feats = extract_title_features(df)
    ori_feats = extract_orientation_features(df)
    ratio_feats = compute_ratio_features(df)

    if is_train:
        community_enc, _ = frequency_encode(df, df, "community")
        subdistrict_enc, _ = frequency_encode(df, df, "subdistrict")
        district_enc, _ = target_encode(df, df, "district", target)
        ref_freq = {
            "community": df["community"].value_counts().to_dict(),
            "subdistrict": df["subdistrict"].value_counts().to_dict(),
        }
        # 保存目标编码映射
        global_mean = target.mean()
        agg = df.groupby("district")[target.name].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + global_mean) / (agg["count"] + 1)
        ref_target = smoothed.to_dict()
    else:
        freq_c = ref_freq.get("community", {})
        freq_s = ref_freq.get("subdistrict", {})
        community_enc = df["community"].map(freq_c).fillna(1).values
        subdistrict_enc = df["subdistrict"].map(freq_s).fillna(1).values
        district_enc = df["district"].map(ref_target).fillna(target.mean()).values

    numeric_cols = ["bedrooms", "livingrooms", "area_sqm", "total_floor", "build_year"]
    numeric_data = df[numeric_cols].copy()
    for col in numeric_cols:
        numeric_data[col] = pd.to_numeric(numeric_data[col], errors="coerce")

    if is_train:
        medians = numeric_data.median()
        numeric_data = numeric_data.fillna(medians)
    else:
        # 在实际使用中可以从外部传入 medians
        numeric_data = numeric_data.fillna(numeric_data.median())

    X = np.column_stack(
        [
            numeric_data.values,
            ratio_feats.values,
            title_feats.values,
            ori_feats.values,
            community_enc,
            subdistrict_enc,
            district_enc,
        ]
    )

    # floor_region OneHot
    floor_series = df["floor_region"].fillna("未知").astype(str)
    if is_train:
        all_floor = floor_series.unique()
    else:
        all_floor = ref_freq.get("floor_cats", floor_series.unique())

    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", categories=[all_floor])
    floor_ohe = ohe.fit_transform(floor_series.values.reshape(-1, 1))
    X = np.column_stack([X, floor_ohe])

    if is_train:
        return X, ref_freq, ref_target, medians, all_floor, ohe

    return X


# ============================================================
# 3. 模型训练与评估
# ============================================================

def evaluate_model(model, X_train, y_train, X_val, y_val, name):
    """训练并评估单个模型"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"  {name:35s} | RMSE: {rmse:8.2f} | MAE: {mae:8.2f} | R2: {r2:.4f}")
    return {"name": name, "rmse": rmse, "mae": mae, "r2": r2, "model": model}


def train_and_select(X, y):
    """训练多个模型，选择最优"""
    print("\n" + "=" * 60)
    print("3. 模型训练与选择")
    print("=" * 60)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = []

    # ---- 线性模型 ----
    models = [
        ("Ridge(alpha=1.0)", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ("Ridge(alpha=10.0)", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
        ("ElasticNet(l1=0.5)", ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=RANDOM_STATE)),
    ]

    # ---- 树集成模型 ----
    models += [
        ("RandomForest(n=200)", RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_leaf=5,
            n_jobs=-1, random_state=RANDOM_STATE
        )),
        ("RandomForest(n=400)", RandomForestRegressor(
            n_estimators=400, max_depth=18, min_samples_leaf=3,
            n_jobs=-1, random_state=RANDOM_STATE
        )),
        ("GBR(lr=0.1,n=200)", GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE
        )),
        ("GBR(lr=0.05,n=500)", GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE
        )),
        ("GBR(lr=0.03,n=800)", GradientBoostingRegressor(
            n_estimators=800, learning_rate=0.03, max_depth=5,
            min_samples_leaf=10, subsample=0.8, random_state=RANDOM_STATE
        )),
        ("HistGBR(default)", HistGradientBoostingRegressor(
            max_depth=6, min_samples_leaf=10, random_state=RANDOM_STATE
        )),
    ]

    # ---- SVR (需要先标准化) ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # ---- 训练与评估 ----
    for name, model in models:
        if "SVR" in name:
            result = evaluate_model(model, X_train_s, y_train, X_val_s, y_val, name)
        else:
            result = evaluate_model(model, X_train, y_train, X_val, y_val, name)
        results.append(result)

    # 找最优模型
    best = min(results, key=lambda r: r["rmse"])
    print(f"\n{'=' * 60}")
    print(f"最优模型: {best['name']}")
    print(f"  RMSE: {best['rmse']:.2f}")
    print(f"  MAE : {best['mae']:.2f}")
    print(f"  R2  : {best['r2']:.4f}")

    return best


# ============================================================
# 4. 交叉验证
# ============================================================

def cross_validate_best(X, y, model, name):
    """对最优模型进行 K 折交叉验证"""
    print("\n" + "=" * 60)
    print("4. 交叉验证")
    print("=" * 60)

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmse_scores = []
    mae_scores = []
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        model_clone = (
            joblib.load(MODEL_SAVE_PATH) if os.path.exists(MODEL_SAVE_PATH)
            else model.__class__(**model.get_params())
        )
        # 如果 model 本身是已训练的 clone
        from copy import deepcopy
        m = deepcopy(model)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_vl)

        rmse = np.sqrt(mean_squared_error(y_vl, pred))
        mae = mean_absolute_error(y_vl, pred)
        r2 = r2_score(y_vl, pred)

        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        print(f"  Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

    print(f"\n  5-Fold CV 平均:")
    print(f"    RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}")
    print(f"    MAE : {np.mean(mae_scores):.2f} ± {np.std(mae_scores):.2f}")
    print(f"    R2  : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

    return np.mean(rmse_scores)


# ============================================================
# 5. 特征重要性分析
# ============================================================

def analyze_feature_importance(model, feature_names):
    """分析特征重要性并绘图"""
    print("\n" + "=" * 60)
    print("5. 特征重要性分析")
    print("=" * 60)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\n特征重要性排名:")
        for i in range(min(20, len(feature_names))):
            print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

        plt.figure(figsize=(10, 8))
        plt.title("特征重要性 (Top 20)")
        plt.barh(range(min(20, len(feature_names))), importances[indices][:20][::-1])
        plt.yticks(
            range(min(20, len(feature_names))),
            [feature_names[i] for i in indices[:20][::-1]],
        )
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.close()
        print("\n[已保存] feature_importance.png")
    else:
        print("该模型不支持 feature_importances_ 属性")


# ============================================================
# 6. 全量训练与模型保存
# ============================================================

def train_final_model(X, y, best_model):
    """在全部数据上重新训练并保存模型"""
    print("\n" + "=" * 60)
    print("6. 全量训练并保存模型")
    print("=" * 60)

    final_model = best_model.__class__(**best_model.get_params())
    final_model.fit(X, y)

    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"  模型已保存至: {MODEL_SAVE_PATH}")

    # 验证保存的模型可以正常加载
    loaded = joblib.load(MODEL_SAVE_PATH)
    sample_pred = loaded.predict(X[:5])
    print(f"  加载验证 - 前 5 条预测值: {sample_pred}")

    return final_model


# ============================================================
# 主流程
# ============================================================

def main():
    # Step 1: 加载与 EDA
    df = load_and_explore(DATA_PATH)

    # Step 2: 特征工程
    print("\n" + "=" * 60)
    print("2. 特征工程")
    print("=" * 60)

    # 分离特征和目标
    y = df["house_price"].values
    # 构建特征
    X, ref_freq, ref_target, medians, all_floor, _ = build_features(df, df["house_price"], is_train=True)

    # 特征名（用于可视化）
    numeric_names = ["bedrooms", "livingrooms", "area_sqm", "total_floor", "build_year"]
    ratio_names = [
        "total_rooms", "rooms_per_sqm", "area_per_room", "bedroom_ratio",
        "house_age", "is_new_building", "is_old_building",
        "floor_region_ord", "floor_height_ratio",
    ]
    title_names = [
        "has_subway", "has_decorated", "has_elevator", "has_school",
        "has_movein", "has_mature", "has_garden", "has_quiet",
        "has_corner", "title_len",
    ]
    ori_names = ["has_south", "has_north", "has_east", "has_west", "is_cross_vent"]
    enc_names = ["community_freq", "subdistrict_freq", "district_target"]
    floor_names = [f"floor_{c}" for c in all_floor]
    feature_names = numeric_names + ratio_names + title_names + ori_names + enc_names + floor_names

    print(f"  特征矩阵形状: {X.shape}")
    print(f"  特征数量: {X.shape[1]}")

    # Step 3: 训练与选择
    best_result = train_and_select(X, y)

    # Step 4: 交叉验证
    cv_rmse = cross_validate_best(X, y, best_result["model"], best_result["name"])

    # Step 5: 特征重要性
    best_result["model"].fit(X, y)
    analyze_feature_importance(best_result["model"], feature_names)

    # Step 6: 全量训练并保存
    final_model = train_final_model(X, y, best_result["model"])

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  模型已保存至: {MODEL_SAVE_PATH}")
    print(f"  5-Fold CV RMSE: {cv_rmse:.2f}")
    print("=" * 60)

    print("\n【后续步骤】")
    print(f"  1. 将 {MODEL_SAVE_PATH} 和 predict.py 一起打包为 zip")
    print("  2. 在 Codabench 提交 zip 文件")
    print("  3. 平台会自动运行: python3 predict.py --train train.csv --test test.csv --output predictions.csv")


if __name__ == "__main__":
    main()
