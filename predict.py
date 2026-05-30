"""
上海房价预测 —— Codabench 提交脚本 (Phase 2)
==============================================
用法: python3 predict.py --train train.csv --test test.csv --output predictions.csv

流程:
  1. 读取训练集和测试集
  2. 特征工程（缺失值处理、类别编码、文本特征提取等）
  3. 在训练集上训练回归模型（目标: log(price_per_sqm)）
  4. 对测试集进行预测并还原为总价
  5. 输出 predictions.csv（包含 Id 和 house_price 两列）

模型: 优先加载预训练 model.pkl / model_stacking.pkl
      若无预训练模型则训练 RandomForestRegressor

依赖: pandas, numpy, scikit-learn, xgboost (可选)
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PPS_LOWER_PERCENTILE = 0.5
PPS_UPPER_PERCENTILE = 99.5
AREA_LOWER_PERCENTILE = 0.5
AREA_UPPER_PERCENTILE = 99.5

# ============================================================
# 1. 特征工程函数（与 train_model.py 保持同步）
# ============================================================

def _clean_district(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 district 字段: 去掉 '二手房' 后缀"""
    df = df.copy()
    df["district"] = df["district"].astype(str).str.strip()
    df["district"] = df["district"].str.replace("二手房", "", regex=False)
    return df


def _extract_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 title 字段提取关键词特征（18 个关键词）。"""
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
        "has_luxury": "豪华装修",
        "has_simple": "简单装修",
        "has_lianjia": "链家好房",
        "has_full_light": "全明",
        "has_original": "原装",
        "has_brand": "品牌",
        "has_view": "景观",
        "has_traffic": "交通",
        "has_tax_free": "满五|满二",
    }

    result = pd.DataFrame(index=df.index)
    for col_name, pattern in keywords.items():
        result[col_name] = title_series.str.contains(pattern, regex=True).astype(int)

    result["title_len"] = title_series.str.len()
    return result


def _extract_orientation_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 orientation 字段提取朝向分组特征（5 组）。"""
    ori_series = df["orientation"].fillna("").astype(str).str.lower()

    result = pd.DataFrame(index=df.index)
    result["ori_south_north"] = (
        ori_series.str.contains("南") & ori_series.str.contains("北")
    ).astype(int)
    result["ori_south"] = (
        ori_series.str.contains("南") & ~ori_series.str.contains("北")
    ).astype(int)
    result["ori_east_west"] = (
        ~ori_series.str.contains("南")
        & ~ori_series.str.contains("北")
        & (ori_series.str.contains("东") | ori_series.str.contains("西")
           | ori_series.str.contains("东南") | ori_series.str.contains("西南")
           | ori_series.str.contains("东北") | ori_series.str.contains("西北"))
    ).astype(int)
    result["ori_north"] = (
        ori_series.str.contains("北") & ~ori_series.str.contains("南")
    ).astype(int)
    result["ori_unknown"] = (
        (ori_series == "") | ori_series.isna()
        | ori_series.str.contains("进门", na=False)
        | (~result["ori_south_north"].astype(bool)
           & ~result["ori_south"].astype(bool)
           & ~result["ori_east_west"].astype(bool)
           & ~result["ori_north"].astype(bool))
    ).astype(int)
    return result


def _compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """计算比例和派生数值特征（含 log 变换）。"""
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

    result["area_per_bedroom"] = area / (bedrooms + 1e-6)

    floor_bucket = pd.cut(
        total_floor,
        bins=[0, 6, 18, 200],
        labels=["low_rise", "mid_rise", "high_rise"],
    )
    result["floor_bucket_low"] = (floor_bucket == "low_rise").astype(int)
    result["floor_bucket_mid"] = (floor_bucket == "mid_rise").astype(int)
    result["floor_bucket_high"] = (floor_bucket == "high_rise").astype(int)
    result["floor_ratio_x_total"] = result["floor_height_ratio"] * total_floor

    # ---- Log 变换特征 ----
    result["log_area_sqm"] = np.log(area + 1e-6)
    result["log_total_rooms"] = np.log(result["total_rooms"] + 1e-6)
    result["log_area_per_bedroom"] = np.log(result["area_per_bedroom"] + 1e-6)

    return result


def _detect_villa(df: pd.DataFrame) -> pd.DataFrame:
    """识别别墅: floor_region 缺失 且 总楼层 <= 3"""
    floor_region_missing = df["floor_region"].isna()
    total_floor_low = pd.to_numeric(df["total_floor"], errors="coerce").fillna(99) <= 3
    return pd.DataFrame({
        "is_villa": (floor_region_missing & total_floor_low).astype(int)
    }, index=df.index)


def _bucket_build_year(df: pd.DataFrame) -> pd.DataFrame:
    """build_year 分桶: pre-1950, 1950-1990, 1990-2000, 2000-2010, 2010+, unknown"""
    by = pd.to_numeric(df["build_year"], errors="coerce")

    result = pd.DataFrame(index=df.index)
    result["by_pre1950"] = ((by < 1950) & by.notna()).astype(int)
    result["by_1950_1990"] = ((by >= 1950) & (by < 1990)).astype(int)
    result["by_1990_2000"] = ((by >= 1990) & (by < 2000)).astype(int)
    result["by_2000_2010"] = ((by >= 2000) & (by < 2010)).astype(int)
    result["by_2010plus"] = (by >= 2010).astype(int)
    result["by_unknown"] = by.isna().astype(int)
    return result


def _frequency_encode(train: pd.DataFrame, test: pd.DataFrame, col: str) -> tuple:
    """频率编码"""
    freq = train[col].value_counts().to_dict()
    train_enc = train[col].map(freq).fillna(1)
    test_enc = test[col].map(freq).fillna(1)
    return train_enc.values, test_enc.values


def _target_encode(train: pd.DataFrame, test: pd.DataFrame,
                   col: str, target: pd.Series, smooth: int = 1) -> tuple:
    """目标编码（含平滑）"""
    global_mean = target.mean()
    agg = train.groupby(col)[target.name].agg(["mean", "count"])
    smoothed = (agg["mean"] * agg["count"] + global_mean * smooth) / (agg["count"] + smooth)
    encoding_map = smoothed.to_dict()
    train_enc = train[col].map(encoding_map).fillna(global_mean)
    test_enc = test[col].map(encoding_map).fillna(global_mean)
    return train_enc.values, test_enc.values


def _get_interaction_features(df_train, df_test):
    """计算交互特征: district_rooms, by_district"""
    # district × total_rooms
    le_d = LabelEncoder()
    district_all = pd.concat([
        df_train["district"].astype(str).str.strip().str.replace("二手房", "", regex=False),
        df_test["district"].astype(str).str.strip().str.replace("二手房", "", regex=False),
    ])
    le_d.fit(district_all)

    for df, name in [(df_train, "train"), (df_test, "test")]:
        bedrooms = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(2)
        livingrooms = pd.to_numeric(df["livingrooms"], errors="coerce").fillna(1)
        total_rooms = bedrooms + livingrooms
        district_clean = df["district"].astype(str).str.strip().str.replace("二手房", "", regex=False)
        district_num = le_d.transform(district_clean)
        if name == "train":
            train_dr = district_num * total_rooms
            train_by = district_num * pd.to_numeric(df["build_year"], errors="coerce").fillna(2000)
        else:
            test_dr = district_num * total_rooms
            test_by = district_num * pd.to_numeric(df["build_year"], errors="coerce").fillna(2000)

    return (train_dr.values, train_by.values), (test_dr.values, test_by.values)


# ============================================================
# 2. 主预测函数
# ============================================================

def predict(train_path: str, test_path: str, output_path: str) -> None:
    """
    完整预测流程: 读取 -> 特征工程 -> 训练 -> 预测 -> 输出
    """
    print(f"[INFO] 读取训练集: {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"[INFO] 读取测试集: {test_path}")
    test_df = pd.read_csv(test_path)

    # 保存测试集 Id 和 area_sqm（用于还原总价）
    test_ids = test_df["Id"].copy()
    test_area = pd.to_numeric(test_df["area_sqm"], errors="coerce").fillna(
        train_df["area_sqm"].median()
    )

    # ================================================================
    # 数据清洗
    # ================================================================
    print("[INFO] 数据清洗...")

    # 1. 清洗 district 字段
    train_df = _clean_district(train_df)
    test_df = _clean_district(test_df)

    # 2. 删除训练集中可疑记录: bedrooms==1 AND area_sqm>200
    bedrooms_num = pd.to_numeric(train_df["bedrooms"], errors="coerce")
    suspicious = (bedrooms_num == 1) & (train_df["area_sqm"] > 200)
    if suspicious.sum() > 0:
        print(f"[INFO] 删除 {suspicious.sum()} 条可疑记录 (bedrooms==1 & area_sqm>200)")
        train_df = train_df[~suspicious].reset_index(drop=True)

    # 3. area_sqm winsorization
    area_train = pd.to_numeric(train_df["area_sqm"], errors="coerce")
    area_lower = np.percentile(area_train.dropna(), AREA_LOWER_PERCENTILE)
    area_upper = np.percentile(area_train.dropna(), AREA_UPPER_PERCENTILE)
    train_df["area_sqm"] = np.clip(area_train, area_lower, area_upper)
    test_df["area_sqm"] = np.clip(
        pd.to_numeric(test_df["area_sqm"], errors="coerce"), area_lower, area_upper
    )
    test_area = np.clip(test_area, area_lower, area_upper)

    # ================================================================
    # 计算 PPS 目标变量
    # ================================================================
    train_pps = (
        pd.to_numeric(train_df["house_price"], errors="coerce")
        / pd.to_numeric(train_df["area_sqm"], errors="coerce")
    )
    # 双侧裁剪
    pps_lower = np.percentile(train_pps, PPS_LOWER_PERCENTILE)
    pps_upper = np.percentile(train_pps, PPS_UPPER_PERCENTILE)
    train_pps_clipped = np.clip(train_pps, pps_lower, pps_upper)
    train_log_pps = np.log(train_pps_clipped)
    train_df["log_pps"] = train_log_pps
    target_series = train_df["log_pps"]
    print(f"[INFO] PPS 目标: 裁剪范围=[{pps_lower:.2f}, {pps_upper:.2f}]")

    # ================================================================
    # 特征工程
    # ================================================================
    print("[INFO] 特征工程...")

    # --- Villa 识别 ---
    villa_train = _detect_villa(train_df)
    villa_test = _detect_villa(test_df)

    # --- build_year 分桶 ---
    by_train = _bucket_build_year(train_df)
    by_test = _bucket_build_year(test_df)

    # --- title 文本特征 ---
    title_train = _extract_title_features(train_df)
    title_test = _extract_title_features(test_df)

    # --- orientation 朝向分组特征 ---
    ori_train = _extract_orientation_features(train_df)
    ori_test = _extract_orientation_features(test_df)

    # --- 数值比例特征（含 log 变换）---
    ratio_train = _compute_ratio_features(train_df)
    ratio_test = _compute_ratio_features(test_df)

    # --- 高基数类别变量: community / subdistrict 频率编码 ---
    community_train, community_test = _frequency_encode(
        train_df, test_df, "community"
    )
    subdistrict_train, subdistrict_test = _frequency_encode(
        train_df, test_df, "subdistrict"
    )

    # --- district 目标编码 ---
    district_train_enc, district_test_enc = _target_encode(
        train_df, test_df, "district", target_series, smooth=1
    )

    # --- subdistrict 目标编码 ---
    subdistrict_target_train, subdistrict_target_test = _target_encode(
        train_df, test_df, "subdistrict", target_series, smooth=10
    )

    # --- 交互特征 ---
    (train_dr, train_by), (test_dr, test_by) = _get_interaction_features(train_df, test_df)

    # --- floor_region OneHot ---
    floor_region_train = train_df["floor_region"].fillna("未知").astype(str)
    floor_region_test = test_df["floor_region"].fillna("未知").astype(str)

    # --- 原始数值特征 ---
    numeric_cols = ["bedrooms", "livingrooms", "area_sqm", "total_floor", "build_year"]
    numeric_train = train_df[numeric_cols].copy()
    numeric_test = test_df[numeric_cols].copy()
    for col in numeric_cols:
        numeric_train[col] = pd.to_numeric(numeric_train[col], errors="coerce")
        numeric_test[col] = pd.to_numeric(numeric_test[col], errors="coerce")
    medians = numeric_train.median()
    numeric_train = numeric_train.fillna(medians)
    numeric_test = numeric_test.fillna(medians)

    # ================================================================
    # 组装特征矩阵
    # ================================================================
    X_train = np.column_stack(
        [
            numeric_train.values,
            ratio_train.values,
            title_train.values,
            ori_train.values,
            by_train.values,
            villa_train.values,
            community_train,
            subdistrict_train,
            district_train_enc,
            subdistrict_target_train,
            train_dr,
            train_by,
        ]
    )

    X_test = np.column_stack(
        [
            numeric_test.values,
            ratio_test.values,
            title_test.values,
            ori_test.values,
            by_test.values,
            villa_test.values,
            community_test,
            subdistrict_test,
            district_test_enc,
            subdistrict_target_test,
            test_dr,
            test_by,
        ]
    )

    # floor_region OneHot
    all_floor = pd.concat(
        [floor_region_train, floor_region_test], axis=0
    ).unique()
    ohe = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore", categories=[all_floor]
    )
    floor_ohe_train = ohe.fit_transform(floor_region_train.values.reshape(-1, 1))
    floor_ohe_test = ohe.transform(floor_region_test.values.reshape(-1, 1))

    X_train = np.column_stack([X_train, floor_ohe_train])
    X_test = np.column_stack([X_test, floor_ohe_test])

    y_train = target_series.values
    print(f"[INFO] 特征维度: {X_train.shape[1]}")

    # ================================================================
    # 优先尝试加载预训练模型，否则重新训练
    # ================================================================
    model = None

    # 尝试加载 stacking 模型
    stacking_path = "model_stacking.pkl"
    if os.path.exists(stacking_path):
        print(f"[INFO] 加载预训练 Stacking 模型: {stacking_path}")
        try:
            stacked = joblib.load(stacking_path)
            if isinstance(stacked, dict) and stacked.get("type") == "stacking":
                # Stacking 模型: 用 base models 预测，meta-learner 整合
                base_models = stacked["base_models"]
                meta_learner = stacked["meta_learner"]
                model_names = stacked["model_names"]
                print(f"[INFO] Stacking 模型加载成功: base={model_names}")

                all_preds = np.column_stack([m.predict(X_train) for m_name, m in base_models.items()])
                print(f"[INFO] Stacking meta features shape: {all_preds.shape}")

                # 定义 predict 函数闭包
                class StackingWrapper:
                    def __init__(self, base_models, meta_learner, model_names):
                        self.base_models = base_models
                        self.meta_learner = meta_learner
                        self.model_names = model_names

                    def predict(self, X):
                        meta_feats = np.column_stack([
                            m.predict(X) for m_name, m in self.base_models.items()
                        ])
                        return self.meta_learner.predict(meta_feats)

                model = StackingWrapper(base_models, meta_learner, model_names)
            else:
                print(f"[INFO] model_stacking.pkl 格式不匹配，尝试 model.pkl")
        except Exception as e:
            print(f"[INFO] 加载 stacking 模型失败 ({e})，尝试 model.pkl")

    # 尝试加载单模型
    model_path = "model.pkl"
    if model is None and os.path.exists(model_path):
        print(f"[INFO] 加载预训练模型: {model_path}")
        try:
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and loaded.get("type") == "stacking":
                base_models = loaded["base_models"]
                meta_learner = loaded["meta_learner"]
                model_names = loaded["model_names"]

                class StackingWrapper:
                    def __init__(self, base_models, meta_learner, model_names):
                        self.base_models = base_models
                        self.meta_learner = meta_learner
                        self.model_names = model_names

                    def predict(self, X):
                        meta_feats = np.column_stack([
                            m.predict(X) for m_name, m in self.base_models.items()
                        ])
                        return self.meta_learner.predict(meta_feats)

                model = StackingWrapper(base_models, meta_learner, model_names)
                print(f"[INFO] Stacking 模型从 model.pkl 加载成功")
            else:
                model = loaded
                print(f"[INFO] 单模型加载成功: {type(model).__name__}")
        except Exception as e:
            print(f"[INFO] 加载 model.pkl 失败 ({e})，重新训练")

    # 如果没有加载到模型，重新训练
    if model is None:
        print("[INFO] 训练模型...")
        try:
            import xgboost as xgb
            print("[INFO] 训练 XGBoost (目标: log(PPS)) ...")
            model = xgb.XGBRegressor(
                n_estimators=1000, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                early_stopping_rounds=50, eval_metric="rmse",
                n_jobs=-1, random_state=RANDOM_STATE, verbosity=0
            )
            model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
        except ImportError:
            print("[INFO] xgboost 未安装，训练 RandomForest ...")
            model = RandomForestRegressor(
                n_estimators=500, max_depth=20, min_samples_leaf=2,
                min_samples_split=4, n_jobs=-1, random_state=RANDOM_STATE
            )
            model.fit(X_train, y_train)

    # ================================================================
    # 预测并还原为总价
    # ================================================================
    print("[INFO] 预测测试集...")
    pred_log_pps = model.predict(X_test)
    # 还原: house_price = exp(pred_log_PPS) * area_sqm
    y_pred = np.exp(pred_log_pps) * test_area

    # 确保预测值非负
    y_pred = np.maximum(y_pred, 0.1)

    # ================================================================
    # 输出结果
    # ================================================================
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    result = pd.DataFrame({"Id": test_ids, "house_price": np.round(y_pred, 2)})
    result.to_csv(output_path, index=False)

    print(f"[INFO] 预测结果已保存至: {output_path}")
    print(f"[INFO] 共预测 {len(result)} 条样本")


# ============================================================
# 3. 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shanghai House Price Prediction")
    parser.add_argument("--train", type=str, required=True, help="训练集路径 (train.csv)")
    parser.add_argument("--test", type=str, required=True, help="测试集路径 (test.csv)")
    parser.add_argument("--output", type=str, required=True, help="输出路径 (predictions.csv)")
    args = parser.parse_args()

    predict(args.train, args.test, args.output)
