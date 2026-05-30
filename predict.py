"""
上海房价预测 —— Codabench 提交脚本
====================================
用法: python3 predict.py --train train.csv --test test.csv --output predictions.csv

流程:
  1. 读取训练集和测试集
  2. 特征工程（缺失值处理、类别编码、文本特征提取等）
  3. 在训练集上训练回归模型
  4. 对测试集进行预测
  5. 输出 predictions.csv（包含 Id 和 house_price 两列）

模型: GradientBoostingRegressor（集成学习，鲁棒性好）
依赖: pandas, numpy, scikit-learn（均为 Codabench 环境已有库）
"""

import argparse
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# 1. 特征工程函数
# ============================================================

def _extract_title_features(df: pd.DataFrame) -> pd.DataFrame:
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

    # 标题长度（长短标题可能反映房源描述详细程度）
    result["title_len"] = title_series.str.len()
    return result


def _extract_orientation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从 orientation 字段提取朝向特征。
    中国房屋朝向价值经验: 南 > 南北 > 东/东南 > 西/西南 > 北
    """
    ori_series = df["orientation"].fillna("").astype(str).str.lower()

    result = pd.DataFrame(index=df.index)
    result["has_south"] = ori_series.str.contains("南").astype(int)
    result["has_north"] = ori_series.str.contains("北").astype(int)
    result["has_east"] = ori_series.str.contains("东").astype(int)
    result["has_west"] = ori_series.str.contains("西").astype(int)
    # 是否是南北通透（最佳朝向）
    result["is_cross_vent"] = (
        ori_series.str.contains("南") & ori_series.str.contains("北")
    ).astype(int)
    return result


def _compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
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

    # 总房间数
    result["total_rooms"] = bedrooms + livingrooms
    # 每平米对应的房间数（密度）
    result["rooms_per_sqm"] = (bedrooms + livingrooms) / (area + 1e-6)
    # 平均每间房面积
    result["area_per_room"] = area / (bedrooms + livingrooms + 1e-6)
    # 卧室占比（卧室偏好）
    result["bedroom_ratio"] = bedrooms / (bedrooms + livingrooms + 1e-6)
    # 房屋年龄（2024 年作为基准，与 build_year 较为匹配）
    house_age = 2024 - build_year
    result["house_age"] = house_age.clip(0, 100)
    # 楼龄分段: 次新房(<=5年), 中年房(6-20年), 老房(>20年)
    result["is_new_building"] = (house_age <= 5).astype(int)
    result["is_old_building"] = (house_age > 20).astype(int)
    # floor_region 定序编码: 低 -> 中 -> 高
    floor_map = {"低": 0, "中": 1, "高": 2, "低区": 0, "中区": 1, "高区": 2}
    result["floor_region_ord"] = (
        df["floor_region"].astype(str).str.strip().map(floor_map).fillna(1)
    )
    # 楼层相对高度比 (floor_region * total_floor 的粗略估计)
    result["floor_height_ratio"] = (result["floor_region_ord"] + 1) / (
        total_floor + 1
    )

    return result


# 高基数类别使用频率编码（避免 OneHot 维度爆炸）
def _frequency_encode(
    train: pd.DataFrame, test: pd.DataFrame, col: str
) -> tuple:
    """频率编码: 用类别在训练集中的出现频次替代原始值"""
    freq = train[col].value_counts().to_dict()
    train_enc = train[col].map(freq).fillna(1)
    # 对测试集中未见过类别默认频次为 1
    test_enc = test[col].map(freq).fillna(1)
    return train_enc.values, test_enc.values


def _target_encode(
    train: pd.DataFrame, test: pd.DataFrame, col: str, target: pd.Series
) -> tuple:
    """
    目标编码: 用类别的目标均值替代原始值。
    加入全局平滑避免过拟合。
    """
    global_mean = target.mean()
    # 计算每个类别的均值
    agg = train.groupby(col)[target.name].agg(["mean", "count"])
    # 平滑: (count * mean + global_mean) / (count + 1)
    smoothed = (agg["mean"] * agg["count"] + global_mean) / (agg["count"] + 1)
    encoding_map = smoothed.to_dict()

    train_enc = train[col].map(encoding_map).fillna(global_mean)
    test_enc = test[col].map(encoding_map).fillna(global_mean)
    return train_enc.values, test_enc.values


# ============================================================
# 2. 主预测函数
# ============================================================

def predict(train_path: str, test_path: str, output_path: str) -> None:
    """
    完整预测流程: 读取 -> 处理 -> 训练 -> 预测 -> 输出
    """
    print(f"[INFO] 读取训练集: {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"[INFO] 读取测试集: {test_path}")
    test_df = pd.read_csv(test_path)

    # 保存 Id
    test_ids = test_df["Id"].copy()

    # ================================================================
    # 特征工程
    # ================================================================
    print("[INFO] 特征工程...")

    # --- title 文本特征 ---
    title_train = _extract_title_features(train_df)
    title_test = _extract_title_features(test_df)

    # --- orientation 朝向特征 ---
    ori_train = _extract_orientation_features(train_df)
    ori_test = _extract_orientation_features(test_df)

    # --- 数值比例特征 ---
    ratio_train = _compute_ratio_features(train_df)
    ratio_test = _compute_ratio_features(test_df)

    # --- 高基数类别变量: community, subdistrict ---
    # 使用频率编码（简单且不易过拟合）
    community_train, community_test = _frequency_encode(
        train_df, test_df, "community"
    )
    subdistrict_train, subdistrict_test = _frequency_encode(
        train_df, test_df, "subdistrict"
    )

    # --- district 采用目标编码 ---
    target_series = train_df["house_price"]
    district_train, district_test = _target_encode(
        train_df, test_df, "district", target_series
    )

    # --- floor_region 和 orientation 原始值保留给低频 OneHot ---
    floor_region_train = train_df["floor_region"].fillna("未知").astype(str)
    floor_region_test = test_df["floor_region"].fillna("未知").astype(str)

    # --- 原始数值特征 ---
    numeric_cols = ["bedrooms", "livingrooms", "area_sqm", "total_floor", "build_year"]
    numeric_train = train_df[numeric_cols].copy()
    numeric_test = test_df[numeric_cols].copy()
    # 只替换非数值为 NaN，再统一填充中位数
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
            community_train,
            subdistrict_train,
            district_train,
        ]
    )

    X_test = np.column_stack(
        [
            numeric_test.values,
            ratio_test.values,
            title_test.values,
            ori_test.values,
            community_test,
            subdistrict_test,
            district_test,
        ]
    )

    # 对 floor_region 进行 OneHot 编码
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
    # 模型训练
    # ================================================================
    print("[INFO] 训练 GradientBoostingRegressor ...")

    model = GradientBoostingRegressor(
        n_estimators=500,  # 树的数量
        learning_rate=0.05,  # 学习率
        max_depth=5,  # 每棵树最大深度
        min_samples_leaf=10,  # 叶节点最少样本数
        min_samples_split=20,  # 内部节点最少样本数
        subsample=0.8,  # 行采样，防止过拟合
        max_features="sqrt",  # 列采样
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train)

    # ================================================================
    # 预测
    # ================================================================
    print("[INFO] 预测测试集...")
    y_pred = model.predict(X_test)

    # 确保预测值非负（房价不可能为负）
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
