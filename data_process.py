#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
层次聚类（只用价格/汇率等“价格系”序列）→ 组β / 个体α
------------------------------------------------------
输入CSV格式：
- 第一列为日期（列名建议 'date'），其余每列为一个标的的价格或收益。
- 若是价格：将自动转为对数收益；若已是收益，请把 INPUT_IS_PRICES=False。

输出文件（带前缀 OUTPUT_PREFIX='price_'）：
- price_clusters.csv        每个标的所属簇（cluster）
- price_cluster_beta.csv    每个簇的组β（日收益等权平均）
- price_alpha_residuals.csv 每个标的的α残差（个体-组β）
- price_dendrogram.png      树状图（帮助你选簇数）

依赖：pandas、numpy、scipy、matplotlib
安装：pip install pandas numpy scipy matplotlib
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

# ===================== 用户可调参数 =====================
FILE_PATH = "train.csv"     # TODO: 改成你的CSV路径
INPUT_IS_PRICES = True          # True=输入为价格, 自动转对数收益; False=输入已是收益
DATE_COL = "date"               # 日期列名；若没有就把第一列当日期
N_CLUSTERS = 10                 # 目标分组数量（建议 9~12；本脚本默认 10）
LINKAGE_METHOD = "average"      # 'average'推荐；也可'complete'等（不要用'ward'，它需欧氏距离）
CORR_METHOD = "pearson"         # 相关方式：'pearson' 或 'spearman'
MIN_NON_NA_RATIO = 0.8          # 至少多少比例非缺失才保留该列（0.8=80%）
OUTPUT_PREFIX = "price_"        # 输出前缀，便于与旧版区分
RANDOM_SEED = 42
# ======================================================


def load_data(file_path: str, date_col: str = "date", input_is_prices: bool = True) -> pd.DataFrame:
    """
    读取CSV -> DataFrame（index=日期, columns=标的）
    如果是价格列，转成对数收益；如果本来就是收益，直接返回（并清理缺失）
    """
    df = pd.read_csv(file_path)

    # 识别日期列
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        # 若没指定列名，就把第一列当日期
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])

    # 去掉完全空值的列
    df = df.dropna(axis=1, how='all')

    if input_is_prices:
        # 价格先前向填充，减少缺口影响
        prices = df.sort_index().ffill().bfill()
        # 对数收益
        returns = np.log(prices).diff().dropna(how='all')
    else:
        # 已是收益：简单对齐，去掉全NaN行
        returns = df.sort_index().dropna(how='all')

    # 按“非缺失比例”筛列，太稀疏的列剔除
    non_na_ratio = returns.notna().mean(axis=0)
    keep_cols = non_na_ratio[non_na_ratio >= MIN_NON_NA_RATIO].index.tolist()
    returns = returns[keep_cols]

    # 剩余的缺失用0填（也可改成其它策略）
    returns = returns.fillna(0.0)
    return returns


def is_volume_like(colname: str) -> bool:
    """
    判断列名是否为“量/持仓类”，用于过滤：volume / turnover / open_interest / OI。
    小心不要误杀 'oil'（含 'oi' 的单词）。
    规则：
      - 包含 'volume', 'turnover', 'openinterest', 'open_interest'
      - 或者 'oi' 作为独立token出现（边界为非字母数字或字符串边界）
    """
    s = colname.lower()
    if ('volume' in s) or ('turnover' in s) or ('openinterest' in s) or ('open_interest' in s):
        return True
    # 匹配独立 token 'oi'，避免匹配到 'oil'
    # 说明：\w 不含标点；用 [_\W] 视为分隔符；也允许开头/结尾
    if re.search(r'(^|[^0-9a-zA-Z])oi([^0-9a-zA-Z]|$)', s):
        return True
    return False


def filter_price_columns(returns: pd.DataFrame) -> pd.DataFrame:
    """
    仅保留“价格系”列（价格/汇率/期货价等），剔除量/持仓类列
    """
    price_cols = [c for c in returns.columns if not is_volume_like(c)]
    filtered = returns[price_cols]
    if filtered.shape[1] < 2:
        raise ValueError("过滤后列数 < 2。请检查列名规则或输入数据。")
    return filtered


def corr_to_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """
    相关矩阵 -> 距离矩阵：d_ij = sqrt(2*(1 - rho_ij))
    """
    corr_clip = corr.clip(-1.0, 1.0)
    dist = np.sqrt(2.0 * (1.0 - corr_clip))
    np.fill_diagonal(dist.values, 0.0)
    return dist


def hierarchical_clustering(returns: pd.DataFrame,
                            linkage_method: str = "average",
                            corr_method: str = "pearson",
                            n_clusters: int = 10) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    根据收益的相关性做层次聚类，返回：
    - labels_df: 每个标的的簇标签
    - Z: linkage矩阵（可画树状图）
    - dist_condensed: 压缩距离向量（linkage用）
    """
    # 相关矩阵
    corr = returns.corr(method=corr_method)
    # 转距离矩阵 -> 压缩向量
    dist_mat = corr_to_distance(corr)
    dist_condensed = squareform(dist_mat.values, checks=False)

    # 层次聚类
    Z = linkage(dist_condensed, method=linkage_method)
    # 按簇数量切
    n_clusters = max(2, min(n_clusters, returns.shape[1]))  # 保护
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    labels_df = pd.DataFrame({
        "instrument": returns.columns,
        "cluster": labels
    }).sort_values(["cluster", "instrument"]).reset_index(drop=True)

    return labels_df, Z, dist_condensed


def plot_dendrogram(Z, labels: List[str], save_path: str = "dendrogram.png", max_d=None) -> None:
    """
    画树状图（可能较长），可选 max_d 为横切阈值（只画参考线）
    """
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8, color_threshold=None)
    if max_d is not None:
        plt.axhline(y=max_d, c='gray', lw=1, linestyle='--')
    plt.title("Hierarchical Clustering Dendrogram")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_cluster_beta_and_alpha(returns: pd.DataFrame, labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算：
    - cluster_beta_df：每个簇的组β（组内等权平均收益）
    - alpha_resid_df ：每个标的的α残差（个体 - 所属簇β）
    """
    clusters = sorted(labels_df["cluster"].unique().tolist())
    cluster_members: Dict[int, List[str]] = {
        c: labels_df.loc[labels_df["cluster"] == c, "instrument"].tolist()
        for c in clusters
    }

    # 组β = 组内等权平均收益
    cluster_beta = {}
    for c, members in cluster_members.items():
        cluster_beta[c] = returns[members].mean(axis=1)

    cluster_beta_df = pd.DataFrame(cluster_beta, index=returns.index)
    cluster_beta_df.columns = [f"cluster_{c}" for c in cluster_beta_df.columns]

    # α = 个体 - 所属簇β
    alpha_resid = {}
    ins2clu = dict(zip(labels_df["instrument"], labels_df["cluster"]))
    for ins in returns.columns:
        c = ins2clu[ins]
        beta_series = cluster_beta_df[f"cluster_{c}"]
        alpha_resid[ins] = returns[ins] - beta_series

    alpha_resid_df = pd.DataFrame(alpha_resid, index=returns.index)

    return cluster_beta_df, alpha_resid_df


def main():
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"CSV 没找到：{FILE_PATH}")

    print("1) 读取并处理数据 ...")
    returns = load_data(FILE_PATH, date_col=DATE_COL, input_is_prices=INPUT_IS_PRICES)
    print(f"原始收益维度（T×N）: {returns.shape}")

    # ===== 仅保留“价格系”列（过滤 volume / open_interest / turnover / 独立 token 'oi'）=====
    returns = filter_price_columns(returns)
    print(f"仅价格列后的维度（T×N）: {returns.shape}")
    # =====================================================================================

    print("2) 计算相关并做层次聚类 ...")
    labels_df, Z, dist_condensed = hierarchical_clustering(
        returns,
        linkage_method=LINKAGE_METHOD,
        corr_method=CORR_METHOD,
        n_clusters=N_CLUSTERS
    )

    # 输出簇标签
    clusters_csv = f"{OUTPUT_PREFIX}clusters.csv"
    labels_df.to_csv(clusters_csv, index=False)
    print(f"已保存分组：{clusters_csv}")

    # 画树状图
    dendro_png = f"{OUTPUT_PREFIX}dendrogram.png"
    plot_dendrogram(Z, labels=returns.columns.tolist(), save_path=dendro_png)
    print(f"已保存树状图：{dendro_png}")

    print("3) 计算组β与个体α ...")
    cluster_beta_df, alpha_resid_df = compute_cluster_beta_and_alpha(returns, labels_df)

    beta_csv = f"{OUTPUT_PREFIX}cluster_beta.csv"
    alpha_csv = f"{OUTPUT_PREFIX}alpha_residuals.csv"
    cluster_beta_df.to_csv(beta_csv)
    alpha_resid_df.to_csv(alpha_csv)

    print(f"已保存组β：{beta_csv}")
    print(f"已保存α残差：{alpha_csv}")

    # 简单汇总：每簇大小
    size_summary = labels_df.groupby("cluster").size().rename("count")
    print("\n簇大小分布：")
    print(size_summary.to_string())

    print("\n完成！")

if __name__ == "__main__":
    main()
