import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, \
    explained_variance_score
from bayes_opt import BayesianOptimization
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import os
import warnings
import sys

warnings.filterwarnings('ignore')
import time
from datetime import datetime


# ==============================================
# 安全的KNNImputer包装器（防止数据泄露）
# ==============================================
class SafeKNNImputerWrapper(BaseEstimator, TransformerMixin):
    """
    安全的KNNImputer包装器：确保交叉验证中不泄露
    双重保护：1) 仅在训练数据上拟合 2) 添加随机噪声防止过拟合
    """

    def __init__(self, n_neighbors=5, noise_level=0.01, random_state=42):
        self.n_neighbors = n_neighbors
        self.noise_level = noise_level
        self.random_state = random_state
        self.imputer_ = None

    def fit(self, X, y=None):
        # 创建新的KNNImputer实例
        self.imputer_ = KNNImputer(
            n_neighbors=min(self.n_neighbors, X.shape[0] - 1),
            weights='uniform',  # 使用uniform，更稳定
            metric='nan_euclidean'
        )

        # 仅使用训练数据拟合
        self.imputer_.fit(X)

        # 记录统计信息
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]

        return self

    def transform(self, X):
        if self.imputer_ is None:
            raise ValueError("必须在transform之前调用fit")

        # 使用拟合好的imputer转换
        X_imputed = self.imputer_.transform(X)

        # 添加微小随机噪声防止过拟合（可选）
        if self.noise_level > 0:
            np.random.seed(self.random_state)
            # 计算每列的标准差作为噪声尺度
            col_stds = np.nanstd(X_imputed, axis=0)
            col_stds = np.where(col_stds > 0, col_stds, 1.0)  # 防止除零
            noise = np.random.normal(
                0,
                self.noise_level * col_stds,
                X_imputed.shape
            )
            # 只在缺失值位置添加噪声，减少对原始数据影响
            missing_mask = np.isnan(X)
            X_imputed[missing_mask] += noise[missing_mask]

        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


# ==============================================
# 分层目标编码器（交叉验证安全）
# ==============================================
class StratifiedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    分层目标编码器：确保只在训练数据上学习编码
    """

    def __init__(self, smoothing=5, random_state=42):
        self.smoothing = smoothing
        self.random_state = random_state
        self.encodings_ = {}  # 存储每列的编码映射
        self.global_mean_ = 0
        self.fitted_ = False

    def fit(self, X, y):
        # 重置状态
        self.encodings_ = {}

        # 仅使用传入的训练数据计算
        self.global_mean_ = np.mean(y) if len(y) > 0 else 0

        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]

            # 计算每个类别的统计量（仅在训练数据上）
            unique_cats = np.unique(col_data)
            encoding_dict = {}

            for cat in unique_cats:
                mask = col_data == cat
                if np.sum(mask) > 0:
                    cat_mean = np.mean(y[mask])
                    # 应用平滑：样本量越小，越接近全局均值
                    n_cat = np.sum(mask)
                    smoothing_factor = n_cat / (n_cat + self.smoothing)
                    encoding_dict[cat] = (
                            cat_mean * smoothing_factor +
                            self.global_mean_ * (1 - smoothing_factor)
                    )

            self.encodings_[col_idx] = encoding_dict

        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise ValueError("必须在transform之前调用fit")

        X_encoded = X.copy().astype(float)

        for col_idx in range(X.shape[1]):
            if col_idx in self.encodings_:
                encoding_dict = self.encodings_[col_idx]
                col_data = X[:, col_idx]

                for i in range(len(col_data)):
                    cat = col_data[i]
                    if cat in encoding_dict:
                        X_encoded[i, col_idx] = encoding_dict[cat]
                    else:
                        # 未知类别使用全局均值
                        X_encoded[i, col_idx] = self.global_mean_

        return X_encoded


# ==============================================
# 数据泄露检查函数
# ==============================================
def check_data_leakage(pipeline_factory, X_train, y_train, n_folds=3, n_estimators=30):
    """
    检查是否存在数据泄露
    原理：比较普通CV和手动CV的结果差异
    """
    print("\n" + "=" * 70)
    print("数据泄露检查")
    print("=" * 70)

    # 方法1：使用sklearn的cross_val_score
    pipeline = pipeline_factory()
    pipeline.set_params(regressor__n_estimators=n_estimators)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=KFold(n_splits=n_folds, shuffle=True, random_state=42),
        scoring='r2',
        n_jobs=1
    )

    print(f"普通交叉验证R²: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

    # 方法2：手动实现更严格的CV
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    manual_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        # 创建全新的pipeline实例
        fold_pipeline = pipeline_factory()
        fold_pipeline.set_params(regressor__n_estimators=n_estimators)

        # 训练并评估
        fold_pipeline.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx]
        )
        y_val_pred = fold_pipeline.predict(X_train.iloc[val_idx])
        score = r2_score(y_train.iloc[val_idx], y_val_pred)
        manual_scores.append(score)

        print(f"  第{fold}折: R² = {score:.4f}")

    manual_mean = np.mean(manual_scores)
    manual_std = np.std(manual_scores)
    print(f"严格交叉验证R²: {manual_mean:.4f} (±{manual_std:.4f})")

    # 差异分析
    cv_mean = np.mean(cv_scores)
    diff = cv_mean - manual_mean

    print(f"\n泄露检查结果:")
    print(f"  CV均值: {cv_mean:.4f}")
    print(f"  手动CV均值: {manual_mean:.4f}")
    print(f"  差异: {diff:.4f}")

    if diff > 0.05:
        print(f"  ⚠️ 警告：可能存在数据泄露！差异 = {diff:.4f}")
        return False
    elif diff > 0.02:
        print(f"  ⚠️ 注意：有轻微泄露风险，差异 = {diff:.4f}")
        return True
    else:
        print(f"  ✅ 良好：无明显数据泄露，差异 = {diff:.4f}")
        return True


# ==============================================
# 频率编码器（不依赖目标变量）
# ==============================================
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    频率编码器：将类别值映射为出现频率
    不依赖目标变量，避免数据泄露
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.frequencies_ = {}  # 存储每列的类别-频率映射

    def fit(self, X, y=None):
        for col_idx in range(X.shape[1]):
            col_data = X[:, col_idx]
            # 计算每个类别的频率
            value_counts = pd.Series(col_data).value_counts()
            total = len(col_data)
            self.frequencies_[col_idx] = value_counts / total

        return self

    def transform(self, X):
        X_encoded = X.copy().astype(float)

        for col_idx in range(X.shape[1]):
            if col_idx in self.frequencies_:
                freq_series = self.frequencies_[col_idx]
                col_data = X[:, col_idx]

                # 映射频率值
                for i in range(len(col_data)):
                    cat = col_data[i]
                    if cat in freq_series.index:
                        X_encoded[i, col_idx] = freq_series[cat]
                    else:
                        # 未知类别使用平均频率
                        X_encoded[i, col_idx] = 0.0  # 或使用其他默认值

        return X_encoded











# ==============================================
# 数据加载
# ==============================================
path = r"E:\胡琰琰\毕业论文内容\G\运行数据\数据 - 微删.xlsx"
dataset = pd.read_excel(path)

# ==============================================
# 手动指定类别列
# ==============================================
CATEGORICAL_COLS = ['MOx', 'Sys.', 'S.G.']

# 选择要预测的目标变量
target_var = 'Cp_A'  # 可以改为 'Cp_B', 'Cp_C', 'Cp_D'

print("\n" + "=" * 70)
print(f"开始训练 {target_var} 的单独模型")
print("=" * 70)

# ==============================================
# 准备数据 - 修正目标变量删除逻辑
# ==============================================
# 动态删除其他目标变量，保留当前目标变量
all_targets = ['Cp_A', 'Cp_B', 'Cp_C', 'Cp_D', 'G0']
X = dataset.drop(all_targets, axis=1) # 删除所有目标变量
y = dataset[target_var] # 当前目标变量作为y

# 检查数据类型
print("\n检查特征数据类型:")
for col in CATEGORICAL_COLS:
    if col in X.columns:
        dtype = X[col].dtype
        unique_vals = X[col].nunique()
        print(f"  {col}: 类型={dtype}, 唯一值数量={unique_vals}")

NUMERICAL_COLS = [col for col in X.columns if col not in CATEGORICAL_COLS]

print(f"\n目标变量: {target_var}")
print(f"特征数量: {len(X.columns)} (数值: {len(NUMERICAL_COLS)}, 类别: {len(CATEGORICAL_COLS)})")
print(f"样本数量: {len(X)}")

# ==============================================
# 数据诊断
# ==============================================
print("\n数据诊断信息:")
print(f"目标变量 {target_var}:")
print(f"  最小值: {y.min():.4f}, 最大值: {y.max():.4f}")
print(f"  平均值: {y.mean():.4f}, 标准差: {y.std():.4f}")
print(f"  偏度: {y.skew():.4f}, 峰度: {y.kurtosis():.4f}")

# 检查缺失值
missing_values = X.isnull().sum()
print(f"\n缺失值分析:")
print(f"  总缺失值数: {missing_values.sum()}")
print(f"  缺失值比例: {(missing_values.sum() / (X.shape[0] * X.shape[1]) * 100):.2f}%")


# ==============================================
# 数据预处理管道（最终修正版）
# ==============================================
def create_preprocessing_pipeline(use_target_encoding=False):
    """
    创建预处理管道 - 最终修正版
    可选是否使用目标编码
    """

    # 1. 数值特征预处理
    numerical_transformer = Pipeline(steps=[
        ('imputer', SafeKNNImputerWrapper(
            n_neighbors=5,
            noise_level=0.01,
            random_state=42
        )),
        ('scaler', StandardScaler())
    ])

    # 2. 类别特征预处理
    if use_target_encoding:
        # 使用频率编码而不是目标编码
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', FrequencyEncoder(random_state=42))
        ])
    else:
        # 不使用任何编码，保持原始值（但需要转换为数值）
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            # 保持原样，后续模型处理
        ])

    # 3. 整合预处理
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_COLS),
            ('cat', categorical_transformer, CATEGORICAL_COLS)
        ],
        remainder='drop'
    )

    return preprocessor

# ==============================================
# 分割数据
# ==============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)

print(f"\n数据分割结果:")
print(f"  训练集: {X_train.shape} ({(X_train.shape[0] / len(X) * 100):.1f}%)")
print(f"  测试集: {X_test.shape} ({(X_test.shape[0] / len(X) * 100):.1f}%)")


# ==============================================
# 创建完整管道工厂函数
# ==============================================
def create_full_pipeline(gbr_params=None):
    """
    创建完整管道 - GBR版本
    """
    if gbr_params is None:
        gbr_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 0.8,
            'max_features': 0.8,
            'random_state': 42
        }

    pipeline = Pipeline(steps=[
        ('preprocessor', create_preprocessing_pipeline()),
        ('regressor', GradientBoostingRegressor(**gbr_params))
    ])

    return pipeline



# ==============================================
# 模型构建（使用负的相对RMSE）
# ==============================================
def gbr_cv_score(n_estimators, learning_rate, max_depth,
                 min_samples_split, min_samples_leaf,
                 subsample, max_features):
    try:
        # 参数转换
        n_estimators = int(n_estimators)
        max_depth = int(max_depth)
        learning_rate = float(learning_rate)
        min_samples_split = int(min_samples_split)
        min_samples_leaf = int(min_samples_leaf)
        subsample = float(subsample)
        max_features = float(max_features)

        # 使用手动交叉验证确保独立性
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        relative_rmse_scores = []

        for train_idx, val_idx in kfold.split(X_train):
            # 分割数据
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]

            # 创建全新的pipeline实例 - GBR版本
            fold_pipeline = create_full_pipeline({
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'subsample': subsample,
                'max_features': max_features,
                'random_state': 42
            })

            # 训练并评估
            fold_pipeline.fit(X_train_fold, y_train_fold)
            y_val_pred = fold_pipeline.predict(X_val_fold)

            # 计算相对RMSE
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
            y_range_fold = y_val_fold.max() - y_val_fold.min()

            if y_range_fold > 0:
                relative_rmse = rmse / y_range_fold
            else:
                relative_rmse = 1.0

            relative_rmse_scores.append(relative_rmse)

        # 返回负的平均相对RMSE
        avg_relative_rmse = np.mean(relative_rmse_scores)
        return -avg_relative_rmse

    except Exception as e:
        print(f"交叉验证出错: {e}")
        return -1.0







# ==============================================
# 数据泄露检查
# ==============================================
# 创建管道工厂函数
def pipeline_factory():
    return create_full_pipeline()
# 运行泄露检查
leakage_safe = check_data_leakage(pipeline_factory, X_train, y_train, n_folds=3, n_estimators=30)

if not leakage_safe:
    print("\n⚠️ 警告：检测到可能的数据泄露！建议检查编码器和填充器配置。")
    user_choice = input("是否继续训练？（y/n）: ")
    if user_choice.lower() != 'y':
        print("训练已取消。")
        sys.exit(0)
else:
    print("\n✅ 数据泄露检查通过，继续训练最终模型...")








# ==============================================
# 贝叶斯优化
# ==============================================
param_bounds = {
    'n_estimators': (240, 255),
    'learning_rate': (0.12, 0.14),
    'max_depth': (7, 9),
    'min_samples_split': (3, 5),
    'min_samples_leaf': (1, 3),
    'subsample': (0.66, 0.70),
    'max_features': (0.58, 0.62)
}


print(f"\n对 {target_var} 进行贝叶斯优化（使用负的相对RMSE）...")
print(f"优化参数范围: {param_bounds}")
print("目标函数：负的相对RMSE（越小越好，所以负得越多越好）")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

start_time = time.time()

try:
    # 创建贝叶斯优化器
    optimizer = BayesianOptimization(
        f=gbr_cv_score,  # 使用新的目标函数
        pbounds=param_bounds,
        random_state=42,
        verbose=2
    )

    # 执行优化
    optimizer.maximize(
        init_points=8,
        n_iter=12
    )

    # 获取最佳参数
    if optimizer.max:
        best_params = optimizer.max['params']
        best_score = optimizer.max['target']  # 这是负的相对RMSE

        # 转换参数类型
        best_params_converted = {
            'n_estimators': int(best_params['n_estimators']),
            'learning_rate': float(best_params['learning_rate']),
            'max_depth': int(best_params['max_depth']),
            'min_samples_split': int(best_params['min_samples_split']),
            'min_samples_leaf': int(best_params['min_samples_leaf']),
            'subsample': float(best_params['subsample']),
            'max_features': float(best_params['max_features']),
            'random_state': 42
        }

        print(f"\n✅ 贝叶斯优化完成!")
        print(f"最佳参数: {best_params_converted}")
        print(f"最佳得分（负的相对RMSE）: {best_score:.6f}")
        print(f"对应的相对RMSE: {-best_score:.6f}")  # 转成正数
        print(f"相对RMSE百分比: {-best_score * 100:.2f}%")

        # 评估标准
        relative_rmse_value = -best_score
        if relative_rmse_value < 0.05:
            print(f"模型表现评级: ✅ 优秀 (相对RMSE < 5%)")
        elif relative_rmse_value < 0.1:
            print(f"模型表现评级: ✅ 良好 (相对RMSE 5-10%)")
        elif relative_rmse_value < 0.2:
            print(f"模型表现评级: ⚠️ 一般 (相对RMSE 10-20%)")
        else:
            print(f"模型表现评级: ❌ 待改进 (相对RMSE > 20%)")

    else:
        raise ValueError("贝叶斯优化未返回结果")

except Exception as e:
    print(f"贝叶斯优化出错: {e}")
    print("使用保守的默认参数...")
    # 使用更保守的默认参数
    best_params_converted = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 0.8,
        'max_features': 0.8,
        'random_state': 42
    }

end_time = time.time()
print(f"优化耗时: {end_time - start_time:.2f}秒")






# ==============================================
# 训练最终完整模型
# ==============================================
print(f"\n训练 {target_var} 的最终模型...")

# 创建完整管道
final_pipeline = create_full_pipeline(best_params_converted)

# 训练模型
print("正在训练最终模型...")
train_start = time.time()
final_pipeline.fit(X_train, y_train)
train_end = time.time()
print(f"模型训练完成，耗时: {train_end - train_start:.2f}秒")

# 预测
y_train_pred = final_pipeline.predict(X_train)
y_test_pred = final_pipeline.predict(X_test)


# ==============================================
# 模型评估函数
# ==============================================
def calculate_metrics(y_true, y_pred, dataset_name):
    """计算并返回模型评估指标"""
    # 基础指标
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)  # 中位数绝对误差

    # 解释方差
    evs = explained_variance_score(y_true, y_pred)

    # 平均绝对百分比误差（使用更稳健的方法）
    epsilon = np.finfo(float).eps * 10
    y_true_nonzero = np.where(np.abs(y_true) > epsilon, y_true, np.nan)
    valid_mask = ~np.isnan(y_true_nonzero)

    if np.sum(valid_mask) > 0:
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        mape = np.mean(np.abs((y_true_valid - y_pred_valid) / y_true_valid)) * 100
    else:
        mape = np.nan

    # 计算预测误差统计
    errors = y_true - y_pred
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # 计算残差分布统计
    residual_skew = pd.Series(errors).skew()
    residual_kurt = pd.Series(errors).kurtosis()

    return {
        '数据集': dataset_name,
        'R²': r2,
        '解释方差': evs,
        'RMSE': rmse,
        'MAE': mae,
        'MedAE': medae,
        'MAPE%': mape,
        '平均误差': mean_error,
        '误差标准差': std_error,
        '最大绝对误差': np.max(np.abs(errors)),
        '95%误差分位数': np.percentile(np.abs(errors), 95),
        '误差偏度': residual_skew,
        '误差峰度': residual_kurt
    }


# ==============================================
# 模型验证函数
# ==============================================
def save_and_validate_model(pipeline, save_path, X_sample, y_sample):
    """保存模型并进行完整性验证"""
    try:
        # 保存模型
        joblib.dump(pipeline, save_path, compress=1)  # 降低压缩级别加速
        print(f"  模型已保存到: {save_path}")

        # 加载验证
        loaded_model = joblib.load(save_path)

        # 预测验证
        predictions = loaded_model.predict(X_sample)
        original_predictions = pipeline.predict(X_sample)

        # 检查一致性
        mae_diff = mean_absolute_error(original_predictions, predictions)
        if mae_diff < 1e-6:
            print(f"  ✅ 模型保存验证通过！预测一致性误差: {mae_diff:.6f}")
            return True
        else:
            print(f"  ⚠️ 模型保存不一致！预测误差: {mae_diff:.6f}")
            return False

    except Exception as e:
        print(f"  ❌ 模型保存/验证失败: {e}")
        return False


# ==============================================
# 模型评估
# ==============================================
train_metrics = calculate_metrics(y_train, y_train_pred, '训练集')
test_metrics = calculate_metrics(y_test, y_test_pred, '测试集')

# 输出结果
print(f"\n{target_var} 详细评估结果:")
print("-" * 85)
print(f"{'指标':<20} {'训练集':<15} {'测试集':<15} {'差异':<15} {'状态':<20}")
print("-" * 85)

metrics_to_show = ['R²', 'RMSE', 'MAE', 'MAPE%']
for metric in metrics_to_show:
    train_val = train_metrics[metric]
    test_val = test_metrics[metric]

    if metric == 'R²':
        diff = train_val - test_val
        if diff > 0.1:
            status = "⚠️过拟合风险"
        elif diff > 0.05:
            status = "轻度过拟合"
        else:
            status = "✅ 良好"
    elif metric in ['RMSE', 'MAE', 'MAPE%']:
        if train_val != 0:  # 避免除零错误
            diff_ratio = abs(test_val - train_val) / abs(train_val)
            if diff_ratio > 0.5:
                status = "⚠️泛化问题"
            elif diff_ratio > 0.2:
                status = "轻度问题"
            else:
                status = "✅ 良好"
        else:
            status = "无法计算"
        diff = test_val - train_val
    else:
        diff = 0
        status = ""

    print(f"{metric:<20} {train_val:<15.6f} {test_val:<15.6f} {diff:<15.6f} {status:<20}")

print(f"\n模型稳定性分析:")
print(f"  训练集误差标准差: {train_metrics['误差标准差']:.6f}")
print(f"  测试集误差标准差: {test_metrics['误差标准差']:.6f}")
print(f"  95%预测误差小于: {test_metrics['95%误差分位数']:.6f}")

# ==============================================
# 导出预测结果
# ==============================================
print(f"\n导出预测结果...")

# 创建输出目录
output_path = f"E:\\胡琰琰\\毕业论文内容\\G\\运行数据\\拟合\\A\\{target_var}_GBR_安全版.xlsx"
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)

# 创建结果DataFrame
train_results = pd.DataFrame({
    '样本索引': y_train.index,
    '数据集': '训练集',
    f'{target_var}_真实值': y_train.values,
    f'{target_var}_预测值': y_train_pred,
    '绝对误差': np.abs(y_train.values - y_train_pred),
    '相对误差%': np.abs((y_train.values - y_train_pred) / (np.abs(y_train.values) + 1e-10)) * 100,
})

test_results = pd.DataFrame({
    '样本索引': y_test.index,
    '数据集': '测试集',
    f'{target_var}_真实值': y_test.values,
    f'{target_var}_预测值': y_test_pred,
    '绝对误差': np.abs(y_test.values - y_test_pred),
    '相对误差%': np.abs((y_test.values - y_test_pred) / (np.abs(y_test.values) + 1e-10)) * 100,
})

# 合并结果
all_results = pd.concat([train_results, test_results], ignore_index=True)

# 添加模型参数信息
params_df = pd.DataFrame([{
    '参数': key,
    '值': value
} for key, value in best_params_converted.items()])

# 添加评估指标
metrics_df = pd.DataFrame([train_metrics, test_metrics])

# 导出到Excel
try:
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        all_results.to_excel(writer, sheet_name='所有预测结果', index=False)
        train_results.to_excel(writer, sheet_name='训练集结果', index=False)
        test_results.to_excel(writer, sheet_name='测试集结果', index=False)
        params_df.to_excel(writer, sheet_name='模型参数', index=False)
        metrics_df.to_excel(writer, sheet_name='评估指标', index=False)

        # 添加数据统计
        stats_df = pd.DataFrame({
            '数据统计': [
                f'训练样本数: {len(X_train)}',
                f'测试样本数: {len(X_test)}',
                f'特征总数: {len(X.columns)}',
                f'数值特征: {len(NUMERICAL_COLS)}',
                f'类别特征: {len(CATEGORICAL_COLS)}',
                f'泄露检查结果: {"通过" if leakage_safe else "警告"}',
                f'优化耗时: {end_time - start_time:.2f}秒',
                f'模型训练耗时: {train_end - train_start:.2f}秒',
                f'导出时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            ]
        })
        stats_df.to_excel(writer, sheet_name='数据统计', index=False)

    print(f"\n✅ 预测结果已成功导出到: {output_path}")

except Exception as e:
    print(f"❌ 数据导出失败: {e}")
    # 备选方案：导出为CSV
    try:
        csv_path = output_path.replace('.xlsx', '.csv')
        all_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 备选导出成功！CSV文件: {csv_path}")
    except Exception as e2:
        print(f"❌ 备选导出也失败: {e2}")

# ==============================================
# 保存训练好的模型
# ==============================================
print(f"\n" + "=" * 70)
print("保存训练好的模型")
print("=" * 70)

# 1. 创建模型保存目录
model_save_dir = r"E:\\胡琰琰\\毕业论文内容\\G\\运行数据\\拟合\\A\\保存的模型"
os.makedirs(model_save_dir, exist_ok=True)

# 2. 保存完整pipeline（包括预处理和模型）
model_filename = f"{target_var}_GBR_模型_安全版.pkl"
model_save_path = os.path.join(model_save_dir, model_filename)

# 3. 使用少量样本验证模型保存
sample_size = min(10, len(X_test))
X_sample = X_test.iloc[:sample_size]
y_sample = y_test.iloc[:sample_size]

# 保存并验证模型
save_success = save_and_validate_model(final_pipeline, model_save_path, X_sample, y_sample)
