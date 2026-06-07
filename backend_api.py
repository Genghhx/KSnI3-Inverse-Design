from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from autogluon.tabular import TabularPredictor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from category_encoders import LeaveOneOutEncoder
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# ============ 全局配置 ============
ALLOWED_VALUES = {
    'ETL thickness(nm)': [10.0, 20.0],
    'Perovskite thickness(nm)': [400, 500, 820, 880, 980],
    'HTL thickness(nm)': [10.0, 30.0, 200.0, 400.0],
    'Back Contact WF (eV)': [5.00, 5.10, 5.22]
}

# ============ 全局变量（缓存） ============
_cache = {
    'data': None,
    'scaler': None,
    'train_selected': None,
    'encoder': None,
    'encoding_range_df': None,
    'predictor': None,
    'feature_names': None,
    'ETL_list': None
}

# ============ 初始化函数 ============
def initialize():

    global _cache

    print("[INIT] Starting initialization...")

    # ===== Cell 1: 数据预处理（离散变量编码+归一化） =====
    print("[INIT] Loading data and preprocessing...")

    # 读取数据
    data = pd.read_csv(r'C:\Users\geng\Desktop\论文\结果+(2)\表\ML_Data.csv')
    data = data.drop(columns=['VOC(V)', 'JSC(mA/cm2)', 'FF(%)', 'Device Architecture', 'All inorganic PSC',
                              'Inorganic ETL', 'Inorganic HTL', 'Rseries (ohm cm2)', 'Rshunt (ohm cm2)'])

    numeric_cols = [
        'ETL VB DOS(cm-3)',
        'ETL donor density(cm-3)',
        'ETL defect density(cm-3)',
        'Perovskite donor density(cm-3)',
        'Perovskite defect density(cm-3)',
        'HTL VB DOS(cm-3)',
        'ETL/Absorber defect density(cm-3)'
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    _cache['data'] = data

    # 划分训练集和测试集
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

    # 分类变量处理
    categorical_cols = ['ETL Materials', 'Perovskite', 'HTL Materials', 'Back Contact']
    encoder = LeaveOneOutEncoder(cols=categorical_cols)

    X_train = train_data.iloc[:, :-1]
    Y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    Y_test = test_data.iloc[:, -1]

    # 在训练集上训练编码器
    encoder.fit(X_train, Y_train)

    # 编码训练集和测试集
    encoded_X_train = encoder.transform(X_train)
    encoded_X_test = encoder.transform(X_test)

    # 创建一个DataFrame来保存编码范围
    encoding_range_df = pd.DataFrame(columns=['Column', 'Category', 'Min Encoded Value', 'Max Encoded Value'])

    # 对每个分类列计算每个类别的编码范围
    for col in categorical_cols:
        unique_categories = X_train[col].unique()
        for category in unique_categories:
            temp_df = X_train[X_train[col] == category]
            temp_encoded = encoder.transform(temp_df)
            min_val = temp_encoded[col].min()
            max_val = temp_encoded[col].max()

            new_row = pd.DataFrame({
                'Column': [col],
                'Category': [category],
                'Min Encoded Value': [min_val],
                'Max Encoded Value': [max_val]
            })
            encoding_range_df = pd.concat([encoding_range_df, new_row], ignore_index=True)

    # 合并编码后的训练集和测试集以检查常量特征列
    combined_data = pd.concat([encoded_X_train, encoded_X_test])

    # 寻找在合并数据集中都相等的特征列
    constant_columns = [col for col in combined_data.columns if combined_data[col].nunique() == 1]

    # 分别从训练集和测试集中删除这些特征列
    encoded_X_train.drop(columns=constant_columns, inplace=True)
    encoded_X_test.drop(columns=constant_columns, inplace=True)

    # 将目标变量合并到特征DataFrame中
    encoded_X_train['PCE(%)'] = Y_train
    encoded_X_test['PCE(%)'] = Y_test

    # 分离特征和目标变量
    X_train_original = encoded_X_train.drop(columns=['PCE(%)'])
    y_train_original = encoded_X_train['PCE(%)']
    X_test_original = encoded_X_test.drop(columns=['PCE(%)'])
    y_test_original = encoded_X_test['PCE(%)']

    # 只对特征进行标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_original)
    X_test_scaled = scaler.transform(X_test_original)

    # 将归一化后的特征和原始目标变量合并成 DataFrame
    train = pd.DataFrame(X_train_scaled, columns=X_train_original.columns)
    train['PCE(%)'] = y_train_original.values

    test = pd.DataFrame(X_test_scaled, columns=X_test_original.columns)
    test['PCE(%)'] = y_test_original.values

    _cache['scaler'] = scaler
    _cache['encoder'] = encoder
    _cache['encoding_range_df'] = encoding_range_df

    print(f"  Data shape: {train.shape}")
    print(f"  Columns: {train.columns.tolist()}")

    # ===== Cell 2: 获取 ETL_list =====
    ETL_list = data['ETL Materials'].unique().tolist()
    _cache['ETL_list'] = ETL_list
    print(f"  ETL Materials count: {len(ETL_list)}")

    # ===== Cell 13: 特征筛选 =====
    print("[INIT] Selecting features with Bayesian Optimization...")

    def evaluate_model(**kwargs):
        selected_indices = [int(key.split('_')[1]) for key, value in kwargs.items() if value > 0.5]
        if not selected_indices:
            return -np.inf
        X_selected = train.iloc[:, selected_indices]
        model = RandomForestRegressor()
        scores = cross_val_score(model, X_selected, train['PCE(%)'], cv=5, scoring='neg_mean_squared_error')
        return np.mean(scores)

    pbounds = {f'feature_{i}': (0, 1) for i in range(train.shape[1] - 1)}

    optimizer = BayesianOptimization(
        f=evaluate_model,
        pbounds=pbounds,
        random_state=0,
        verbose=2
    )

    optimizer.maximize(
        init_points=10,
        n_iter=20
    )

    best_features = [int(key.split('_')[1]) for key, value in optimizer.max['params'].items() if value > 0.5]
    selected_feature_names = [train.columns[i] for i in best_features]
    print(f"  Selected {len(best_features)} features:")
    print(f"  {selected_feature_names}")

    # ===== Cell 14: 提取选中特征的数据集 =====
    train_selected = train.iloc[:, best_features].copy()
    test_selected = test.iloc[:, best_features].copy()

    train_selected['PCE(%)'] = train['PCE(%)']
    test_selected['PCE(%)'] = test['PCE(%)']

    print(f"  train_selected shape: {train_selected.shape}")
    print(f"  train_selected columns: {train_selected.columns.tolist()}")

    _cache['train_selected'] = train_selected

    # ===== Cell 21: 模型建立（AutoGluon） =====
    print("[INIT] Training AutoGluon model...")

    predictor = TabularPredictor(
        label='PCE(%)',
        problem_type='regression',
        eval_metric='r2'
    ).fit(
        train_selected,
        presets='best_quality',
        time_limit=1000  # 增加到 600 秒（10分钟），因为特征选择已耗时
    )

    _cache['predictor'] = predictor
    _cache['feature_names'] = train_selected.drop(columns=['PCE(%)']).columns.tolist()

    # ===== 评估模型 =====
    print("[INIT] Evaluating model...")
    from sklearn.metrics import r2_score, mean_squared_error

    y_train = train_selected['PCE(%)']
    train_features = train_selected.drop(columns=['PCE(%)'])
    train_pred = predictor.predict(train_features)
    r2 = r2_score(y_train, train_pred)
    rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print(f"  Train R²: {r2:.4f}, RMSE: {rmse:.4f}")

    print("[INIT] Initialization completed!")


def get_material_encoded_value(material_name):
    """获取材料的编码值 - 复制 notebook 逻辑"""
    data = _cache['data']
    encoder = _cache['encoder']
    scaler = _cache['scaler']

    matching_rows = data[data['ETL Materials'] == material_name]
    if len(matching_rows) == 0:
        raise ValueError(f"Material '{material_name}' not found")

    first_match_idx = matching_rows.index[0]
    temp_df = data.iloc[[first_match_idx]]
    X_temp = temp_df.iloc[:, :-1]
    encoded_temp = encoder.transform(X_temp)
    return encoded_temp['ETL Materials'].iloc[0]


def convert_physical_to_scaled(physical_value, feature_name):
    """将物理值转换为缩放值 - 复制 notebook 逻辑"""
    scaler = _cache['scaler']
    original_feature_names = list(scaler.feature_names_in_)

    if feature_name not in original_feature_names:
        raise ValueError(f"Feature '{feature_name}' not found")

    idx = original_feature_names.index(feature_name)
    mean = scaler.mean_[idx]
    scale = scaler.scale_[idx]
    scaled_value = (physical_value - mean) / scale
    return scaled_value


def find_material_by_encoded_value(encoded_value):
    """根据编码值反向查找材料名 - 复制 notebook 逻辑"""
    encoding_range_df = _cache['encoding_range_df']
    etl_ranges = encoding_range_df[encoding_range_df['Column'] == 'ETL Materials']

    closest_material = None
    closest_distance = float('inf')

    for idx, row in etl_ranges.iterrows():
        min_val = row['Min Encoded Value']
        max_val = row['Max Encoded Value']
        mid_val = (min_val + max_val) / 2

        distance = abs(encoded_value - mid_val)
        if distance < closest_distance:
            closest_distance = distance
            closest_material = row['Category']

    return closest_material


# ============ API 端点 ============
@app.route('/api/options', methods=['GET'])
def get_options():
    """获取所有可选值"""
    return jsonify({
        'ETL_thickness': ALLOWED_VALUES['ETL thickness(nm)'],
        'perovskite_thickness': ALLOWED_VALUES['Perovskite thickness(nm)'],
        'HTL_thickness': ALLOWED_VALUES['HTL thickness(nm)'],
        'back_contact_wf': ALLOWED_VALUES['Back Contact WF (eV)']
    })


@app.route('/api/optimize', methods=['POST'])
def optimize():
    """执行优化计算 - 复制 notebook Cell 38 逻辑"""
    try:
        params = request.json or {}

        specified_etl_thickness = params.get('ETL_thickness')
        specified_perovskite_thickness = params.get('perovskite_thickness')
        specified_htl_thickness = params.get('HTL_thickness')
        specified_back_contact_wf = params.get('back_contact_wf')

        # 验证参数
        if specified_etl_thickness and specified_etl_thickness not in ALLOWED_VALUES['ETL thickness(nm)']:
            return jsonify({'success': False, 'error': f'Invalid ETL thickness: {specified_etl_thickness}'})
        if specified_perovskite_thickness and specified_perovskite_thickness not in ALLOWED_VALUES['Perovskite thickness(nm)']:
            return jsonify({'success': False, 'error': f'Invalid Perovskite thickness: {specified_perovskite_thickness}'})
        if specified_htl_thickness and specified_htl_thickness not in ALLOWED_VALUES['HTL thickness(nm)']:
            return jsonify({'success': False, 'error': f'Invalid HTL thickness: {specified_htl_thickness}'})
        if specified_back_contact_wf and specified_back_contact_wf not in ALLOWED_VALUES['Back Contact WF (eV)']:
            return jsonify({'success': False, 'error': f'Invalid Back Contact WF: {specified_back_contact_wf}'})

        predictor = _cache['predictor']
        scaler = _cache['scaler']
        train_selected = _cache['train_selected']
        feature_names = _cache['feature_names']

        # 锁定最佳点
        best_idx = train_selected['PCE(%)'].idxmax()
        best_row = train_selected.iloc[best_idx]
        best_features_scaled = best_row[feature_names].to_dict()

        pbounds = {}
        search_radius = 2.0
        positive_keywords = ['thickness', 'density', 'mobility', 'permittivity', 'dos', 'gap', 'temperature']
        original_feature_names = list(scaler.feature_names_in_)

        for feat, val in best_features_scaled.items():
            feat_min = train_selected[feat].min()
            feat_max = train_selected[feat].max()
            feat_std = train_selected[feat].std()

            ub = max(val + search_radius, feat_max + (feat_std * 0.5))
            lb = min(val - search_radius, feat_min - (feat_std * 0.5))

            if feat in original_feature_names:
                idx = original_feature_names.index(feat)
                mean = scaler.mean_[idx]
                scale = scaler.scale_[idx]

                is_positive_physical = any(k in feat.lower() for k in positive_keywords)

                if is_positive_physical:
                    physical_min_limit = 0.0001
                    scaled_min_limit = (physical_min_limit - mean) / scale
                    if lb < scaled_min_limit:
                        lb = scaled_min_limit

            if lb >= ub:
                lb = ub - 0.01

            pbounds[feat] = (lb, ub)

        # 锁定指定的参数
        specified_params_scaled = {}

        param_mappings = {
            'ETL thickness(nm)': specified_etl_thickness,
            'Perovskite thickness(nm)': specified_perovskite_thickness,
            'HTL thickness(nm)': specified_htl_thickness,
            'Back Contact WF (eV)': specified_back_contact_wf
        }

        for param_name, param_value in param_mappings.items():
            if param_value is not None:
                scaled_value = convert_physical_to_scaled(param_value, param_name)
                pbounds[param_name] = (scaled_value, scaled_value)
                specified_params_scaled[param_name] = scaled_value

        # 执行贝叶斯优化
        def target_function(**kwargs):
            input_df = pd.DataFrame([kwargs], columns=feature_names)
            return predictor.predict(input_df).iloc[0]

        optimizer = BayesianOptimization(
            f=target_function,
            pbounds=pbounds,
            random_state=42,
            verbose=2
        )

        optimizer.probe(params=best_features_scaled, lazy=True)
        optimizer.maximize(init_points=5, n_iter=50)

        best_target = optimizer.max['target']
        best_params_scaled = optimizer.max['params']

        # 反归一化
        def inverse_transform_safe(scaled_params_dict, scaler):
            original_feature_names = list(scaler.feature_names_in_)
            full_scaled_array = np.zeros((1, len(original_feature_names)))
            for i, feat in enumerate(original_feature_names):
                if feat in scaled_params_dict:
                    full_scaled_array[0, i] = scaled_params_dict[feat]
                else:
                    full_scaled_array[0, i] = 0.0
            full_original_array = scaler.inverse_transform(full_scaled_array)
            original_params = {}
            for feat in scaled_params_dict.keys():
                if feat in original_feature_names:
                    idx = original_feature_names.index(feat)
                    original_params[feat] = full_original_array[0, idx]
            return original_params

        real_params = inverse_transform_safe(best_params_scaled, scaler)

        # 移除 ETL Materials 从结果中
        real_params.pop('ETL Materials', None)

        # 构建响应
        response = {
            'success': True,
            'optimal_pce': round(float(best_target), 4),
            'optimal_parameters': real_params,
            'specified_parameters': {
                'ETL_thickness': specified_etl_thickness,
                'perovskite_thickness': specified_perovskite_thickness,
                'HTL_thickness': specified_htl_thickness,
                'back_contact_wf': specified_back_contact_wf
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        print(f"[ERROR] {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok'})


# ============ 主程序 ============
if __name__ == '__main__':
    print("Initializing backend...")
    initialize()

    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
