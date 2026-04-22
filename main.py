import logging
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

def train_models(X_train, X_test, y_train, config):
    """训练多个模型并返回预测结果+保存模型"""
    # 标准化数值特征（提升模型效果）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 模型配置（统一管理超参数）
    model_configs = {
        'lr': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'scaled': True  # LR需要标准化
        },
        'rf': {
            'model': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),  # 多线程加速
            'scaled': False
        },
        'lgb': {
            'model': LGBMClassifier(random_state=42, n_estimators=100, n_jobs=-1),
            'scaled': False
        }
    }
    
    # 训练+预测
    preds = {}
    model_save_path = config['models']['save_path']
    os.makedirs(model_save_path, exist_ok=True)
    
    for model_name, cfg in model_configs.items():
        model = cfg['model']
        train_data = X_train_scaled if cfg['scaled'] else X_train
        test_data = X_test_scaled if cfg['scaled'] else X_test
        
        # 训练
        model.fit(train_data, y_train)
        # 预测
        preds[f'y_train_preds_{model_name}'] = model.predict(train_data)
        preds[f'y_test_preds_{model_name}'] = model.predict(test_data)
        # 保存模型
        joblib.dump(model, os.path.join(model_save_path, f'{model_name}_model.pkl'))
        logging.info(f"SUCCESS: {model_name.upper()} model trained and saved.")
    
    # 保存scaler
    joblib.dump(scaler, os.path.join(model_save_path, 'scaler.pkl'))
    return preds