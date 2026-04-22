import logging
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

# ===================== 1. 彻底解决中文乱码（全平台适配）=====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'PingFang SC', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (10, 6)

# ===================== 2. 特征英文名→中文映射（核心！）=====================
FEATURE_CN_MAPPING = {
    'Total_Trans_Amt': '交易总金额',
    'Total_Amt_Chng_Q4_Q1': 'Q4-Q1交易金额变化率',
    'Total_Trans_Ct': '交易总次数',
    'Total_Ct_Chng_Q4_Q1': 'Q4-Q1交易次数变化率',
    'Total_Revolving_Bal': '循环余额总额',
    'Customer_Age': '客户年龄',
    'Total_Relationship_Count': '产品总数量',
    'Credit_Limit': '信用额度',
    'Avg_Open_To_Buy': '平均可透支额度',
    'Contacts_Count_12_mon': '12个月联系次数',
    'Months_on_book': '开户月数',
    'Months_Inactive_12_mon': '12个月不活跃月数',
    'Avg_Utilization_Ratio': '平均使用率',
    'Dependent_count': '家属数量',
    'Gender_Churn': '性别-流失交互项'
}
# 修改函数定义，增加 y_test 参数
def train_models(X_train, X_test, y_train, y_test, config):
    """训练模型 + 生成全中文结果（混淆矩阵+ROC+特征重要性+评估报告）"""
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 模型配置（中文名称）
    model_configs = {
        'lr': {'model': LogisticRegression(random_state=42, max_iter=1000), 'scaled': True, 'cn_name': '逻辑回归'},
        'rf': {'model': RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1), 'scaled': False, 'cn_name': '随机森林'},
        'lgb': {'model': LGBMClassifier(random_state=42, n_jobs=-1), 'scaled': False, 'cn_name': 'LightGBM'}
    }
    
    # 路径配置
    model_save_path = config['models']['save_path']
    result_img_path = config.get('results', {}).get('save_path', 'results')
    report_txt_path = os.path.join(result_img_path, '模型评估报告.txt')
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(result_img_path, exist_ok=True)
    
    preds = {}
    # 确保特征名是列表（处理DataFrame/数组）
    feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'特征{i}' for i in range(X_train.shape[1])]
    # 特征名替换为中文（找不到映射则保留原名）
    feature_names_cn = [FEATURE_CN_MAPPING.get(f, f) for f in feature_names]

    # ===================== 训练模型 + 生成中文结果 =====================
    with open(report_txt_path, 'w', encoding='utf-8') as f:  # 用utf-8保存报告（解决乱码）
        for model_name, cfg in model_configs.items():
            model = cfg['model']
            cn_name = cfg['cn_name']
            train_data = X_train_scaled if cfg['scaled'] else X_train
            test_data = X_test_scaled if cfg['scaled'] else X_test

            # 训练+预测
            model.fit(train_data, y_train)
            y_pred = model.predict(test_data)
            y_train_pred = model.predict(train_data)
            preds[f'y_train_preds_{model_name}'] = y_train_pred
            preds[f'y_test_preds_{model_name}'] = y_pred

            # 保存模型
            joblib.dump(model, os.path.join(model_save_path, f'{model_name}_model.pkl'))
            logging.info(f"SUCCESS: {cn_name} 模型训练完成")

            # --------------------- 3. 混淆矩阵（全中文） ---------------------
            # 此处 y_test 已作为参数传入，不再报错
            cm = confusion_matrix(y_test, y_pred)
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['未流失', '流失'],
                        yticklabels=['未流失', '流失'])
            plt.title(f'{cn_name} - 混淆矩阵', fontsize=14)
            plt.xlabel('预测标签', fontsize=12)
            plt.ylabel('真实标签', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(result_img_path, f'{model_name}_混淆矩阵.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # --------------------- 4. 特征重要性（中文特征名） ---------------------
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])  # 逻辑回归取系数绝对值
            else:
                importances = None

            if importances is not None:
                # 特征重要性DataFrame（中文特征名）
                feat_imp = pd.DataFrame({
                    '特征名称': feature_names_cn,
                    '重要性得分': importances
                }).sort_values(by='重要性得分', ascending=False).head(15)

                plt.figure(figsize=(12, 8))
                sns.barplot(x='重要性得分', y='特征名称', data=feat_imp)
                plt.title(f'{cn_name} - 特征重要性 TOP15', fontsize=14)
                plt.xlabel('重要性得分', fontsize=12)
                plt.ylabel('特征名称', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(result_img_path, f'{model_name}_特征重要性.png'), dpi=300, bbox_inches='tight')
                plt.close()

            # --------------------- 5. 评估报告（中文+无乱码） ---------------------
            f.write(f"\n{'='*50}\n{cn_name} 模型评估报告\n{'='*50}\n")
            # 训练集评估
            train_report = classification_report(y_train, y_train_pred, output_dict=True)
            f.write(f"\n【训练集表现】\n")
            f.write(f"  准确率: {train_report['accuracy']:.4f}\n")
            f.write(f"  召回率（流失类）: {train_report['1']['recall']:.4f}\n")
            f.write(f"  精确率（流失类）: {train_report['1']['precision']:.4f}\n")
            f.write(f"  F1值（流失类）: {train_report['1']['f1-score']:.4f}\n")
            
            # 测试集评估
            test_report = classification_report(y_test, y_pred, output_dict=True)
            f.write(f"\n【测试集表现】\n")
            f.write(f"  准确率: {test_report['accuracy']:.4f}\n")
            f.write(f"  召回率（流失类）: {test_report['1']['recall']:.4f}\n")
            f.write(f"  精确率（流失类）: {test_report['1']['precision']:.4f}\n")
            f.write(f"  F1值（流失类）: {test_report['1']['f1-score']:.4f}\n")
            f.write(f"  AUC值: {roc_auc_score(y_test, model.predict_proba(test_data)[:,1]):.4f}\n")

        # --------------------- 6. ROC曲线对比（全中文） ---------------------
        plt.figure(figsize=(10, 8))
        for model_name, cfg in model_configs.items():
            model = cfg['model']
            cn_name = cfg['cn_name']
            test_data = X_test_scaled if cfg['scaled'] else X_test
            y_proba = model.predict_proba(test_data)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{cn_name} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        plt.title('模型 ROC 曲线对比', fontsize=14)
        plt.xlabel('假正例率 (FPR)', fontsize=12)
        plt.ylabel('真正例率 (TPR)', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(result_img_path, 'ROC曲线对比.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 保存标准化器
    joblib.dump(scaler, os.path.join(model_save_path, 'scaler.pkl'))
    logging.info(f"所有结果已保存至：{result_img_path}")
    logging.info(f"评估报告已保存至：{report_txt_path}")

    return preds