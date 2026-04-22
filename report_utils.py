import os
import matplotlib.pyplot as plt
import logging
from constants import PLOT_STYLE

def format_report_to_chinese(report_str):
    """将英文分类报告转换为中文"""
    lines = report_str.split('\n')
    result_lines = []
    for line in lines:
        line = line.replace('precision', '精确率')
        line = line.replace('recall', '召回率')
        line = line.replace('f1-score', 'F1分数')
        line = line.replace('accuracy', '准确率')
        line = line.replace('macro avg', '宏平均')
        line = line.replace('weighted avg', '加权平均')
        result_lines.append(line)
    return '\n'.join(result_lines)

def classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_test_preds_lr,
                                y_train_preds_rf, y_test_preds_rf,
                                y_train_preds_lgb=None, y_test_preds_lgb=None):
    """生成分类报告可视化图像并保存"""
    from sklearn.metrics import classification_report
    
    results_path = './images/results'
    os.makedirs(results_path, exist_ok=True)

    # 通用报告生成函数（减少冗余）
    def _generate_report(model_name, y_train, y_test, y_train_pred, y_test_pred, save_name):
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        text_content = f"""{'='*60}
{model_name}模型评估报告
{'='*60}

【训练集结果】
{'-'*40}
  准确率: {train_report['accuracy']:.4f}
  精确率 (流失类): {train_report['1']['precision']:.4f}
  召回率 (流失类): {train_report['1']['recall']:.4f}
  F1分数 (流失类): {train_report['1']['f1-score']:.4f}

【测试集结果】
{'-'*40}
  准确率: {test_report['accuracy']:.4f}
  精确率 (流失类): {test_report['1']['precision']:.4f}
  召回率 (流失类): {test_report['1']['recall']:.4f}
  F1分数 (流失类): {test_report['1']['f1-score']:.4f}

{'='*60}"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        ax.text(0.05, 0.95, text_content, fontsize=10, fontfamily='monospace',
                transform=ax.transAxes, verticalalignment='top')
        plt.savefig(os.path.join(results_path, save_name), **PLOT_STYLE)
        plt.close()

    # 生成各模型报告
    _generate_report('逻辑回归', y_train, y_test, y_train_preds_lr, y_test_preds_lr, 'logistic_regression_results.png')
    _generate_report('随机森林', y_train, y_test, y_train_preds_rf, y_test_preds_rf, 'random_forest_results.png')
    
    # LGBM（可选）
    if y_train_preds_lgb is not None and y_test_preds_lgb is not None:
        _generate_report('LightGBM', y_train, y_test, y_train_preds_lgb, y_test_preds_lgb, 'lightgbm_results.png')

    logging.info("SUCCESS: Classification reports saved as images.")