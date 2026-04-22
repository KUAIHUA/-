# 通用标签映射常量
LABEL_MAP = {
    # 目标变量
    'Attrition_Flag': '客户状态',
    'Existing Customer': '现有客户',
    'Attrited Customer': '流失客户',
    'Churn': '是否流失',
    
    # 人口统计特征
    'Customer_Age': '客户年龄',
    'Gender': '性别',
    'Gender_M': '男', 'Gender_F': '女',
    'Dependent_count': '受抚养人数',
    'Education_Level': '教育水平',
    'Marital_Status': '婚姻状况',
    'Income_Category': '收入等级',
    
    # 信用卡相关
    'Card_Category': '卡类型',
    'Credit_Limit': '信用额度',
    'Avg_Utilization_Ratio': '平均使用率',
    
    # 账户信息
    'Months_on_book': '开户月数',
    'Total_Relationship_Count': '关系总数',
    'Months_Inactive_12_mon': '近12月不活跃月数',
    'Contacts_Count_12_mon': '近12月联系次数',
    
    # 交易行为
    'Total_Revolving_Bal': '循环余额总额',
    'Avg_Open_To_Buy': '平均可用额度',
    'Total_Amt_Chng_Q4_Q1': '交易金额变化(Q4/Q1)',
    'Total_Trans_Amt': '总交易金额',
    'Total_Trans_Ct': '总交易次数',
    'Total_Ct_Chng_Q4_Q1': '交易次数变化(Q4/Q1)',
    
    # 模型评估
    'Precision': '精确率',
    'Recall': '召回率',
    'F1-score': 'F1分数',
    'Accuracy': '准确率',
    'ROC AUC': 'ROC曲线下面积',
    'True Positive Rate': '真正例率',
    'False Positive Rate': '假正例率',
    
    # 特征重要性
    'Feature Importance': '特征重要性',
    'Features': '特征'
}

# 分类特征列表（特征工程用）
CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

# 保留特征列表（特征工程用）
KEEP_COLS = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
    'Income_Category_Churn', 'Card_Category_Churn'
]

# 可视化样式配置
RC_PARAMS = {
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'],
    'axes.unicode_minus': False,
}

SAVE_FIG_KWARGS = {
    'dpi': 150,
    'bbox_inches': 'tight'
}

# 为了兼容旧代码或方便引用，可以保留 PLOT_STYLE 但仅包含保存参数
# 或者直接在代码中使用 SAVE_FIG_KWARGS
PLOT_STYLE = SAVE_FIG_KWARGS 