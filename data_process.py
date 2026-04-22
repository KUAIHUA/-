import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from constants import LABEL_MAP, CAT_COLUMNS, KEEP_COLS, PLOT_STYLE 

def import_data(pth):
    """导入CSV数据并返回DataFrame（优化内存）"""
    dtype_config = {
        'Gender': 'category',
        'Education_Level': 'category',
        'Marital_Status': 'category',
        'Income_Category': 'category',
        'Card_Category': 'category'
    }
    df = pd.read_csv(pth, dtype=dtype_config)
    # 内存优化：只保留必要列（提前过滤无用列）
    all_required_cols = list(dtype_config.keys()) + [
        'Attrition_Flag', 'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
        'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
    ]
    df = df[all_required_cols].copy()
    
    logging.info(f"SUCCESS: Data imported (shape: {df.shape}, memory: {df.memory_usage().sum()/1024/1024:.2f}MB)")
    return df

def _draw_bar_chart(data, x_label, y_label, title, save_path, color='steelblue', rotation=0):
    """通用柱状图绘制函数（减少代码冗余）"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(data.index, data.values, color=color)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.xticks(rotation=rotation, ha='right')
    # 添加数值标签
    for bar, val in zip(bars, data.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (val*0.01 if val < 1 else 0.5), 
                f'{val:.1f}%' if y_label.endswith('(%)') else f'{val:.1%}', 
                ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_STYLE['dpi'], bbox_inches=PLOT_STYLE['bbox_inches'])
    plt.close()

def perform_eda(df, config):
    """执行EDA并保存可视化图表（优化资源释放）"""
    eda_path = config['eda']['save_path']
    os.makedirs(eda_path, exist_ok=True)
    # 提前生成流失标签（避免重复计算）
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    # 1. 流失分布饼图
    plt.figure(figsize=(10, 8))
    churn_counts = df['Attrition_Flag'].value_counts()
    churn_counts.index = [LABEL_MAP.get(x, x) for x in churn_counts.index]
    plt.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'], explode=(0, 0.05), shadow=True)
    plt.title('客户流失分布', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(eda_path, 'churn_distribution_pie.png'), **PLOT_STYLE)
    plt.close()

    # 2. 流失分布柱状图
    churn_counts = df['Attrition_Flag'].value_counts(normalize=True)
    churn_counts.index = [LABEL_MAP.get(x, x) for x in churn_counts.index]
    _draw_bar_chart(
        data=churn_counts,
        x_label='客户状态',
        y_label='比例',
        title='客户流失分布',
        save_path=os.path.join(eda_path, 'churn_distribution_bar.png'),
        color=['#2ecc71', '#e74c3c']
    )

    # 3. 客户年龄分布
    plt.figure(figsize=(12, 6))
    plt.hist(df['Customer_Age'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title('客户年龄分布', fontsize=14, fontweight='bold')
    plt.xlabel('年龄 (岁)', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.axvline(df['Customer_Age'].mean(), color='red', linestyle='--', 
                label=f'均值: {df["Customer_Age"].mean():.1f}')
    plt.axvline(df['Customer_Age'].median(), color='green', linestyle='--', 
                label=f'中位数: {df["Customer_Age"].median():.0f}')
    plt.legend()
    plt.savefig(os.path.join(eda_path, 'customer_age_distribution.png'), **PLOT_STYLE)
    plt.close()

    # 4. 婚姻状况分布
    marital_labels = {'Married': '已婚', 'Single': '单身', 'Divorced': '离异', 'Unknown': '未知', 'Uneducated': '未说明'}
    marital_counts = df['Marital_Status'].value_counts(normalize=True)
    marital_counts.index = [marital_labels.get(x, x) for x in marital_counts.index]
    _draw_bar_chart(
        data=marital_counts,
        x_label='婚姻状况',
        y_label='比例',
        title='婚姻状况分布',
        save_path=os.path.join(eda_path, 'marital_status_distribution.png'),
        color='coral',
        rotation=45
    )

    # 5. 收入等级分布
    income_labels = {
        'Less than $40K': '低于4万美元', '$40K - $60K': '4-6万美元',
        '$60K - $80K': '6-8万美元', '$80K - $120K': '8-12万美元',
        '$120K +': '12万美元以上', 'Unknown': '未知'
    }
    income_counts = df['Income_Category'].value_counts(normalize=True)
    income_counts.index = [income_labels.get(x, x) for x in income_counts.index]
    _draw_bar_chart(
        data=income_counts,
        x_label='收入等级',
        y_label='比例',
        title='收入等级分布',
        save_path=os.path.join(eda_path, 'income_distribution.png'),
        color='mediumseagreen',
        rotation=45
    )

    # 6. 总交易次数分布（区分流失）
    plt.figure(figsize=(12, 6))
    churn_labels_map = {'Existing Customer': '现有客户', 'Attrited Customer': '流失客户'}
    for status in df['Attrition_Flag'].unique():
        subset = df[df['Attrition_Flag'] == status]
        sns.kdeplot(subset['Total_Trans_Ct'], label=churn_labels_map.get(status, status), linewidth=2)
    plt.title('总交易次数分布 (按客户状态)', fontsize=14, fontweight='bold')
    plt.xlabel('交易次数', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.legend()
    plt.savefig(os.path.join(eda_path, 'total_transaction_count_by_churn.png'), **PLOT_STYLE)
    plt.close()

    # 7. 数值特征相关性热力图
    plt.figure(figsize=(16, 14))
    numeric_df = df.select_dtypes(include=[np.number])
    corr_df = numeric_df[['Customer_Age', 'Dependent_count', 'Months_on_book', 
                          'Total_Relationship_Count', 'Months_Inactive_12_mon',
                          'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                          'Total_Trans_Amt', 'Total_Trans_Ct', 'Avg_Utilization_Ratio']].copy()
    corr_df.columns = [LABEL_MAP.get(col, col) for col in corr_df.columns]
    mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
    sns.heatmap(corr_df.corr(), annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, mask=mask, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    plt.title('数值特征相关性热力图', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(eda_path, 'correlation_heatmap.png'), **PLOT_STYLE)
    plt.close()

    # 8. 各收入等级流失率
    df_temp = df.copy()
    df_temp['Income_Category_CN'] = df_temp['Income_Category'].map(income_labels)
    churn_by_income = df_temp.groupby('Income_Category_CN')['Churn'].mean() * 100
    churn_by_income = churn_by_income.sort_values(ascending=False)
    _draw_bar_chart(
        data=churn_by_income,
        x_label='收入等级',
        y_label='流失率 (%)',
        title='各收入等级流失率',
        save_path=os.path.join(eda_path, 'churn_rate_by_income.png'),
        color='salmon',
        rotation=45
    )

    # 9. 各卡类型流失率
    card_labels = {'Blue': '蓝卡', 'Silver': '银卡', 'Gold': '金卡', 'Platinum': '白金卡'}
    df_temp['Card_Category_CN'] = df_temp['Card_Category'].map(card_labels)
    churn_by_card = df_temp.groupby('Card_Category_CN')['Churn'].mean() * 100
    churn_by_card = churn_by_card.sort_values(ascending=False)
    _draw_bar_chart(
        data=churn_by_card,
        x_label='卡类型',
        y_label='流失率 (%)',
        title='各卡类型流失率',
        save_path=os.path.join(eda_path, 'churn_rate_by_card.png'),
        color='lightblue'
    )

    # 10. 不活跃月数与流失率
    churn_by_inactive = df.groupby('Months_Inactive_12_mon')['Churn'].mean() * 100
    _draw_bar_chart(
        data=churn_by_inactive,
        x_label='不活跃月数',
        y_label='流失率 (%)',
        title='近12月不活跃月数与流失率关系',
        save_path=os.path.join(eda_path, 'churn_rate_by_inactive_months.png'),
        color='lightcoral'
    )

    logging.info("SUCCESS: EDA completed and images saved.")

def encoder_helper(df, category_lst, response):
    """分类特征目标编码（优化循环效率）"""
    for cat in category_lst:
        # 避免重复计算：只在列不存在时生成
        if f'{cat}_{response}' not in df.columns:
            # 【修复点】: 添加 observed=True 以消除 FutureWarning
            # 当 groupby 的键是 category 类型且存在未出现的类别时，observed=True 只计算出现过的类别
            cat_groups = df.groupby(cat, observed=True)[response].mean()
            df[f'{cat}_{response}'] = df[cat].map(cat_groups)
    return df

def perform_feature_engineering(df, response='Churn'):
    """执行特征工程，返回训练/测试集（优化拆分逻辑）"""
    # 生成目标变量
    if response not in df.columns:
        df[response] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    y = df[response]

    # 目标编码
    df = encoder_helper(df, CAT_COLUMNS, response)
    
    # 过滤最终特征
    X = df[KEEP_COLS].copy()
    
    # 划分数据集（固定随机种子，分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y  # 分层抽样保证流失率分布一致
    )

    logging.info(f"SUCCESS: Feature engineering completed (train shape: {X_train.shape}, test shape: {X_test.shape})")
    return X_train, X_test, y_train, y_test