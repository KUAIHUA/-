# 银行客户流失预测项目

基于 `BankChurners.csv` 数据集的机器学习项目，用于完成客户流失（Churn）分析与预测。  
项目包含完整流程：数据导入、EDA 分析、特征工程、多模型训练（逻辑回归 / 随机森林 / LightGBM）、评估报告与图表输出。

## 项目特性

- 一键运行完整流程（`run.py`）
- 自动生成 EDA 图表（分布、相关性、流失率分析等）
- 训练并保存多个模型与标准化器
- 输出中文化评估结果（混淆矩阵、ROC 对比、特征重要性、文本报告）
- 日志统一写入 `logs/churn_library.log`

## 项目结构

```text
xianmu4/
├─ run.py                   # 主入口脚本（推荐运行）
├─ data_process.py          # 数据导入、EDA、特征工程
├─ model_train.py           # 模型训练与评估结果输出
├─ config_utils.py          # 日志与配置加载
├─ constants.py             # 常量配置（字段映射、图表参数等）
├─ report_utils.py          # 报告可视化辅助函数
├─ config.yaml              # 路径配置文件
├─ BankChurners.csv         # 数据文件
├─ images/
│  └─ results/              # 示例结果目录（运行后会继续产出）
└─ logs/
   └─ churn_library.log
```

## 环境要求

- Python 3.9 及以上（建议 3.10+）
- Windows / macOS / Linux 均可

核心依赖：

- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- seaborn
- joblib
- pyyaml

## 安装与运行

### 1) 创建并激活虚拟环境（可选但推荐）

Windows PowerShell:

```powershell   “‘powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) 安装依赖

```powershell   “‘powershell
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn joblib pyyamlPIP安装pandas numpy scikit-learn lightgbm matplotlib seaborn joblib pyyaml
```

### 3) 运行项目

```powershell   “‘powershell
python run.py
```

## 配置说明

项目通过 `config.yaml` 管理输入与输出路径，默认配置如下：

- `data.path`: 输入数据文件（默认 `BankChurners.csv`）
- `eda.save_path`: EDA 图表输出目录（默认 `./images/eda`）
- `results.save_path`: 模型评估结果目录（默认 `./images/results`）
- `models.save_path`: 训练后模型目录（默认 `./models`）
- `logs.save_path`: 日志目录（默认 `./logs`）

> `config_utils.py` 会自动将相对路径解析为项目根目录下的绝对路径。

## 运行产出

运行完成后，主要产物包括：

- `images/eda/`：EDA 分析图（流失分布、年龄分布、相关性热力图等）
- `images/results/`：
  - 各模型混淆矩阵
  - 各模型特征重要性图
  - ROC 曲线对比图
  - `模型评估报告.txt`
- `models/`：`lr_model.pkl`、`rf_model.pkl`、`lgb_model.pkl`、`scaler.pkl`
- `logs/churn_library.log`：完整运行日志

## 常见问题

### 1) LightGBM 安装失败

先升级 `pip` 后重试：

```powershell
python -m pip install --upgrade pipPython -m PIP install——升级PIP
pip install lightgbm
```

### 2) 中文显示异常（图表乱码）

项目已内置常见中文字体配置；若系统仍缺字体，请安装中文字体（如微软雅黑）后重试。

### 3) 找不到数据文件

请确认 `BankChurners.csv` 位于项目根目录，或在 `config.yaml` 中修改 `data.path` 为正确路径。

## 说明

- 当前推荐入口为 `run.py`。
- 若你后续准备交作业或上线演示，建议补充：
  - `requirements.txt`
  - 实验结果截图
  - 模型效果对比结论（业务解释）
