import os
import yaml
import logging
from pathlib import Path
# 【重要】在导入 pyplot 之前设置后端，避免 GUI 初始化卡死
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
# config_utils.py
from constants import RC_PARAMS, PLOT_STYLE # 导入新的常量

def setup_logging():
    # ...
    # 【修复点】: 使用 RC_PARAMS 更新全局配置
    plt.rcParams.update(RC_PARAMS)
    # ...
    """设置日志目录和配置"""
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'churn_library.log'),
        level=logging.INFO,
        filemode='a',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 【修复点】：只提取 rcParams 支持的键进行更新
    # dpi 和 bbox_inches 不是 rcParams 的有效键，不能直接 update
    # 必须从 constants.PLOT_STYLE 中筛选出合法的 rcParams 键
    valid_rc_params = {
        'font.sans-serif': PLOT_STYLE.get('font.sans-serif', ['SimHei']),
        'axes.unicode_minus': PLOT_STYLE.get('axes.unicode_minus', False)
    }
    
    # 使用筛选后的字典更新 rcParams
    plt.rcParams.update(valid_rc_params)
    
    logging.info("SUCCESS: Logging and matplotlib config initialized.")

def load_config():
    """加载配置文件，转换相对路径为绝对路径"""
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / 'config.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_path}。请确保在该目录下创建config.yaml")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 路径标准化（相对路径转绝对路径）
    path_sections = ['data', 'eda', 'results', 'models', 'logs']
    for section in path_sections:
        if section in config and 'path' in config[section]:
            p = Path(config[section]['path'])
            config[section]['path'] = str(script_dir / p) if not p.is_absolute() else str(p)
        if section in config and 'save_path' in config[section]:
            p = Path(config[section]['save_path'])
            config[section]['save_path'] = str(script_dir / p) if not p.is_absolute() else str(p)

    logging.info(f"SUCCESS: Config loaded from {config_path}")
    return config