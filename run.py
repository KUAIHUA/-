import logging
import os
from config_utils import setup_logging, load_config
from data_process import import_data, perform_eda, perform_feature_engineering
from model_train import train_models

if __name__ == "__main__":
    # 1. 设置日志和配置
    setup_logging()
    config = load_config()
    
    try:
        # 2. 导入数据
        data_path = config['data']['path']
        df = import_data(data_path)
        logging.info(f"Data imported successfully: {df.shape}")
        
        # 3. 执行EDA
        perform_eda(df, config)
        logging.info("EDA completed successfully")
        
        # 4. 特征工程
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        logging.info(f"Feature engineering completed: train={X_train.shape}, test={X_test.shape}")
        
        # 5. 训练模型并生成结果
        preds = train_models(X_train, X_test, y_train, y_test, config)
        logging.info("Model training and evaluation completed successfully")
        
        print("\n=== 运行完成 ===")
        print(f"1. EDA结果已保存至: {config['eda']['save_path']}")
        print(f"2. 模型评估结果已保存至: {config['results']['save_path']}")
        print(f"3. 模型已保存至: {config['models']['save_path']}")
        print(f"4. 日志已保存至: ./logs/churn_library.log")
        print("\n请查看相应目录获取详细结果。")
        
    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        print(f"运行出错: {str(e)}")
        print("请查看日志文件获取详细错误信息。")
