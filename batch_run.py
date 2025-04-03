#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yaml
import os
import subprocess
import logging
import pandas as pd
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 配置文件路径
CONFIG_PATH = "./config.yaml"
# Excel文件路径
SCORE_VARIABLES_FILE = "./得分变量提示词.xlsx"
# 文档目录
DOCS_DIR = "./docs"

def load_config():
    """加载配置文件"""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {e}")
        return None

def save_config(config):
    """保存配置文件"""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        logging.error(f"保存配置文件失败: {e}")
        return False

def update_config(config, folder_path, score_variable, query):
    """更新配置文件中的关键配置"""
    # 更新文件夹路径
    if 'document_upload' in config and 'folder_path' in config['document_upload']:
        config['document_upload']['folder_path'] = folder_path
    
    # 更新得分变量
    if 'qa_example' in config:
        if 'score_variable' not in config['qa_example']:
            config['qa_example']['score_variable'] = ""
        config['qa_example']['score_variable'] = score_variable
    
    # 更新查询
    if 'qa_example' in config and 'queries' in config['qa_example']:
        config['qa_example']['queries'] = [query]
    
    return config

def run_main_py():
    """运行main.py脚本"""
    try:
        process = subprocess.Popen(['python', 'main.py'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logging.error(f"运行main.py失败: {stderr}")
            return False
        
        logging.info("main.py运行成功")
        return True
    except Exception as e:
        logging.error(f"运行main.py时发生错误: {e}")
        return False

def get_folder_list():
    """获取docs文件夹下的所有子文件夹"""
    try:
        if not os.path.exists(DOCS_DIR):
            logging.error(f"文档目录 {DOCS_DIR} 不存在")
            return []
            
        folders = [os.path.join(DOCS_DIR, f) for f in os.listdir(DOCS_DIR) 
                  if os.path.isdir(os.path.join(DOCS_DIR, f))]
        
        # 转换为相对路径格式
        folders = [f"./docs/{os.path.basename(f)}" for f in folders]
        
        logging.info(f"找到 {len(folders)} 个文档文件夹: {folders}")
        return folders
    except Exception as e:
        logging.error(f"获取文件夹列表失败: {e}")
        return []

def read_score_variables():
    """从Excel文件中读取得分变量和查询提示词"""
    try:
        if not os.path.exists(SCORE_VARIABLES_FILE):
            logging.error(f"得分变量文件 {SCORE_VARIABLES_FILE} 不存在")
            return []
            
        df = pd.read_excel(SCORE_VARIABLES_FILE)
        
        # 假设Excel文件有两列：score_variable和queries
        if 'score_variable' not in df.columns or 'queries' not in df.columns:
            logging.error(f"Excel文件格式不正确，需要包含'score_variable'和'queries'列")
            return []
            
        # 将DataFrame转换为字典列表
        variables = []
        for _, row in df.iterrows():
            variables.append({
                'score_variable': row['score_variable'],
                'queries': row['queries']
            })
            
        logging.info(f"从Excel文件中读取了 {len(variables)} 个得分变量")
        return variables
    except Exception as e:
        logging.error(f"读取得分变量失败: {e}")
        return []

def generate_batch_config():
    """生成批量运行的配置"""
    folders = get_folder_list()
    if not folders:
        return []
        
    variables = read_score_variables()
    if not variables:
        return []
        
    # 生成所有文件夹和得分变量的组合
    batch_config = []
    for folder in folders:
        for var in variables:
            batch_config.append({
                'folder_path': folder,
                'score_variable': var['score_variable'],
                'queries': var['queries']
            })
            
    logging.info(f"生成了 {len(batch_config)} 个批量运行配置")
    return batch_config

def main():
    """主函数"""
    start_time = datetime.now()
    
    # 生成批量运行的配置
    batch_config = generate_batch_config()
    if not batch_config:
        logging.error("无法生成批量运行配置，程序退出")
        return
    
    logging.info(f"批量运行开始，共{len(batch_config)}个配置")
    
    success_count = 0
    
    for i, batch in enumerate(batch_config, 1):
        logging.info(f"正在处理第{i}个配置: {batch['folder_path']} - {batch['score_variable']}")
        
        # 加载配置
        config = load_config()
        if not config:
            logging.error("无法加载配置，跳过此批次")
            continue
        
        # 更新配置
        config = update_config(
            config, 
            batch['folder_path'], 
            batch['score_variable'], 
            batch['queries']
        )
        
        # 保存配置
        if not save_config(config):
            logging.error("无法保存配置，跳过此批次")
            continue
        
        # 运行main.py
        if run_main_py():
            success_count += 1
        
        logging.info(f"第{i}个配置处理完成")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logging.info(f"批量运行完成，成功: {success_count}/{len(batch_config)}")
    logging.info(f"总耗时: {duration:.2f}秒")

if __name__ == "__main__":
    main()