import logging
from ragflow_sdk import RAGFlow # 导入 RAGFlow SDK 主类

# 配置基本日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_ragflow_client(ragflow_config):
    """根据传入的 RAGFlow 配置字典初始化并返回 RAGFlow 客户端实例。"""
    # 检查传入的配置字典是否存在
    if not ragflow_config:
        logging.error("未提供 RAGFlow 配置字典。")
        return None

    # 直接从传入的字典中获取 API Key 和基础 URL
    api_key = ragflow_config.get('api_key')
    base_url = ragflow_config.get('base_url')

    # 检查 API Key 和 URL 是否都已提供
    if not api_key or not base_url:
        logging.error("RAGFlow 配置中缺少 API Key 或 Base URL。")
        return None

    try:
        # 实例化 RAGFlow 客户端
        # 确保已安装 ragflow-sdk: pip install ragflow-sdk
        client = RAGFlow(api_key=api_key, base_url=base_url)
        logging.info(f"RAGFlow 客户端已成功初始化，目标地址: {base_url}.")
        return client
    except Exception as e:
        logging.error(f"初始化 RAGFlow 客户端失败: {e}")
        return None
