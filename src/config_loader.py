import yaml
import logging

# 配置基本日志记录 (如果项目复杂，可以考虑移到专门的日志配置文件)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path='config.yaml'):
    """从指定的 YAML 文件加载配置。"""
    try:
        # 以只读模式打开配置文件
        with open(config_path, 'r', encoding='utf-8') as stream: # 指定 utf-8 编码以支持中文等字符
            # 使用 PyYAML 安全地加载 YAML 内容
            config = yaml.safe_load(stream)
            logging.info(f"配置已成功从 {config_path} 加载.")
            return config
    except FileNotFoundError:
        # 如果配置文件未找到，记录错误
        logging.error(f"配置文件在路径 {config_path} 未找到.")
        return None
    except yaml.YAMLError as exc:
        # 如果 YAML 文件解析出错，记录错误
        logging.error(f"解析配置文件 {config_path} 时出错: {exc}")
        return None
    except Exception as e:
        # 捕获其他可能的异常，例如权限问题
        logging.error(f"加载配置文件 {config_path} 时发生未知错误: {e}")
        return None
