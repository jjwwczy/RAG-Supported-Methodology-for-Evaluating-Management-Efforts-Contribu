import os
import logging
import pandas as pd
from datetime import datetime

def save_reference_data_to_excel(reference_data, result, output_file=None, score_variable=''):
    """
    将引用数据和结果数据保存到Excel文件中的不同sheet。
    
    Args:
        reference_data (list): 包含引用信息的字典列表
        result (dict): 包含处理结果的字典
        output_file (str, optional): 输出文件路径。如果为None，将自动生成文件名。
        score_variable (str, optional): 得分变量名称
    
    Returns:
        str: 保存的文件路径
    """
    if not reference_data:
        logging.warning("没有引用数据可保存")
        return None
        
    # 如果没有指定输出文件，则自动生成文件名
    if not output_file:
        # 获取当前时间作为文件名的一部分
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 从第一条引用数据中获取数据集名称和查询
        dataset_name = reference_data[0].get('dataset_name', 'unknown_dataset')
        query_short = reference_data[0].get('query', 'unknown_query')[:20].replace(' ', '_')
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        output_file = os.path.join(output_dir, "运行记录.xlsx")
    
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 检查文件是否已存在
        file_exists = os.path.isfile(output_file)
        resultsheet='结果_'+str(score_variable)
        # 为所有数据添加时间戳列
        for item in reference_data:
            item['timestamp'] = timestamp
        
        # 为结果数据添加时间戳
        if isinstance(result, dict):
            result['timestamp'] = timestamp
        # 将result转换为DataFrame
        df_result = pd.DataFrame([result])
        
        # 将引用数据转换为DataFrame
        df_reference = pd.DataFrame(reference_data)
        # 创建ExcelWriter对象以写入多个sheet
        # 如果文件已存在，使用追加模式；否则创建新文件
        if file_exists:
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:

                
                # 处理结果数据sheet
                if resultsheet in writer.book.sheetnames:
                    # 读取现有数据
                    existing_df = pd.read_excel(output_file, sheet_name=resultsheet)
                    # 合并数据
                    combined_df = pd.concat([existing_df, df_result], ignore_index=True)
                    # 写入合并后的数据
                    combined_df.to_excel(writer, sheet_name=resultsheet, index=False)
                else:
                    # 如果sheet不存在，创建新sheet
                    df_result.to_excel(writer, sheet_name=resultsheet, index=False)
                
                # 处理引用数据sheet
                if '引用数据' in writer.book.sheetnames:
                    # 读取现有数据
                    existing_df = pd.read_excel(output_file, sheet_name='引用数据')
                    # 合并数据
                    combined_df = pd.concat([existing_df, df_reference], ignore_index=True)
                    # 写入合并后的数据
                    combined_df.to_excel(writer, sheet_name='引用数据', index=False)
                else:
                    # 如果sheet不存在，创建新sheet
                    df_reference.to_excel(writer, sheet_name='引用数据', index=False)
        else:
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
                # 如果文件不存在，直接写入新数据
                df_result.to_excel(writer, sheet_name=resultsheet, index=False)
                df_reference.to_excel(writer, sheet_name='引用数据', index=False)
               
        logging.info(f"数据已保存到: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"保存数据时出错: {e}")
        logging.exception("详细错误信息:")
        return None

import json

def parse_response(response_str: str) -> dict:
    """安全解析大模型返回的JSON字符串"""
    try:
        # 尝试直接解析
        data = json.loads(response_str)
        return data
    except json.JSONDecodeError:
        # 处理代码块标记，如 ```json ```
        import re
        # 移除代码块标记
        code_block_pattern = r'```(?:json|javascript|python)?\s*([\s\S]*?)```'
        code_block_match = re.search(code_block_pattern, response_str)
        if code_block_match:
            response_str = code_block_match.group(1)
        
        # 清理常见干扰符
        cleaned_str = response_str.strip("` \n")
        
        # 只保留括号包裹的内容
        bracket_pattern = r'(\{[\s\S]*?\})\s*\}\s*\]'
        bracket_match = re.search(bracket_pattern, cleaned_str)
        if bracket_match:
            # 找到了类似 {...} } ] 这样的模式，取第一个完整的JSON对象
            cleaned_str = bracket_match.group(1)
        else:
            # 尝试常规的括号匹配
            bracket_pattern = r'(\{[\s\S]*\})'
            bracket_match = re.search(bracket_pattern, cleaned_str)
            if bracket_match:
                cleaned_str = bracket_match.group(1)
        
        # 处理括号不成对的情况
        open_count = cleaned_str.count('{')
        close_count = cleaned_str.count('}')
        
        # 如果左括号多，移除最左边多余的括号
        if open_count > close_count:
            excess = open_count - close_count
            for _ in range(excess):
                if cleaned_str.startswith('{'):
                    cleaned_str = cleaned_str[1:]
        # 如果右括号多，移除最右边多余的括号
        elif close_count > open_count:
            excess = close_count - open_count
            for _ in range(excess):
                if cleaned_str.endswith('}'):
                    cleaned_str = cleaned_str[:-1]
        
        # 尝试修复常见的JSON错误
        # 1. 处理多余的逗号
        cleaned_str = re.sub(r',\s*}', '}', cleaned_str)
        cleaned_str = re.sub(r',\s*]', ']', cleaned_str)
        # 2. 处理注释标记 ##数字$$ 
        cleaned_str = re.sub(r'##\d+\$\$', '', cleaned_str)
        
        # 3. 处理多余的右括号和方括号
        # 找到最后一个有效的右花括号位置
        last_valid_brace = -1
        brace_count = 0
        for i, char in enumerate(cleaned_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_valid_brace = i
                    break
        
        if last_valid_brace > 0:
            cleaned_str = cleaned_str[:last_valid_brace+1]
        
        try:
            return json.loads(cleaned_str)
        except Exception as e:
            # 如果仍然无法解析，尝试更激进的修复
            try:
                # 使用正则表达式找到第一个完整的JSON对象
                json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
                json_match = re.search(json_pattern, cleaned_str)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    raise ValueError("无法找到有效的JSON对象")
            except Exception as e2:
                raise ValueError(f"无法解析JSON数据: {str(e)}, 进一步尝试: {str(e2)}")

def process_response(response_str: str) -> dict:
    """处理大模型返回的JSON数据，提取得分点概念、引用块索引和引用块内容，返回无嵌套的字典结构便于Excel记录"""
    
    result = {
        "总分": 0,
        "得分点数量": 0,
        "error": ""  # 默认为空字符串
    }
    
    try:
        # 解析原始数据（确保类型正确）
        raw_data = parse_response(response_str)
        if not isinstance(raw_data, dict):
            raise ValueError("返回数据非字典格式")

        # 添加得分点信息，使用编号确保属性名不重复
        points = raw_data.get("得分点", [])
        for i, point in enumerate(points, 1):
            if not isinstance(point, dict):
                continue
                
            # 为每个得分点添加带编号的属性
            result[f"得分点{i}_概念"] = str(point.get("概念", ""))
            result[f"得分点{i}_引用块索引"] = int(point.get("引用块索引", -1))
            result[f"得分点{i}_引用块"] = str(point.get("引用块", ""))
            result[f"得分点{i}_得分"] = int(point.get("得分", 0))
        
        # 计算总分
        total_score = sum(point.get("得分", 0) for point in points if isinstance(point, dict))
        result["总分"] = min(total_score, 100)  # 总分上限为100
        
        # 添加得分点数量信息
        result["得分点数量"] = len(points)
        
    except Exception as e:
        # 出错时，保持相同的字典结构，只是添加错误信息
        result["error"] = f"处理失败: {str(e)}"
    
    return result