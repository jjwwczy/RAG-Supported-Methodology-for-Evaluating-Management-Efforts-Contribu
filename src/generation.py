# -*- coding: utf-8 -*-
import logging
import ollama
import json
import re
from ragflow_sdk import Chunk
from pydantic import BaseModel, ValidationError
from typing import Tuple, List, Dict, Optional, Any, Union
from .policy_schemas import get_city_schema, evaluate_city_policy
from .keywords_manager import KeywordsManager, create_keywords_manager

def create_default_json_prompt(query: str, context_text: str) -> str:
    """
    创建默认的 JSON 格式提示模板。
    
    Args:
        query: 用户查询
        context_text: 上下文文本
        
    Returns:
        格式化后的提示字符串
    """
    return f"""请严格根据下面提供的上下文信息来回答问题。请只使用提供的信息，不要添加任何外部知识或进行猜测。如果上下文信息不足以回答问题，请明确说明。

上下文:
---
{context_text}
---

问题: {query}

请将你的最终回答格式化为一个 JSON 对象，该对象包含一个名为 'answer' 的键，其值为字符串形式的回答。例如：
{{"answer": "这是您的回答。"}}

JSON 回答:"""

# --- Define Pydantic Models --- 
class OllamaAnswer(BaseModel):
    answer: str
    
class PolicyExtractionResult(BaseModel):
    """Result of policy extraction with both raw JSON and evaluation scores."""
    city: str
    raw_json: Dict[str, Any]
    evaluation_scores: Dict[str, float]

async def generate_answer_from_chunks(ollama_config: dict, query: str, chunks: list[Chunk], city: Optional[str] = None, dimension: Optional[str] = None, config: Optional[dict] = None) -> tuple[Optional[str], list[Chunk]]:
    """
    使用 Ollama 模型根据提供的文本块生成对查询的回答，并包含溯源信息。
    要求 Ollama 以 JSON 格式返回回答。

    Args:
        ollama_config (dict): 包含 Ollama 连接信息的字典 (例如, 'host', 'model', 'options')。
        query (str): 用户的原始查询。
        chunks (list[Chunk]): 从 RAGFlow 检索到的相关文本块列表。

    Returns:
        tuple[Optional[str], list[Chunk]]: 一个元组，包含从 JSON 解析出的回答文本字符串。
                                           如果生成或解析失败，则返回 (None, chunks)。
    """
    if not chunks:
        logging.warning("没有提供用于生成回答的文本块。")
        # 返回 None 表示没有生成回答
        return None, chunks

    # 从配置中获取 Ollama 客户端参数
    host = ollama_config.get('host')
    model = ollama_config.get('model')
    options = ollama_config.get('options', {})

    if not host or not model:
        logging.error("Ollama 配置不完整，缺少 'host' 或 'model'。")
        return None, chunks

    # --- 构建 Prompt ---
    # 提取每个块的内容
    context_text = "\n---\n".join([chunk.content for chunk in chunks if hasattr(chunk, 'content')])
    
    # 使用关键词管理器获取适合的模板
    if config:
        keywords_manager = create_keywords_manager(config)
        
        # 如果没有明确指定城市和维度，尝试从查询中检测
        if not city or not dimension:
            detected_city, detected_dimension = keywords_manager.detect_keywords_in_query(query)
            city = city or detected_city
            dimension = dimension or detected_dimension
            if detected_city:
                logging.info(f"从查询中检测到城市: {detected_city}")
            if detected_dimension:
                logging.info(f"从查询中检测到维度: {detected_dimension}")
        
        # 获取适合的模板
        template = keywords_manager.get_or_create_template(city, dimension)
        if city or dimension:
            city_info = f"城市: {city}" if city else ""
            dimension_info = f"维度: {dimension}" if dimension else ""
            combined_info = " ".join(filter(None, [city_info, dimension_info]))
            logging.info(f"使用特定模板生成回答 ({combined_info})")
        else:
            logging.info("使用默认模板生成回答")
            
        # 使用模板格式化提示
        prompt = template.format(context=context_text)
    else:
        # 如果没有配置，使用默认的 JSON 提示
        prompt = create_default_json_prompt(query, context_text)
        logging.warning("未提供配置，使用默认 JSON 提示")

    logging.info(f"准备向 Ollama 模型 '{model}' 发送生成请求。")
    logging.debug(f"使用的 Prompt (前 500 字符): {prompt[:500]}...")

    # --- 调用 Ollama API ---
    try:
        client = ollama.AsyncClient(host=host)
        response = await client.generate(
            model=model,
            prompt=prompt,
            options=options,
            format="json" # <- 明确要求 Ollama 输出 JSON
        )

        generated_json_text = response.get('response', '').strip()
        logging.debug(f"Ollama 返回的原始 JSON 文本: {generated_json_text}")

        # --- 解析 JSON 响应 ---
        try:
            # 尝试提取有效的 JSON 部分（以防模型输出了额外的文本）
            json_match = re.search(r'\{.*\}', generated_json_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)
            else:
                parsed_json = json.loads(generated_json_text)
            
            # 如果是城市特定的模式，进行特殊处理
            if city:
                # 获取城市特定的 schema
                schema_class = get_city_schema(city)
                if schema_class:
                    try:
                        # 尝试验证 JSON 数据符合城市特定的 schema
                        validated_data = schema_class(**parsed_json)
                        # 评估政策
                        scores = evaluate_city_policy(parsed_json, city, dimension)
                        
                        # 记录评估信息
                        eval_info = f"{city}" + (f"_{dimension}" if dimension else "")
                        logging.info(f"{eval_info} 政策评估分数: {scores}")
                        
                        # 创建包含原始 JSON 和评估分数的结果
                        result_json = {
                            "policy_data": validated_data.dict(),
                            "evaluation": scores,
                            "city": city,
                            "dimension": dimension
                        }
                        return json.dumps(result_json, ensure_ascii=False, indent=2), chunks
                    except ValidationError as ve:
                        logging.error(f"城市政策数据验证失败: {ve}")
                        # 如果验证失败，仍然返回原始 JSON 和错误信息
                        result_json = {
                            "error": f"数据验证失败: {str(ve)}",
                            "raw_data": parsed_json,
                            "city": city,
                            "dimension": dimension
                        }
                        return json.dumps(result_json, ensure_ascii=False, indent=2), chunks
            
            # 默认情况：如果有 'answer' 字段，使用 OllamaAnswer 模型
            if 'answer' in parsed_json:
                validated_answer = OllamaAnswer(**parsed_json)
                logging.info("成功从 Ollama 响应解析出 Pydantic 模型。")
                return validated_answer.answer, chunks
            else:
                # 如果没有 'answer' 字段但 JSON 有效，直接返回格式化的 JSON
                logging.info("Ollama 返回了有效的 JSON，但没有 'answer' 字段。直接返回格式化的 JSON。")
                return json.dumps(parsed_json, ensure_ascii=False, indent=2), chunks
                
        except ValidationError as ve:
            logging.error(f"无法将 Ollama 响应解析为 Pydantic 模型: {ve}")
            logging.error(f"无效的 JSON 响应: {generated_json_text}")
            return None, chunks # 解析失败
        except json.JSONDecodeError as je:
            logging.error(f"Ollama 返回的不是有效的 JSON: {je}")
            logging.error(f"无效的文本响应: {generated_json_text}")
            return None, chunks # JSON 格式错误

    except Exception as e:
        logging.error(f"调用 Ollama API 生成回答时出错: {e}")
        return None, chunks

# --- 可以在这里添加一个简单的测试函数（如果需要） ---
# async def main_test():
#     # 示例配置和数据
#     test_ollama_config = {'host': 'http://localhost:11434', 'model': 'llama3'}
#     test_query = "杭州市的绿地规划政策是什么？"
#     # 模拟 Chunk 对象 (你需要根据实际 Chunk 结构调整)
#     class MockChunk:
#         def __init__(self, id, content):
#             self.id = id
#             self.content = content # 假设有这个属性
#             self.name = f"doc_{id}.pdf" # 假设来源文件名
#     
#     test_chunks = [
#         MockChunk("chunk1", "杭州市致力于提升城市绿化覆盖率，计划在未来五年内新增公园绿地 500 公顷。"),
#         MockChunk("chunk2", "根据《杭州市城市绿地系统规划（2020-2035）》，将重点建设沿江、沿河、沿山生态廊道。"),
#         MockChunk("chunk3", "老旧小区改造也将纳入绿化提升计划，增加居民身边的口袋公园。")
#     ]
#
#     generated_answer, source_chunks = await generate_answer_from_chunks(test_ollama_config, test_query, test_chunks)
#
#     if generated_answer is not None:
#         print("--- 生成的回答 ---")
#         print(generated_answer)
#         print("\n--- 答案来源 (Chunks) ---")
#         for i, chunk in enumerate(source_chunks):
#             print(f"{i+1}. Chunk ID: {chunk.id}, Content (部分): {chunk.content[:100]}... (来自: {getattr(chunk, 'name', 'N/A')}) ") # 假设有 name 属性
#     else:
#         print("生成回答失败。")
#
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO) # 设置日志级别
#     # 使用 asyncio 运行测试函数
#     import asyncio
#     asyncio.run(main_test())
