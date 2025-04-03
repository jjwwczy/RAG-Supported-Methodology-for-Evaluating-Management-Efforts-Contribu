import asyncio # 异步 IO
import logging
import ollama # Ollama 客户端库
import json
import re # 导入 re 模块
from ragflow_sdk import RAGFlow, Chunk # RAGFlow SDK 主类 和 Chunk 对象

# 配置基本日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve_chunks(client: RAGFlow, question: str, dataset_ids: list[str], vector_similarity_weight: float = 0.3, top_k: int = 10, **kwargs) -> list[Chunk]:
    """使用给定的参数从 RAGFlow 检索文档块。"""
    if not client:
        logging.error("RAGFlow 客户端未初始化。")
        return []
    try:
        logging.info(f"正在使用问题 '{question}', 向量相似度权重 {vector_similarity_weight}, top_k {top_k} 进行检索...")
        # 调用 RAGFlow 客户端的 retrieve 方法
        response = client.retrieve(
            question=question,
            dataset_ids=dataset_ids,
            vector_similarity_weight=vector_similarity_weight,
            top_k=top_k,
            **kwargs # 传递任何额外的检索参数
        )

        # 检查 response 类型：预期是 dict，但可能直接是 list
        if isinstance(response, dict):
            chunks = response.get('chunks', [])
        elif isinstance(response, list):
            # 假设如果返回列表，它直接就是块列表
            logging.warning("client.retrieve 返回了一个列表而不是预期的字典。直接将其视为块列表。")
            chunks = response
        else:
            logging.error(f"client.retrieve 返回了意外的类型: {type(response)}。无法提取块。")
            chunks = []

        logging.info(f"为权重 {vector_similarity_weight} 检索到 {len(chunks)} 个块。")
        return chunks if chunks else []
    except Exception as e:
        logging.error(f"使用向量相似度权重 {vector_similarity_weight} 检索块失败: {e}")
        return []

async def evaluate_relevance_with_ollama(ollama_config: dict, query: str, chunks: list[Chunk], top_n: int = 3) -> float:
    """使用 Ollama 异步评估检索到的块与查询的相关性。"""
    # 检查 Ollama 配置是否存在且包含必要信息
    if not ollama_config or not ollama_config.get('model') or not ollama_config.get('host'):
        logging.error("Ollama 配置不完整。需要 'model' 和 'host'。")
        return 0.0 # 返回默认值或引发错误

    # 如果没有检索到块，则返回 0.0 相关性
    if not chunks:
        logging.warning("未提供用于评估的块。")
        return 0.0

    # 仅选择前 N 个块进行评估
    chunks_to_evaluate = chunks[:top_n]
    
    # 提取块内容并格式化为字符串
    context = "\n\n---\n\n".join([chunk.content for chunk in chunks_to_evaluate])

    # 定义评估提示，要求模型输出一个 0.0 到 1.0 之间的数字评分
    prompt = (
        f"请根据以下提供的上下文信息来评估其与查询的相关性。"
        f"仅使用下面提供的上下文，不要依赖任何先前的知识。"
        f"请给出一个介于 0.0 (完全不相关) 到 1.0 (高度相关) 之间的浮点数评分。"
        f"\n\n查询: {query}"
        f"\n\n上下文:\n{context}"
        f"\n\n相关性评分 (0.0-1.0): "
        f"\n\n重要提示: 您的回答应只包含一个介于 0.0 和 1.0 之间的数字分数，不包含任何其他文本或解释。"
    )

    try:
        # 初始化 Ollama 异步客户端
        client = ollama.AsyncClient(host=ollama_config['host'])
        logging.debug(f"向 Ollama 发送评估请求。模型: {ollama_config['model']}, 主机: {ollama_config['host']}")
        
        # 调用 Ollama API
        response = await client.generate(
            model=ollama_config['model'],
            prompt=prompt,
            options=ollama_config.get('options', {}) # 允许传递额外的 Ollama 选项 (如 temperature)
        )
        
        # 从响应中提取生成的文本内容
        response_text = response.get('response', '').strip()
        logging.debug(f"收到 Ollama 的响应: {response_text}")
        
        # 使用正则表达式查找可能是分数的数字 (0.0 到 1.0 之间)
        match = re.search(r"([0-1](?:\.\d+)?|0|1)", response_text)
        
        if match:
            try:
                score_str = match.group(1) # 获取匹配到的数字字符串
                score = float(score_str)
                # 验证分数是否在有效范围内 (0.0 到 1.0)
                # (理论上 regex 已经部分保证了，但再确认一下更安全)
                if 0.0 <= score <= 1.0:
                    logging.info(f"Ollama 评估完成。查询: '{query[:30]}...', 分数: {score:.4f}")
                    return score
                else:
                    # 这理论上不应该发生，因为 regex 限制了范围
                    logging.warning(f"从 Ollama 响应中提取到无效分数 {score} (来自 '{response_text}')。将其视为 0.0。")
                    return 0.0
            except ValueError:
                # 如果提取的字符串无法转为 float (理论上也不应该发生)
                logging.warning(f"无法将从 Ollama 响应 '{response_text}' 中提取的字符串 '{score_str}' 解析为数字分数。将其视为 0.0。")
                return 0.0
        else:
            # 如果正则表达式没有找到匹配项
            logging.warning(f"无法从 Ollama 响应 '{response_text}' 中找到格式为 0.0-1.0 的数字分数。将其视为 0.0。")
            return 0.0

    except Exception as e:
        # 捕获与 Ollama API 调用相关的任何错误
        logging.error(f"调用 Ollama API 时出错: {e}")
        return 0.0 # 发生错误时返回 0.0


# 网格搜索函数
async def run_grid_search(client: RAGFlow, dataset_id: str, config: dict):
    """运行基于 Ollama 评估的 vector_similarity_weight 网格搜索。
    
    Args:
        client: RAGFlow 客户端
        dataset_id: 数据集 ID
        config: 包含网格搜索和 Ollama 评估配置的完整配置字典
        
    Returns:
        tuple: (最佳权重值, 最高分数)
    """
    # 获取网格搜索配置
    grid_search_config = config.get('grid_search', {}) 
    # 获取 Ollama 评估配置
    ollama_eval_config = config.get('ollama_evaluation', {})
    
    # 检查配置是否存在并且是否启用
    if not grid_search_config:
        logging.error("在配置中缺少 'grid_search' 部分")
        return None, -1.0
        
    if not grid_search_config.get('enabled', False):
        logging.info("网格搜索在配置中未启用")
        return None, -1.0
        
    if not ollama_eval_config:
        logging.error("在配置中缺少 'ollama_evaluation' 部分")
        return None, -1.0
        
    if not ollama_eval_config.get('enabled', False):
        logging.info("Ollama 评估在配置中未启用")
        return None, -1.0
        
    # 获取 Ollama 模型配置
    ollama_config = ollama_eval_config.get('model_config')
    if not ollama_config:
        logging.error("在 'ollama_evaluation' 配置中缺少 'model_config'")
        return None, -1.0

    # 从配置中读取网格搜索参数
    test_query = grid_search_config.get('test_query') # 测试用的查询语句
    weights_to_test = grid_search_config.get('vector_weights_to_test') # 需要测试的向量权重列表
    top_n_eval = grid_search_config.get('top_n_chunks_for_eval') # 评估时考虑的前 N 个块
    # 检索时的 top_k 应至少为 top_n_eval，也可以配置
    top_k_retrieval = grid_search_config.get('top_k_retrieval', 10) # 默认为 10，确保 >= top_n_eval

    # 检查必要参数是否存在
    if not test_query:
        logging.error("网格搜索配置中缺少 'test_query' 参数")
        return None, -1.0
        
    if not weights_to_test or not isinstance(weights_to_test, list) or len(weights_to_test) == 0:
        logging.error("网格搜索配置中缺少有效的 'vector_weights_to_test' 列表")
        return None, -1.0
        
    if not top_n_eval:
        logging.error("网格搜索配置中缺少 'top_n_chunks_for_eval' 参数")
        return None, -1.0

    # 确保检索的 top_k 不小于评估的 top_n
    if top_k_retrieval < top_n_eval:
        logging.warning(f"top_k_retrieval ({top_k_retrieval}) 小于 top_n_chunks_for_eval ({top_n_eval})。将 top_k_retrieval 设置为 {top_n_eval}")
        top_k_retrieval = top_n_eval

    logging.info(f"--- 开始向量权重的网格搜索 --- ")
    logging.info(f"测试查询: '{test_query}'")
    logging.info(f"待测试权重: {weights_to_test}")
    logging.info(f"使用 top_k={top_k_retrieval} 检索，评估前 {top_n_eval} 个块。")

    results = {} # 存储结果的字典
    evaluation_tasks = [] # 存储异步评估任务的列表

    # 为每个权重创建评估任务
    for weight in weights_to_test:
        # 检索块 (目前是同步调用，如果 retrieve 是异步的则可以改为异步)
        retrieved_chunks = retrieve_chunks(
            client=client,
            question=test_query,
            dataset_ids=[dataset_id],
            vector_similarity_weight=weight,
            top_k=top_k_retrieval
            # 如果需要，可以从配置中添加其他检索参数 (例如, similarity_threshold)
        )

        # 为此权重创建一个 Ollama 评估的异步任务
        task = asyncio.create_task(
            evaluate_relevance_with_ollama(
                ollama_config=ollama_config,
                query=test_query,
                chunks=retrieved_chunks,
                top_n=top_n_eval
            ),
            name=f"evaluate_weight_{weight}" # 为任务命名以便调试
        )
        evaluation_tasks.append(task) # 添加到任务列表
        results[weight] = {'task': task} # 临时存储任务

    # 等待所有评估任务完成
    logging.info("等待 Ollama 评估完成...")
    await asyncio.gather(*evaluation_tasks)

    # 从完成的任务中收集分数
    max_score = -1.0 # 初始化最高分
    best_weight = None # 初始化最佳权重
    for weight, data in results.items():
        try:
            score = data['task'].result() # 从完成的任务获取结果
            results[weight]['score'] = score # 存储分数
            logging.info(f"  - 权重: {weight}, Ollama 分数: {score:.4f}")
            # 更新最高分和最佳权重
            if score > max_score:
                max_score = score
                best_weight = weight
        except Exception as e:
            # 处理任务执行期间可能出现的异常 (在 evaluate 函数中已记录，但此处也捕获以防万一)
            logging.error(f"获取权重 {weight} 的结果时出错: {e}")
            results[weight]['score'] = 0.0 # 错误时分配默认分数

    logging.info("--- 网格搜索完成 --- ")
    if best_weight is not None:
        logging.info(f"找到的最佳 vector_similarity_weight: {best_weight}，分数: {max_score:.4f}")
        return best_weight, max_score # 成功时返回权重和分数
    else:
        logging.warning("无法从网格搜索结果中确定最佳权重。")
        return None, -1.0 # 找不到最佳权重时也返回可解包的值
    # 可选地，返回结果字典
    # return results
    # pass # 函数结束
