import logging
import asyncio
from src.ragflow_ops import get_dataset_by_name
from ragflow_sdk.modules.chat import Chat
from src.utils import process_response

# ---------------------------------------------------------------------------
# Helper function for QA and Logging using RAGFlow API
# ---------------------------------------------------------------------------
async def perform_qa_and_log(client, dataset_name, query, vector_weight, top_k, llm_config, config=None, city=None, dimension=None):
    """
    Performs Retrieval Augmented Generation (RAG) for a given query and logs the process.
    Uses RAGFlow's official API to create chats and sessions.

    Args:
        client: Initialized RAGFlow client.
        dataset_name (str): The name of the dataset to use.
        query (str): The question to ask.
        vector_weight (float): The weight for vector similarity.
        top_k (int): The number of chunks to retrieve.
        llm_config (dict): Configuration for the LLM (model, base_url).
        config (dict, optional): The complete configuration dictionary.
                               Defaults to None.
        city (str, optional): The city name for city-specific templates and evaluation.
                              Defaults to None (use default template).
        dimension (str, optional): The dimension keyword for focused evaluation.
                                  Defaults to None (use default evaluation).
    """
    logging.info(f"使用查询进行问答: '{query}' (数据集: {dataset_name})")
    logging.info(f"使用向量相似度权重: {vector_weight}")

    try:
        # Set the optimal weight if found, otherwise use default
        final_vector_weight = vector_weight if vector_weight is not None else 0.4
        
        # --- Perform RAG using RAGFlow's official API --- #
        logging.info("\n--- 开始执行 RAG (使用 RAGFlow API) ---")
        
        if not query:
            logging.error("缺少查询，无法执行 RAG。")
            return
            
        # 1. 获取数据集对象
        target_dataset = get_dataset_by_name(client, dataset_name)
        if not target_dataset:
            logging.error(f"未能找到数据集 '{dataset_name}'，无法执行 RAG。")
            return
            
        # 2. 创建或查找聊天 (Chat)
        # 根据 https://ragflow.io/docs/dev/python_api_reference#list-chat-assistants
        chat_name = f"Chat_{dataset_name}"
        try:
            # 查找现有聊天
            existing_chat = None
            try:
                # 列出所有聊天并查找匹配的
                chats = client.list_chats(name=chat_name)
                for chat in chats:
                    if getattr(chat, 'name', '') == chat_name:
                        existing_chat = chat
                        logging.info(f"找到现有聊天: '{chat_name}' (ID: {chat.id})")
                        break
            except Exception as e:
                logging.warning(f"查找聊天时出错: {e}")
                existing_chat = None
                    
            if not existing_chat:
                # 创建新聊天
                # 根据文档设置正确的参数
                # 从配置中获取 LLM 模型名称
                model_name = llm_config.get('model', 'qwq:latest') if llm_config else 'qwq:latest'
                
                # 创建 LLM 配置
                llm_config_obj = {
                    'model_name': model_name,
                    'temperature': 0.1,  # 默认值
                    'top_p': 0.3,       # 默认值
                    'presence_penalty': 0.4,  # 默认值
                    'frequency_penalty': 0.7,  # 默认值
                    'response_format':{"type": "json_object"}
                }
                
                # 创建 Prompt 配置
                prompt_config = {
                    'similarity_threshold': 0.2,  # 默认值
                    'keywords_similarity_weight': 0.7,  # 调整为 0.3，增加向量相似度的权重
                    'top_n': top_k,  # 使用传入的 top_k 参数
                    'variables': [{
                        "key": "knowledge",
                        "optional": True
                    }], "rerank_model": "",
                    'top_k': top_k,  # 使用传入的 top_k 参数
                    'show_quote': True,  # 显示引用源
                    'prompt': """你是一个智能助手，请总结知识库的内容来回答问题，请列举知识库中的数据详细回答。当所有知识库内容都与问题无关时，你的回答必须包括“知识库中未找到您要的答案！”这句话。回答不需要考虑聊天历史。\n以下是知识库：\n{knowledge}\n以上是知识库。"""
                }
                
                # 创建新聊天
                chat = client.create_chat(
                    name=chat_name,
                    dataset_ids=[target_dataset.id],  # 关联到目标数据集
                    llm=Chat.LLM(client, llm_config_obj),  # 使用创建的 LLM 配置
                    prompt=Chat.Prompt(client, prompt_config)  # 使用创建的 Prompt 配置
                )
                logging.info(f"创建新聊天: '{chat_name}' (ID: {chat.id})")
            else:
                chat = existing_chat
                
            # 3. 创建会话 (Session)
            # 根据 https://ragflow.io/docs/dev/python_api_reference#create-a-session
            session_name = f"{query[:20]}"
            session = chat.create_session(name=session_name)
            logging.info(f"创建新会话: '{session_name}' (ID: {session.id})")
            
            # 4. 发送问题并获取回复
            # 根据 https://ragflow.io/docs/dev/python_api_reference#ask-a-question
            # 如果需要 JSON 格式的回答，可以在问题中指定
            
            # 正常查询
            cont = ""
            json_restriction=r'''请用JSON格式返回你的回答。格式模板如下，不要添加额外的属性：
            {"得分点":[{"概念":[得分点内容],"引用块索引":[chunk索引],"引用块":[chunk内容],"得分":[得分]}}
            '''
   
            query=query+json_restriction

            for ans in session.ask(question=query, stream=True):
                print(ans.content[len(cont):], end='', flush=True)
                cont = ans.content
            response = cont
            
            # 5. 记录回复内容
            if response:
                result=process_response(response)
                # 6. 获取并记录检索到的块
                # 根据 https://ragflow.io/docs/dev/python_api_reference#retrieve-chunks
                references = getattr(ans, 'reference', [])
                
                if references:
                    logging.info(f"--- 引用的块 ({len(references)} 个) ---")
                    
                    # 创建一个字典来存储引用信息，可以用于后续处理
                    reference_data = []
                    
                    for i, chunk in enumerate(references):
                        # 提取各种属性，使用 getattr 确保即使属性不存在也不会报错
                        chunk_id = chunk['id']
                        content = chunk['content']
                        document_id = chunk['document_id']
                        document_name = chunk['document_name']
                        image_id = chunk['image_id']
                        positions=str(chunk['positions'])

                        chunk_index=i
                        # similarity = chunk['similarity']
                        # vector_similarity = chunk['vector_similarity']
                        # term_similarity = chunk['term_similarity']
                        # 记录详细信息
                        # logging.info(f"  {i+1}. 块 ID: {chunk_id}")
                        # logging.info(f"     文档: {document_name} (ID: {document_id})")
                        # logging.info(f"     相似度: {similarity:.4f} (向量: {vector_similarity:.4f}, 关键词: {term_similarity:.4f})")
                        # logging.info(f"     内容: {content[:150]}..." if len(content) > 150 else f"     内容: {content}")
                        
                        # 将数据添加到引用列表中
                        reference_data.append({
                            'dataset_name': dataset_name,
                            'query': query,
                            'answer': response,
                            'chunk_index':chunk_index,
                            'id': chunk_id,
                            'document_name': document_name,
                            'document_id': document_id,
                            'content': content,
                            'image_id':image_id,
                            'positions':positions
                            # 'similarity': similarity,
                            # 'vector_similarity': vector_similarity,
                            # 'term_similarity': term_similarity,
                        })
                    result['dataset_name']=dataset_name
                    result['query']=query
                    result['answer']=response
                    # 如果需要将引用数据保存到文件或返回给调用者，可以使用 reference_data
                    return reference_data,result
                else:
                    logging.warning("未检索到相关块或没有引用信息。")
            else:
                logging.error("未收到助手回复。")
                
        except Exception as e:
            logging.error(f"使用 RAGFlow API 执行 RAG 时出错: {e}")
            logging.exception("详细错误信息:")
            
    except Exception as e:
        logging.error(f"执行问答时出错 (查询: '{query}'): {e}")
        logging.exception("Traceback:") # Log the full traceback

    logging.info("\n--- RAG 流程结束 ---")