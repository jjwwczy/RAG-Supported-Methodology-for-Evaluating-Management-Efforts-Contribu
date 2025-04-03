import asyncio
import logging
import os
from ragflow_sdk import RAGFlow
from src import (
    load_config,
    upload_documents,
    run_grid_search, 
    retrieve_chunks, 
    generate_answer_from_chunks, 
    get_dataset_by_name,
)
from src.utils import save_reference_data_to_excel
from src.keywords_manager import create_keywords_manager
from QAandLog import perform_qa_and_log

# ---------------------------------------------------------------------------
# Main Execution Logic
# ---------------------------------------------------------------------------
async def main():
    # Load configuration
    config = load_config('config.yaml')
    if not config:
        logging.error("无法加载配置，程序终止。")
        return

    # --- Setup Logging --- #
    # Use dictionary access for config
    logging_config = config.get('logging', {})
    log_level_str = logging_config.get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = logging_config.get('format', '') # Use format from config if provided
    # Reconfigure logging with level and format from config
    # Ensure basicConfig is only called once effectively, or use handlers
    # For simplicity here, assuming basicConfig is okay if called early
    if log_format:
        logging.basicConfig(level=log_level, format=log_format, force=True)
    else:
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    logging.info("配置已成功从 config.yaml 加载.")
    # Initialize RAGFlow client
    ragflow_config = config.get('ragflow')
    if not ragflow_config or not ragflow_config.get('api_key') or not ragflow_config.get('base_url'):
        logging.error("RAGFlow API key 或 base URL 未在 config.yaml 中配置。")
        return
    client = RAGFlow(ragflow_config['api_key'], ragflow_config['base_url'])
    logging.info("RAGFlow 客户端初始化完成。")

    # --- Optional: Upload Documents --- #
    doc_upload_config = config.get('document_upload', {})
    folder_path = doc_upload_config.get('folder_path')
    dataset_prefix = doc_upload_config.get('dataset_prefix', 'Py_')
    
    # Construct dataset name using prefix and last part of folder path
    target_dataset_name = None
    if folder_path:
        # Extract the last part of the folder path
        folder_name = os.path.basename(os.path.normpath(folder_path))
        if folder_name:
            target_dataset_name = f"{dataset_prefix}{folder_name}"
            logging.info(f"自动构造数据集名称: '{target_dataset_name}'")
    
    embedding_model_for_creation = doc_upload_config.get('embedding_model') # Get model for creation

    if doc_upload_config.get('enabled'):
        logging.info("\n--- 开始处理文档上传 (如果已配置) ---")
        if target_dataset_name and folder_path and embedding_model_for_creation:
             # Pass the client object and potentially embedding model
             dataset_obj = get_dataset_by_name(client, target_dataset_name, embedding_model_for_creation)
             if dataset_obj:
                 # Call the upload_documents function with the correct parameters and pass the config
                 await upload_documents(dataset_obj, folder_path, allowed_extensions=('.txt', '.pdf', '.md', '.docx', '.doc'), config=config)
             else:
                 # Error logged within get_dataset_by_name if creation failed
                 logging.error(f"未能找到或创建数据集 '{target_dataset_name}'，无法上传文档。")
        else:
            if not target_dataset_name:
                logging.warning("文档上传已启用，但无法构造数据集名称。请确保 'folder_path' 有效。跳过上传。")
            if not folder_path:
                logging.warning("文档上传已启用，但未提供 'folder_path'。跳过上传。")
            if not embedding_model_for_creation:
                logging.warning("文档上传已启用，但未提供 'embedding_model'。无法确保数据集存在。跳过上传。")
        logging.info("文档上传处理结束。")
    else:
        logging.info("\n--- 文档上传已禁用 (在 config.yaml 中) ---")

    # --- 注意: 根据用户反馈，RAGFlow API 不支持更新文档启用/禁用状态 --- #
    # 如需按关键词区分文档，请使用多个数据集
    logging.info("\n--- 按关键字更新文档状态功能已移除 ---")

    # --- Optional: Grid Search --- #
    retrieval_config = config.get('retrieval', {})
    grid_search_config = config.get('grid_search', {})
    ollama_eval_config = config.get('ollama_evaluation', {})
    # Use default weight from retrieval config
    optimal_weight = retrieval_config.get('default_vector_weight', 0.4)

    if grid_search_config.get('enabled') and ollama_eval_config.get('enabled'):
        logging.info("\n--- 开始处理网格搜索 (如果已配置) ---")
        eval_llm_model_config = ollama_eval_config.get('model_config')
        test_query_gs = grid_search_config.get('test_query') # Renamed for clarity
        if not eval_llm_model_config:
             logging.error("Ollama 评估已启用，但缺少 'model_config'。无法执行网格搜索。")
        elif not test_query_gs:
             logging.error("网格搜索已启用，但未提供 'test_query'。无法执行网格搜索。")
        elif not target_dataset_name:
             logging.error("网格搜索已启用，但无法构造目标数据集名称。请确保 'folder_path' 有效。无法执行网格搜索。")
        else:
             # Pass the RAGFlow client, dataset ID (need to get it), and config
             # We need the dataset ID here. Let's try getting the object again.
             # Assuming grid search needs the same dataset as uploads.
             dataset_obj_gs = get_dataset_by_name(client, target_dataset_name, embedding_model_for_creation)
             if dataset_obj_gs:
                 optimal_weight_result = await run_grid_search(
                     client=client,
                     dataset_id=dataset_obj_gs.id, # Pass the ID
                     config=config # Pass the whole config for nested access
                 )
                 if optimal_weight_result is not None:
                     optimal_weight = optimal_weight_result
                     logging.info(f"网格搜索完成。找到最佳向量权重: {optimal_weight}")
                 else:
                     logging.warning("网格搜索未返回有效权重，将使用默认值。")
             else:
                 # Error logged within get_dataset_by_name
                 logging.error(f"未能找到或创建数据集 '{target_dataset_name}'，无法执行网格搜索。")

        logging.info("网格搜索处理结束。")
    else:
        logging.info("\n--- 网格搜索或 Ollama 评估未配置/启用，跳过网格搜索。将使用默认权重。 ---")
        logging.info(f"使用向量相似度权重: {optimal_weight}")

    ########################## --- QA Example --- ###########################################
    qa_config = config.get('qa_example', {})
    generation_config = config.get('generation', {})
    llm_config = generation_config.get('llm_config')
    queries = qa_config.get('queries', [])
    score_variable = qa_config.get('score_variable', '')

    if qa_config.get('enabled') and llm_config and queries and target_dataset_name:
        logging.info("\n--- 开始执行问答与溯源示例 ---")
        for query in queries:
            logging.info(f"使用向量相似度权重: {optimal_weight}")

            # Get the dataset object (needed for perform_qa_and_log? Check definition)
            # Let's assume perform_qa_and_log needs the dataset ID or name
            # We already have target_dataset_name

            # 创建关键词管理器
            keywords_manager = create_keywords_manager(config)
            
            # 从查询中检测城市和维度关键词
            city, dimension = keywords_manager.detect_keywords_in_query(query)
            
            if city:
                logging.info(f"检测到查询与城市 '{city}' 相关")
            if dimension:
                logging.info(f"检测到查询与维度 '{dimension}' 相关")
                    
            # Call the perform_qa_and_log function with city and dimension parameters if detected
            reference_data,result=await perform_qa_and_log(
                client=client,
                dataset_name=target_dataset_name, # Pass dataset name
                query=query,
                vector_weight=optimal_weight,
                top_k=retrieval_config.get('top_k', 10), # Example: get top_k from retrieval section
                llm_config=llm_config,
                config=config, # Pass the complete config
                city=city, # Pass the detected city
                dimension=dimension # Pass the detected dimension
            )
            logging.info("--- 单次问答结束 ---") # Added separator

            # 把引用数据保存到excel文件中
            if reference_data:
                # 自动生成输出文件名
                save_reference_data_to_excel(reference_data=reference_data,result=result,output_file=None, score_variable=score_variable)

        logging.info("所有问答与溯源示例执行完毕。")
    else:
        if not qa_config.get('enabled'):
            logging.info("\n--- 问答示例已禁用 (在 config.yaml 中) ---")
        if not llm_config:
            logging.warning("问答示例需要 'generation.llm_config' 配置。")
        if not queries:
            logging.warning("问答示例需要 'qa_example.queries' (非空列表) 配置。")
        if not target_dataset_name:
             logging.warning("问答示例需要有效的 'folder_path' 来构造目标数据集名称。")

    logging.info("\nAutoRAG 脚本执行完毕。")

if __name__ == "__main__":
    # Ensure environment variables are loaded if using .env
    # from dotenv import load_dotenv
    # load_dotenv()
    asyncio.run(main())
