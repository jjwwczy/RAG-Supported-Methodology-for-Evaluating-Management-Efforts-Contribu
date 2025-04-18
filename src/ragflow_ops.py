import os
import logging
import hashlib
import time
from ragflow_sdk import RAGFlow, DataSet, Document # 确保导入了所需类

# 配置基本日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 主要函数：
# 1. get_dataset_by_name - 获取或创建数据集
# 2. upload_document_to_dataset - 上传单个文档（由upload_documents调用）
# 3. upload_documents - 上传文件夹中的所有文档
# 4. parse_single_document - 解析单个文档（由parse_documents_in_dataset调用）
# 5. parse_documents_in_dataset - 解析数据集中的文档
# 6. list_documents_in_dataset - 列出数据集中的文档（由其他函数调用）
# 7. update_document_status_by_keyword - 根据关键字更新文档状态

def get_dataset_by_name(client: RAGFlow, dataset_name: str, embedding_model: str = None) -> DataSet | None:
    """根据名称查找数据集，如果不存在则创建一个新的数据集。
    
    Args:
        client: RAGFlow 客户端实例
        dataset_name: 要查找或创建的数据集名称
        embedding_model: 创建新数据集时使用的嵌入模型名称
        
    Returns:
        找到或创建的 DataSet 对象，如果操作失败则返回 None
    """
    if not client:
        logging.error("RAGFlow 客户端未初始化。")
        return None
        
    try:
        # 尝试直接使用客户端列出所有数据集
        try:
            datasets = client.list_datasets(name=dataset_name)
            logging.info(f"检索到 {len(datasets) if datasets else 0} 个同名数据集。")
        except Exception as list_error:
            logging.error(f"列出数据集失败: {list_error}")
            datasets = []
            
        if not datasets:
            logging.error("无法从 RAGFlow 获取数据集列表或列表为空，将尝试创建新数据集。")
            
        # 查找匹配名称的数据集
        for dataset in datasets:
            if dataset.name == dataset_name:
                logging.info(f"找到数据集 '{dataset_name}' (ID: {dataset.id})")
                return dataset
                
        # 如果未找到，则创建新数据集
        logging.info(f"未找到名为 '{dataset_name}' 的数据集，将创建新数据集。")
            ####先批量更新解析参数#####

        '''
        论文的话不使用naive，而是使用paper：
        chunk_method="paper":
        {"raptor": {"use_raptor": False}}

        这里有个巨坑：文档里写的是user_raptor,实际上应该为use_raptor，且必须跟上prompt、maxtoken等属性的内容。
        如果不想用deepdoc，可以用千问的模型，添加下面的属性即可。
        "layout_recognize": "qwen-vl-plus@Tongyi-Qianwen"   "DeepDOC"    #待验证
        '''
        parser_config = DataSet.ParserConfig( client,{"chunk_token_num":128,"delimiter":"\\n!?;。；！？","html4excel":False,"layout_recognize":"DeepDOC","raptor":{"use_raptor":False,"prompt":"请总结以下段落。 小心数字，不要编造。 段落如下：\n{cluster_content}\n以上就是你需要总结的内容。","max_token":256,"threshold":0.10,"max_cluster":64,"random_seed":0}})

        create_params = {"name": dataset_name,"chunk_method":"naive",
        "parser_config":parser_config}
        
        # 如果提供了嵌入模型，则使用它
        if embedding_model:
            create_params["embedding_model"] = embedding_model
            # create_params["document_parser"] ="Naive"

            logging.info(f"使用指定的嵌入模型: '{embedding_model}'")
        else:
            logging.info("未指定嵌入模型，将使用 RAGFlow 默认模型。")
            
        # 创建新数据集
        new_dataset = client.create_dataset(**create_params)
        logging.info(f"成功创建新数据集: '{new_dataset.name}' (ID: {new_dataset.id})")
        return new_dataset
        
    except Exception as e:
        logging.error(f"查找或创建数据集 '{dataset_name}' 时出错: {e}")
        return None

# 注意：已删除未使用的 create_dataset 函数
# 因为 get_dataset_by_name 函数已经包含了创建数据集的功能

# 注意：已删除未使用的 find_or_create_dataset 函数
# 因为 get_dataset_by_name 函数已经包含了查找或创建数据集的功能


def upload_document_to_dataset(dataset_object: DataSet, file_path: str) -> tuple[bool, str]:
    """读取本地文件并将其上传到指定的数据集对象。
    
    Args:
        dataset_object: 目标数据集对象
        file_path: 要上传的文件路径
        
    Returns:
        tuple: (成功标志, 文档ID) - 成功时返回 (True, 文档ID)，失败时返回 (False, '')
    """
    if not dataset_object:
        logging.error("为上传操作提供了无效的 DataSet 对象。")
        return False, ''
    if not os.path.exists(file_path):
        logging.error(f"文件在路径 {file_path} 未找到。")
        return False, ''

    file_name = os.path.basename(file_path) # 获取文件名
    try:
        # 以二进制读取模式打开文件
        with open(file_path, "rb") as f:
            blob_content = f.read() # 读取文件内容

        # 准备 API 调用所需的文档列表格式
        document_list = [{
            "display_name": file_name, # 显示名称
            "blob": blob_content # 二进制文件内容
        }]

        # 调用 DataSet 对象的 upload_documents 方法
        # 根据文档，此方法成功时不返回值，失败时抛出异常
        dataset_object.upload_documents(document_list=document_list)
        logging.info(f"已为数据集 '{dataset_object.name}' (ID: {dataset_object.id}) 启动文档 '{file_name}' 的上传。")
        
        # 上传成功后，需要获取文档ID以便后续操作
        # 使用 list_documents 方法根据文件名查找文档ID
        try:
            # 尝试获取最近上传的文档 - 根据API文档调整参数
            docs = dataset_object.list_documents(keywords=file_name)
            doc_id = ''
            
            if docs and len(docs) > 0:
                # 找到匹配的文档
                for doc in docs:
                    if hasattr(doc, 'name') and doc.name == file_name or hasattr(doc, 'display_name') and getattr(doc, 'display_name', '') == file_name:
                        doc_id = doc.id
                        break
                
                if doc_id:
                    logging.info(f"  文档 '{file_name}' 上传成功，获得文档ID: {doc_id}")
                else:
                    logging.warning(f"  文档 '{file_name}' 上传成功，但在数据集中未找到精确匹配的文档")
                    # 如果没有精确匹配，但有搜索结果，使用第一个结果
                    if hasattr(docs[0], 'id'):
                        doc_id = docs[0].id
                        logging.info(f"  使用最近上传的文档ID: {doc_id}")
                    else:
                        logging.warning(f"  无法从文档对象中获取ID")
            else:
                logging.warning(f"  文档 '{file_name}' 上传成功，但未能获取文档ID")
        except Exception as list_error:
            logging.error(f"  获取文档ID时出错: {list_error}")
            doc_id = ''
        
        return True, doc_id
    except Exception as e:
        logging.error(f"读取或上传文档 '{file_path}' 到数据集 '{dataset_object.name}' 失败: {e}")
        return False, ''

import hashlib

async def upload_documents(dataset_object: DataSet, folder_path: str, allowed_extensions: tuple = ('.txt', '.pdf', '.md', '.docx', '.doc'), config: dict = None):
    """Iterates through a folder and uploads valid documents to the specified dataset.

    Args:
        dataset_object: The DataSet object to upload documents to.
        folder_path: The absolute or relative path to the folder containing documents.
        allowed_extensions: A tuple of lowercase file extensions to upload.
        config: Configuration dictionary that may contain duplicate handling settings.

    Returns:
        A list of document IDs that were successfully uploaded.
    """
    if not dataset_object:
        logging.error("为上传操作提供了无效的 DataSet 对象。")
        return []

    # Resolve to absolute path
    absolute_folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(absolute_folder_path):
        logging.error(f"指定的文档文件夹路径无效或不存在: {absolute_folder_path}")
        return []

    # Get duplicate handling configuration
    doc_upload_config = config.get('document_upload', {}) if config else {}
    duplicate_handling = doc_upload_config.get('duplicate_handling', 'skip_name')  # Options: 'skip_name', 'skip_content', 'replace', 'allow'
    
    logging.info(f"开始从文件夹上传文档: {absolute_folder_path}")
    logging.info(f"允许的文件扩展名: {allowed_extensions}")
    logging.info(f"重复文件处理策略: {duplicate_handling}")

    uploaded_doc_ids = []
    skipped_count = 0
    replaced_count = 0
    failed_count = 0
    processed_count = 0

    # Get existing document information to handle duplicates
    existing_docs = {}
    existing_display_names = set()  # 存储显示名称，用于更准确的比较
    existing_content_hashes = set()
    
    # 我们不再预先获取所有文档，而是在处理每个文件时使用 keywords 参数查询重复
    # 这样可以避免只获取到默认 30 个文档的限制
    # 同时也更高效，因为我们只需要查询特定文件名的文档

    # Iterate through files in the specified folder
    for filename in os.listdir(absolute_folder_path):
        if filename.lower().endswith(allowed_extensions):
            file_path = os.path.join(absolute_folder_path, filename)
            if os.path.isfile(file_path):
                processed_count += 1
                
                # 使用 keywords 参数直接查询是否存在同名文件
                is_duplicate = False
                if duplicate_handling != 'allow':
                    try:
                        # 使用文件名作为 keywords 查询数据集中的文档
                        # 注意：这里不需要分页查询，因为我们只查找特定文件名的文档
                        # 即使数据集中有很多文档，同名文档应该不多
                        existing_docs_with_name = list_documents_in_dataset(dataset_object, keywords=filename, page_size=100)
                        is_duplicate = len(existing_docs_with_name) > 0
                        
                        if is_duplicate and duplicate_handling in ['skip_name', 'skip_content']:
                            logging.info(f"  跳过: 文档 '{filename}' 已存在于数据集中。")
                            skipped_count += 1
                            continue  # Skip to the next file
                        elif is_duplicate and duplicate_handling == 'replace':
                            # 如果需要替换，需要先删除现有文档
                            logging.info(f"  替换: 文档 '{filename}' 已存在，将被替换。")
                            replaced_count += 1
                            
                            # 实现文档替换逻辑：删除现有文档
                            try:
                                if existing_docs_with_name and len(existing_docs_with_name) > 0:
                                    # 获取第一个匹配的文档
                                    doc_to_delete = existing_docs_with_name[0]
                                    doc_id = None
                                    
                                    # 提取文档ID
                                    if isinstance(doc_to_delete, dict):
                                        doc_id = doc_to_delete.get('id')
                                    elif hasattr(doc_to_delete, 'id'):
                                        doc_id = doc_to_delete.id
                                    
                                    if doc_id:
                                        # 尝试删除文档
                                        try:
                                            # 如果 API 支持删除文档的操作
                                            # 注意：这里需要根据实际 API 调整
                                            doc_obj = Document(dataset_object, doc_id)
                                            doc_obj.delete()
                                            logging.info(f"    成功删除现有文档 ID: {doc_id}")
                                        except Exception as del_err:
                                            logging.error(f"    删除文档失败 ID: {doc_id}, 错误: {del_err}")
                            except Exception as replace_err:
                                logging.error(f"    替换文档时出错: {replace_err}")
                    except Exception as e:
                        logging.warning(f"  检查文档 '{filename}' 是否存在时出错: {e}")

                logging.info(f"  准备上传: '{filename}'")
                # Call the single document upload function from this module
                try:
                    # 调用上传函数，现在返回成功标志和文档ID
                    success, doc_id = upload_document_to_dataset(dataset_object, file_path)

                    if success:
                        logging.info(f"    成功上传: '{filename}'")
                        if doc_id:
                            # 将文档ID添加到已上传文档ID列表
                            uploaded_doc_ids.append(doc_id)
                            
                    else:
                        logging.warning(f"    上传失败: '{filename}'")
                        failed_count += 1
                except Exception as upload_error:
                    logging.error(f"    上传 '{filename}' 时发生异常: {upload_error}")
                    failed_count += 1
            # else: It's a directory or other non-file, ignore silently.
        # else: File extension not allowed, ignore silently.

    logging.info(f"文件夹扫描完成。处理文件数: {processed_count}，新上传: {len(uploaded_doc_ids)}，已存在跳过: {skipped_count}，替换: {replaced_count}，失败: {failed_count}。")
    
    # 如果有成功上传的文档，触发解析
    if uploaded_doc_ids:
        logging.info(f"开始解析新上传的 {len(uploaded_doc_ids)} 个文档...")
        # 使用改进的解析函数，单个处理每个文档并等待完成
        # 从配置中获取重试参数
        doc_upload_config = config.get('document_upload', {}) if config else {}
        max_retries = doc_upload_config.get('parse_retry_count', 10)
        retry_interval = doc_upload_config.get('parse_retry_interval', 5)
        
        # 调用异步解析函数，并等待所有文档解析完成
        parse_success = parse_documents_in_dataset(
            dataset_object, 
            uploaded_doc_ids,
            max_retries=max_retries,
            retry_interval=retry_interval
        )
        
        if parse_success:
            logging.info(f"所有文档解析已完成。")
        else:
            logging.error(f"文档解析过程中出现错误。")
    else:
        logging.info("没有新文档上传，跳过解析步骤。")
        
    return uploaded_doc_ids

# 注意：已删除 parse_single_document 函数，因为其功能已集成到改进的 parse_documents_in_dataset 函数中

def parse_documents_in_dataset(dataset_object: DataSet, doc_ids: list, max_retries: int = 10, retry_interval: int = 5) -> bool:
    """触发对数据集中指定文档 ID 列表的解析，并等待文档解析完成。
    
    逐个处理文档，每次只解析一个文档并等待其完成，以减轻嵌入模型的负担。
    
    Args:
        dataset_object: 数据集对象
        doc_ids: 要解析的文档ID列表（字符串）
        max_retries: 每个文档的最大轮询重试次数
        retry_interval: 轮询间隔（秒）
        
    Returns:
        bool: 文档解析是否成功完成
    """
    if not dataset_object:
        logging.error("为解析操作提供了无效的 DataSet 对象。")
        return False
    if not doc_ids or len(doc_ids) == 0:
        logging.warning("未提供需要解析的文档 ID。")
        return False

    import time
    
    # 确保所有文档 ID 都是字符串
    doc_ids = [str(doc_id) for doc_id in doc_ids if doc_id]
    if not doc_ids:
        logging.warning("处理后没有有效的文档 ID。")
        return False
    
    logging.info(f"将逐个处理数据集 '{dataset_object.name}' 中的 {len(doc_ids)} 个文档解析。")
    
    success_count = 0
    failed_count = 0
    

    # 逐个处理每个文档
    for doc_index, doc_id in enumerate(doc_ids):
        logging.info(f"\n[文档 {doc_index+1}/{len(doc_ids)}] 开始解析文档 ID: {doc_id}")

        # 1. 发起单个文档的解析请求
        try:
            # 在触发解析前等待短暂时间，给文档上传足够的时间完成
            time.sleep(1)  # 等待 1 秒再尝试异步解析
            dataset_object.async_parse_documents(document_ids=[doc_id])
            logging.info(f"  成功发送文档 ID: {doc_id} 的解析请求")
        except Exception as e:
            logging.error(f"  发起文档 ID: {doc_id} 的解析请求失败: {e}")
            failed_count += 1
            continue  # 跳过这个文档，处理下一个
    

     # 2. 轮询检查文档解析状态，直到完成或超时
        retry_count = 0
        failed_status_count = 0  # 解析状态为Failed的次数
        parse_completed = False
        parse_status = None
        
        while failed_status_count < max_retries and not parse_completed:
            retry_count += 1
            logging.info(f"  [查询 {retry_count}, 解析失败次数 {failed_status_count}/{max_retries}] 检查文档 ID: {doc_id} 的解析状态...")
            
            try:
                # 直接使用 ID 查询文档状态
                doc_info = list_documents_in_dataset(dataset_object, id=doc_id)
                
                if doc_info and len(doc_info) > 0:
                    doc = doc_info[0]  # 只应该有一个结果
                    # 检查解析状态
                    parse_status = doc.run
                    
                    if parse_status in ['success', 'DONE']:
                        # 解析成功完成
                        parse_completed = True
                        logging.info(f"  文档 ID: {doc_id} 解析成功完成")
                        success_count += 1
                        break  # 跳出轮询循环，处理下一个文档
                    elif parse_status in ['failed', 'error', 'FAIL']:
                        # 解析状态为失败，计入失败次数
                        failed_status_count += 1
                        logging.warning(f"  文档 ID: {doc_id} 解析失败，状态: {parse_status}，这是第 {failed_status_count} 次检测到失败状态")
                        
                        # 只有当达到最大失败次数时才真正视为失败并退出
                        if failed_status_count >= max_retries:
                            parse_completed = True
                            logging.warning(f"  文档 ID: {doc_id} 解析失败次数达到上限 {max_retries}，放弃处理")
                            failed_count += 1
                            break
                    else:
                        # 仍在解析中，不计入失败次数
                        logging.info(f"  文档 ID: {doc_id} 仍在解析中，状态: {doc.progress if doc.progress else '未知'}")
                else:
                    # 无法获取文档信息，但不计入失败次数
                    logging.warning(f"  无法获取文档 ID: {doc_id} 的状态信息")
            except Exception as e:
                # 查询出错，但不计入失败次数
                logging.error(f"  检查文档 ID: {doc_id} 状态时出错: {e}")
            
            # 如果还没有完成，等待一段时间后重试
            if not parse_completed:
                logging.info(f"  继续监控文档解析状态，等待 {retry_interval} 秒后重试...")
                time.sleep(retry_interval)
        
        # 检查解析结果
        if not parse_completed:
            logging.warning(f"  文档 ID: {doc_id} 解析状态检查完成，但未达到明确的成功或失败状态")
            failed_count += 1
    
    total_processed = success_count + failed_count
    logging.info(f"文档解析统计: 总数 {len(doc_ids)}, 处理 {total_processed}, 成功 {success_count}, 失败 {failed_count}")
    
    # 如果至少有一个文档解析成功，则返回 True
    return success_count > 0

# 注意：已删除未使用的 poll_document_parse_status 函数，因为它的功能已经被 parse_documents_in_dataset 函数集成

def list_documents_in_dataset(dataset_object: DataSet, keywords: str = None, page: int = 1, page_size: int = 100, id: str = None) -> list:
    """列出指定数据集对象中的文档。
    
    Args:
        dataset_object: 数据集对象
        keywords: 用于匹配文档标题的关键字
        page: 页码，默认为1
        page_size: 每页文档数量，默认为100（比API默认值30更大）
        id: 指定文档ID进行精确查询
    """
    if not dataset_object:
        logging.error("为列出文档操作提供了无效的 DataSet 对象。")
        return []
    try:
        # 调用 DataSet 对象的 list_documents 方法，根据文档使用正确的参数
        if id:
            # 如果提供了特定 ID，使用 ID 查询
            documents = dataset_object.list_documents(id=id)
        elif keywords:
            # 使用关键字查询，并传递分页参数
            documents = dataset_object.list_documents(keywords=keywords, page=page, page_size=page_size)
        else:
            # 获取所有文档，使用分页参数
            documents = dataset_object.list_documents(page=page, page_size=page_size)
            
        # 处理可能的 Response 对象
        if hasattr(documents, 'json') and callable(getattr(documents, 'json', None)):
            try:
                # 如果是 Response 对象，尝试提取其 JSON 内容
                response_data = documents.json()
                if isinstance(response_data, dict):
                    # 如果是字典，尝试获取数据字段
                    if 'data' in response_data and isinstance(response_data['data'], list):
                        documents = response_data['data']
                    elif 'documents' in response_data and isinstance(response_data['documents'], list):
                        documents = response_data['documents']
                    else:
                        # 如果没有数据字段，使用整个响应
                        documents = [response_data]
                elif isinstance(response_data, list):
                    # 如果直接是列表，使用该列表
                    documents = response_data
                else:
                    logging.warning(f"无法从 Response 对象提取文档列表，返回类型: {type(response_data)}")
                    documents = []
            except Exception as json_err:
                logging.error(f"尝试从 Response 对象提取 JSON 时出错: {json_err}")
                documents = []
        
        logging.info(f"从数据集 '{dataset_object.name}' 检索到 {len(documents) if documents else 0} 个文档。")
        return documents if documents else []
    except Exception as e:
        logging.error(f"列出数据集 '{dataset_object.name}' 中的文档失败: {e}")
        return []

# 注意: 根据用户反馈，RAGFlow API 不支持更新文档启用/禁用状态
# 如需按关键词区分文档，请使用多个数据集
