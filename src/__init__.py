# 此文件使得 src 目录成为一个 Python 包。
# 可以留空，或者用于包级别的初始化代码。

# Import functions to make them available directly from the 'src' package
from .config_loader import load_config
from .ragflow_ops import (
    upload_documents,
    get_dataset_by_name, # Correct function name
    # Add other ragflow_ops functions here if needed elsewhere
)
from .evaluation import run_grid_search, retrieve_chunks # Correct import for retrieve_chunks
from .generation import generate_answer_from_chunks

# Define what is exposed when 'from src import *' is used (optional but good practice)
__all__ = [
    'load_config',
    'upload_documents',
    'get_dataset_by_name', # Correct function name
    'update_document_status_by_keyword',
    'run_grid_search', # Correct function name
    'generate_answer_from_chunks', # Correct generation function
    'retrieve_chunks',             # Add retrieval function
]
