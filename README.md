# AutoRAG 知识库自动化评分系统

本项目通过RAGFlow API实现知识库的创建、管理和查询，并能够自动对文档内容进行评分。

## 功能特点

* 从`config.yaml`加载配置信息
* 初始化RAGFlow客户端
* 创建/管理RAGFlow数据集（知识库）
* 上传本地文档到数据集
* 解析上传的文档
* 根据提示词对文档内容进行评分
* 将评分结果和引用数据保存到Excel文件
* 支持批量处理多个城市和评分变量

## 项目结构

```
/
├── config.yaml             # 配置文件
├── requirements.txt        # Python依赖
├── README.md               # 本说明文件
├── main.py                 # 主执行脚本（单次测试）
├── batch_run.py            # 批量处理脚本
├── 得分变量提示词.xlsx      # 评分变量和提示词配置
├── docs/                   # 文档目录（按城市分类）
│   ├── 北京/               # 北京市相关文档
│   ├── 上海/               # 上海市相关文档
│   └── ...                 # 其他城市文档
└── src/                    # 源代码目录
    ├── __init__.py         # 标记'src'为Python包
    ├── config_loader.py    # 配置加载逻辑
    ├── ragflow_client.py   # RAGFlow客户端初始化
    ├── ragflow_ops.py      # RAGFlow数据集和文档操作
    ├── utils.py            # 工具函数（JSON解析、Excel保存等）
    └── QAandLog.py         # 问答和日志记录功能
```

## 使用方法

### 环境准备

1. **创建Python虚拟环境**
   ```bash
   conda create -n ragflow-env python=3.10
   conda activate ragflow-env
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 配置修改

1. **修改config.yaml中的RAGFlow服务器地址**
   
   将`config.yaml`中的`ragflow.base_url`修改为服务器IP地址：
   ```yaml
   ragflow:
     api_key: "ragflow-E3NzM3YWI2MGQ1ZTExZjA4MmM2MDI0Mm"
     base_url: "http://10.61.77.156:80" # 修改为服务器IP地址
   ```

### 单次测试

1. **修改config.yaml中的文档路径和查询**
   ```yaml
   document_upload:
     folder_path: "./docs/北京"  # 指定要上传的文档目录
   
   qa_example:
     score_variable: "非化石能源发展和应用的规划目标"  # 指定得分变量
     queries:  # 指定查询提示词
       - "文件中如果有阐述...获得20分。"
   ```

2. **运行main.py进行测试**
   ```bash
   python main.py
   ```

3. **查看结果**
   
   运行结束后，结果将保存在`outputs/运行记录.xlsx`文件中，包含两个sheet：
   - `结果_得分变量名`：包含得分结果
   - `引用数据`：包含引用的文档块
   
   两个sheet中的数据通过`timestamp`时间戳列进行关联，方便手动查看每个得分点对应的引用块是否合理。相同时间戳的记录表示它们来自同一次查询处理。

### 批量测试

1. **准备文档**
   
   在`docs`目录下创建城市子目录（如`北京`、`上海`等），并将相应的文档放入对应目录。

2. **准备评分变量和提示词**
   
   编辑`得分变量提示词.xlsx`文件，包含两列：
   - `score_variable`：得分变量名称
   - `queries`：对应的查询提示词

3. **运行批量处理脚本**
   ```bash
   python batch_run.py
   ```

4. **批处理工作原理**
   
   `batch_run.py`会：
   - 自动扫描`docs`目录下的所有子文件夹
   - 读取`得分变量提示词.xlsx`中的所有评分变量和提示词
   - 为每个城市和评分变量组合生成配置
   - 依次修改`config.yaml`文件中的关键配置并运行`main.py`
   - 所有结果将保存在同一个Excel文件中，可通过时间戳列区分不同批次的结果

## 注意事项

1. 确保RAGFlow服务器已启动并可访问
2. Excel文件中的提示词格式需要正确，包含得分点、概念和分值信息
3. 批量处理可能需要较长时间，请耐心等待
4. 如遇到JSON解析错误，可能是提示词格式有问题，请检查并修正
5. 所有结果都会保存在`outputs/运行记录.xlsx`文件中，可以按时间戳区分不同批次的结果

## 未来待增加功能

1. **文件元数据更新**
   - 根据文件的时间信息为知识库文件更新元数据
   - 参考RAGFlow官方讨论：https://github.com/infiniflow/ragflow/issues/6603
   - 这将允许按时间过滤文档，更好地管理文档版本



