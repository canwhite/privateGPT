#!/usr/bin/env python3 # 使用python3作为解释器
import os # 导入os模块，用于操作系统相关的功能
import glob # 导入glob模块，用于文件名匹配
from typing import List # 导入List类型注解，用于指定列表类型
from dotenv import load_dotenv # 导入load_dotenv函数，用于加载环境变量
from multiprocessing import Pool # 导入Pool类，用于创建进程池
from tqdm import tqdm # 导入tqdm模块，用于显示进度条
# 从langchain.document_loaders模块导入以下类，用于加载不同格式的文档
from langchain.document_loaders import ( 
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter # 从langchain.text_splitter模块导入RecursiveCharacterTextSplitter类，用于将文本拆分为固定大小的片段
from langchain.vectorstores import Chroma # 从langchain.vectorstores模块导入Chroma类，用于创建向量存储
from langchain.embeddings import HuggingFaceEmbeddings # 从langchain.embeddings模块导入HuggingFaceEmbeddings类，用于创建句子向量
from langchain.docstore.document import Document # 从langchain.docstore.document模块导入Document类，用于表示文档对象
from constants import CHROMA_SETTINGS # 从constants模块导入CHROMA_SETTINGS变量，用于配置Chroma数据库


load_dotenv() # 调用load_dotenv函数，从.env文件中读取环境变量


# Load environment variables
# 加载环境变量
persist_directory = os.environ.get('PERSIST_DIRECTORY') # 获取PERSIST_DIRECTORY环境变量的值，赋给persist_directory变量
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents') # 获取SOURCE_DIRECTORY环境变量的值，如果没有则使用'source_documents'作为默认值，赋给source_directory变量
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME') # 获取EMBEDDINGS_MODEL_NAME环境变量的值，赋给embeddings_model_name变量
chunk_size = 500 # 定义chunk_size变量为500，表示每个文本片段的最大长度（单位为token）
chunk_overlap = 50 # 定义chunk_overlap变量为50，表示每个文本片段之间的重叠长度（单位为token）


# 自定义文档加载器
class MyElmLoader(UnstructuredEmailLoader): # 定义一个类MyElmLoader，继承自UnstructuredEmailLoader类
    """Wrapper to fallback to text/plain when default does not work""" # 类的文档字符串，说明这个类的作用是在默认的方式不起作用时，使用text/plain作为备选

    def load(self) -> List[Document]: # 定义一个方法load，返回一个Document对象的列表
        """Wrapper adding fallback for elm without html""" # 方法的文档字符串，说明这个方法的作用是为没有html内容的elm文件添加备选
        try: # 尝试执行以下代码
            try: # 尝试执行以下代码
                doc = UnstructuredEmailLoader.load(self) # 调用父类的load方法，获取文档对象
            except ValueError as e: # 如果发生ValueError异常，将异常对象赋给e变量
                if 'text/html content not found in email' in str(e): # 如果异常信息中包含'text/html content not found in email'这个字符串
                    # Try plain text
                    # 尝试使用纯文本
                    self.unstructured_kwargs["content_source"]="text/plain" # 将self.unstructured_kwargs字典中的"content_source"键的值设为"text/plain"
                    doc = UnstructuredEmailLoader.load(self) # 调用父类的load方法，获取文档对象
                else: # 否则
                    raise # 抛出异常
        except Exception as e: # 如果发生任何其他异常，将异常对象赋给e变量
            # Add file_path to exception message
            # 在异常信息中添加文件路径
            raise type(e)(f"{self.file_path}: {e}") from e # 抛出一个新的异常，类型和原来一样，但是信息中包含文件路径和原来的信息

        return doc  # 返回文档对象



# 将文件扩展名映射到文档加载器和它们的参数
LOADER_MAPPING = { # 定义一个字典变量LOADER_MAPPING
    ".csv": (CSVLoader, {}), # 将".csv"扩展名映射到CSVLoader类和一个空字典
    # ".docx": (Docx2txtLoader, {}), # 将".docx"扩展名映射到Docx2txtLoader类和一个空字典（这一行被注释掉了）
    ".doc": (UnstructuredWordDocumentLoader, {}), # 将".doc"扩展名映射到UnstructuredWordDocumentLoader类和一个空字典
    ".docx": (UnstructuredWordDocumentLoader, {}), # 将".docx"扩展名映射到UnstructuredWordDocumentLoader类和一个空字典
    ".enex": (EverNoteLoader, {}), # 将".enex"扩展名映射到EverNoteLoader类和一个空字典
    ".eml": (MyElmLoader, {}), # 将".eml"扩展名映射到MyElmLoader类和一个空字典
    ".epub": (UnstructuredEPubLoader, {}), # 将".epub"扩展名映射到UnstructuredEPubLoader类和一个空字典
    ".html": (UnstructuredHTMLLoader, {}), # 将".html"扩展名映射到UnstructuredHTMLLoader类和一个空字典
    ".md": (UnstructuredMarkdownLoader, {}), # 将".md"扩展名映射到UnstructuredMarkdownLoader类和一个空字典
    ".odt": (UnstructuredODTLoader, {}), # 将".odt"扩展名映射到UnstructuredODTLoader类和一个空字典
    ".pdf": (PDFMinerLoader, {}), # 将".pdf"扩展名映射到PDFMinerLoader类和一个空字典
    ".ppt": (UnstructuredPowerPointLoader, {}), # 将".ppt"扩展名映射到UnstructuredPowerPointLoader类和一个空字典
    ".pptx": (UnstructuredPowerPointLoader, {}), # 将".pptx"扩展名映射到UnstructuredPowerPointLoader类和一个空字典
    ".txt": (TextLoader, {"encoding": "utf8"}), # 将".txt"扩展名映射到TextLoader类和一个包含"encoding":"utf8"键值对的字典
    # Add more mappings for other file extensions and loaders as needed
    # 根据需要添加更多的文件扩展名和加载器的映射
}


# 定义一个函数load_single_document，接受一个字符串类型的参数file_path，返回一个Document类型的对象
# 该方法是下边load_document使用的方法
def load_single_document(file_path: str) -> Document: 
     # 从file_path中获取文件扩展名，并在前面加上"."，赋给ext变量
    ext = "." + file_path.rsplit(".", 1)[-1]
    # 如果ext在LOADER_MAPPING字典中
    if ext in LOADER_MAPPING: 
        # 从LOADER_MAPPING字典中获取对应的加载器类和参数，赋给loader_class和loader_args变量
        loader_class, loader_args = LOADER_MAPPING[ext] 
         # 使用loader_class和loader_args创建一个加载器对象，赋给loader变量
        loader = loader_class(file_path, **loader_args)
         # 调用加载器的load方法，获取文档对象列表，并返回第一个元素
        return loader.load()[0]
    # 如果ext不在LOADER_MAPPING字典中，抛出一个ValueError异常，提示不支持的文件扩展名
    raise ValueError(f"Unsupported file extension '{ext}'") 



# 定义一个函数load_documents，接受一个字符串类型的参数source_dir和一个字符串列表类型的参数ignored_files（默认为空列表），返回一个Document对象的列表
# 函数的作用是从源文档目录中加载所有文档，忽略指定的文件
# 这个方法下边的process_document会使用
def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    # 定义一个空列表变量all_files，用于存储所有文件的路径
    all_files = []
    # 遍历LOADER_MAPPING字典中的键（即文件扩展名）
    for ext in LOADER_MAPPING:
        # 将以下列表添加到all_files列表中
        all_files.extend(
            # 使用glob模块的glob函数，根据source_dir和扩展名拼接出一个通配符路径，然后递归地匹配所有符合条件的文件路径，返回一个列表
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    # 使用列表推导式，从all_files列表中筛选出不在ignored_files列表中的文件路径，赋给filtered_files变量
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    # 使用Pool类创建一个进程池对象，进程数为os模块的cpu_count函数返回的值，赋给pool变量，并使用with语句管理其生命周期
    with Pool(processes=os.cpu_count()) as pool:
        # 定义一个空列表变量results，用于存储文档对象
        results = []
        # 使用tqdm模块的tqdm函数创建一个进度条对象，总数为filtered_files列表的长度，
        # 描述为'Loading new documents'，列数为80，
        # 赋给pbar变量，并使用with语句管理其生命周期
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            # 使用pool对象的imap_unordered方法，对filtered_files列表中的每个元素（即文件路径）调用load_single_document函数，并返回一个迭代器；使用enumerate函数对迭代器进行编号；遍历编号和文档对象
            for i, doc in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                # 将文档对象添加到results列表中
                results.append(doc)
                # 调用pbar对象的update方法，更新进度条
                pbar.update()

    # 返回results列表
    return results



# 定义一个函数process_documents，
# 接受一个字符串列表类型的参数ignored_files（默认为空列表），
# 返回一个Document对象的列表
def process_documents(ignored_files: List[str] = []) -> List[Document]:
    # 这个函数的作用是加载文档并拆分成片段
    # 打印一条信息，显示正在从source_directory变量指定的目录中加载文档
    print(f"Loading documents from {source_directory}")
    # 调用load_documents函数，传入source_directory和ignored_files参数，获取文档对象列表，赋给documents变量
    documents = load_documents(source_directory, ignored_files)
    # 如果documents列表为空
    if not documents:
        # 打印一条信息，显示没有新的文档可以加载
        print("No new documents to load")
        # 退出程序
        exit(0)

    
    # 打印一条信息，显示从source_directory变量指定的目录中加载了多少个新的文档
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    # 使用RecursiveCharacterTextSplitter类创建一个文本拆分器对象，传入chunk_size和chunk_overlap参数，赋给text_splitter变量
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # 调用text_splitter对象的split_documents方法，传入documents参数，获取拆分后的文本对象列表，赋给texts变量
    texts = text_splitter.split_documents(documents)
    # 打印一条信息，显示拆分成了多少个文本片段，以及每个片段的最大长度（单位为token）
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    # 返回texts列表
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
