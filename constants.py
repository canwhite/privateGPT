import os # 导入os模块，用于操作系统相关的功能
from dotenv import load_dotenv # 导入load_dotenv函数，用于加载环境变量
from chromadb.config import Settings # 导入Settings类，用于配置Chroma数据库

load_dotenv() # 调用load_dotenv函数，从.env文件中读取环境变量

# Define the folder for storing database
# 定义存储数据库的文件夹，这个是在.env中定义好的
PERSIST_DIRECTORY = os.environ.get('PERSIST_DIRECTORY')

# Define the Chroma settings
# 定义Chroma数据库的设置
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet', # 指定Chroma数据库的实现方式为duckdb+parquet
        persist_directory=PERSIST_DIRECTORY, # 指定Chroma数据库的持久化目录为PERSIST_DIRECTORY
        anonymized_telemetry=False # 关闭匿名化的遥测功能
)