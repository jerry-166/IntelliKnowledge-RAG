"""
日志记录配置
"""
import logging

"""
常用格式化选项
%(asctime)s - 时间戳
%(name)s - 记录器名称 - 哪个模块
%(levelname)s - 日志级别名称
%(message)s - 日志消息
%(filename)s - 文件名
%(lineno)d - 行号
%(funcName)s - 函数名

在应用程序入口 import python_services.core.logger_config  # 导入日志配置，
可以将配置传递到它所调用的所有文件中
"""

# 创建格式化器
formatter = logging.Formatter('时间:%(asctime)s | 模块:%(name)s | 级别:%(levelname)s | 内容:%(message)s')

# 配置文件处理器
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# 配置控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# 获取根日志记录器并添加处理器
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
