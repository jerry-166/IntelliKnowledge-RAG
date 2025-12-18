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

1. 在哪里logger比较好呢？是每一个python文件都要吗？logger打印的信息怎么看？可以写入文件吗？
1.1 关键业务逻辑点：在重要的函数入口和出口记录日志
    异常处理位置：在 try-except 块中记录错误信息
    系统初始化：记录组件初始化成功或失败的状态
    重要操作结果：记录关键操作的执行结果（成功/失败
1.2 建议每个模块都有日志：使用 logging.getLogger(__name__) 创建模块级日志记录器
    按需记录：不是每行代码都要记录，而是记录有价值的信息
    不同级别：合理使用 debug、info、warning、error 等不同日志级别
1.3 查看方式：
    控制台输出（开发调试时）
    日志文件查看（生产环境）
"""

# 创建格式化器
formatter = logging.Formatter('时间:%(asctime)s | 模块:%(name)s | 级别:%(levelname)s | 内容:%(message)s')

# 配置文件处理器
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# 配置控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# 获取根日志记录器并添加处理器
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
