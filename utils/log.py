import logging


class Logger(object):
    """这部分代码实现了一个简单的日志记录器（Logger）。以下是对该代码的简单总结：
    1. **输入：**
       - `log_file_name`: 指定日志文件的名称。
       - `log_level`: 指定日志的记录级别。
       - `logger_name`: 指定日志记录器的名称。
    2. **功能：**
       - 创建一个日志记录器 (`self.__logger`)。
       - 设置日志记录器的级别为指定的 `log_level`。
       - 创建一个文件处理器 (`file_handler`)，用于将日志记录到文件。
       - 创建一个控制台处理器 (`console_handler`)，用于将日志输出到控制台。
       - 定义处理器的输出格式，包括时间、文件名、行号和消息。
       - 将文件处理器和控制台处理器添加到日志记录器中。
    3. **输出：**
       - 通过 `get_log` 方法返回创建的日志记录器对象。
    简而言之，该代码创建了一个可配置的日志记录器，可以同时将日志信息输出到文件和控制台。通过提供文件名、日志级别和记录器名称等参数，用户可以方便地使用这个日志记录器来记录程序运行过程中的信息。"""
    def __init__(self, log_file_name, log_level, logger_name):
        # firstly, create a logger
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        # secondly, create a handler
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        # thirdly, define the output form of handler
        formatter = logging.Formatter(
            "[%(asctime)s]-[%(filename)s line:%(lineno)d]:%(message)s "
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        # finally, add the Hander to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger
