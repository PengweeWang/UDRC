import logging

class Logger:
    def __init__(self, log_file="app.log", level=logging.DEBUG):
        """
        初始化日志记录器
        :param log_file: 日志文件名
        :param level: 日志级别
        """
        # 创建一个logger对象
        self.logger = logging.getLogger("ConsoleLogger")
        self.logger.setLevel(level)

        # 创建一个文件处理器，用于写入日志文件
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)

        # 创建一个控制台处理器，用于输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # 定义日志格式：时间 - 日志级别 - 消息
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将处理器添加到logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        """记录调试信息"""
        self.logger.debug(message)

    def info(self, message):
        """记录普通信息"""
        self.logger.info(message)

    def warning(self, message):
        """记录警告信息"""
        self.logger.warning(message)

    def error(self, message):
        """记录错误信息"""
        self.logger.error(message)

    def critical(self, message):
        """记录严重错误信息"""
        self.logger.critical(message)




# 示例用法
if __name__ == "__main__":
    logger = Logger(log_file="example.log")

    logger.debug("这是一个调试信息")
    logger.info("这是一个普通信息")
    logger.warning("这是一个警告信息")
    logger.error("这是一个错误信息")
    logger.critical("这是一个严重错误信息")
