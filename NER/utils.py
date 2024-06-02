import logging
import os

def configure_logging(log_file_path, mod,default_level=logging.INFO,):
    """
    配置日志系统，将日志输出到指定的文件路径，并在控制台输出相同内容。
    日志文件将被覆盖，而不是追加。

    :param log_file_path: 日志文件的保存路径
    :param default_level: 日志的默认级别
    """
    # 创建一个日志记录器实例
    logger = logging.getLogger()
    logger.setLevel(default_level)

    # 确保日志目录存在
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建一个文件处理器，并设置为写入模式，覆盖现有文件
    file_handler = logging.FileHandler(log_file_path, mode=mod)
    file_handler.setLevel(default_level)

    # 创建一个流处理器（控制台）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(default_level)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 设置处理器的格式
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # 避免日志信息重复打印
    logger.propagate = False





