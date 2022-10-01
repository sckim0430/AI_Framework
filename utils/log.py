"""The log implementation.
"""
import os
import warnings
import logging


def level_check(level=1):
    """The operation for check log level.

    Args:
        level (int, optional): The log level. Defaults to 3.

    Returns:
        int: The log level
    """
    if level > 5:
        warnings.warn(
            'If log level larger than 5, you can not log anything. Set the log level 5.')
        level = 5

    return level


def add_handler(handler, logger, level=1, format='%(name)s - %(message)s'):
    """The operation for add log handler.

    Args:
        handler (logging.StreamHandler|logging.FileHandler): The log handler.
        logger (logging.RootLogger): The logger.
        level (int, optional): The log level. Defaults to 3.
        format (str, optional): The log format. Defaults to '%(name)s - %(message)s'.
    """
    #check the level
    level_check(level)

    #get formatter
    formatter = logging.Formatter(format)

    #set the handler and add
    handler.setLevel(level*10)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(log_level=1, stream_level=1, file_level=1, log_dir=None, format='%(name)s - %(message)s'):
    """The operation for get logger.

    Args:
        log_level (int, optional): The log level. Defaults to 3.
        stream_level (int, optional): The stream handler log level. Defaults to 3.
        file_level (int, optional): The file handler log level. Defaults to 3.
        log_dir (str, optional): The log file path. Defaults to None.
        format (str, optional): The log format. Defaults to '%(name)s - %(message)s'.
    
    Returns:
        logging.RootLogger: The logger.
    """
    #get logger
    logger = logging.getLogger()

    #set log level
    level_check(log_level)
    logger.setLevel(log_level*10)

    #set stream handler
    stream_handler = logging.StreamHandler()
    add_handler(stream_handler, stream_level, format, logger)

    #set file handler
    if log_dir is not None and os.path.isfile(log_dir):
        file_handler = logging.FileHandler(log_dir)
        add_handler(file_handler, file_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', logger=logger)

    return logger
