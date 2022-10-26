"""The log implementation.
"""
import os
import warnings
import logging
import multiprocessing as mp


def level_check(level=3):
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


def add_handler(handler, logger, level=3, format='%(name)s - %(message)s'):
    """The operation for add log handler.

    Args:
        handler (logging.StreamHandler|logging.FileHandler): The log handler.
        logger (logging.RootLogger|logging.Logger): The logger.
        level (int, optional): The log level. Defaults to 3.
        format (str, optional): The log format. Defaults to '%(name)s - %(message)s'.
    """
    # check the level
    level_check(level)

    # get formatter
    formatter = logging.Formatter(format)

    # set the handler and add
    handler.setLevel(level*10)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(log_level=3, stream_level=3, file_level=3, log_dir=None, format='%(name)s - %(message)s', distributed=False):
    """The operation for get logger.

    Args:
        log_level (int, optional): The log level. Defaults to 3.
        stream_level (int, optional): The stream handler log level. Defaults to 3.
        file_level (int, optional): The file handler log level. Defaults to 3.
        log_dir (str, optional): The log file path. Defaults to None.
        format (str, optional): The log format. Defaults to '%(name)s - %(message)s'.

    Returns:
        logging.RootLogger|logging.Logger: The logger.
    """
    # get logger
    logger = logging.getLogger()
    # if distributed:
    #     logger = mp.get_logger()
    # else:
    #     logger = logging.getLogger()

    # set log level
    level_check(log_level)
    logger.setLevel(log_level*10)

    # set stream handler
    stream_handler = logging.StreamHandler()
    add_handler(handler=stream_handler, logger=logger,
                level=stream_level, format=format)

    # set file handler
    if log_dir is not None:
        file_handler = logging.FileHandler(log_dir)
        add_handler(handler=file_handler, logger=logger, level=file_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    return logger
