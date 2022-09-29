"""The log implementation.
"""
import os
import warnings
import logging


class LogManager():
    """The log manager.
    """
    def __init__(self, log_level=3, stream_level=3, file_level=3, log_dir=None, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        """The initalization.

        Args:
            log_level (int, optional): The log level. Defaults to 3.
            stream_level (int, optional): The stream handler log level. Defaults to 3.
            file_level (int, optional): The file handler log level. Defaults to 3.
            log_dir (_type_, optional): The log file path. Defaults to None.
            format (str, optional): The format of log. Defaults to '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.
        """
        self.logger = logging.getLogger()
        self.set_level(log_level)
        self.formatter = self.set_format(format)
        self.add_stream_handler(stream_level)
        self.add_file_handler(file_level,log_dir)

    def set_level(self, level=3):
        """The operation for set log level.

        Args:
            level (int, optional): The log level. Defaults to 3.
        """
        if level > 5:
            warnings.warn(
                'If log level larger than 5, you can not log anything.')

        self.logger.setLevel(level*10)

    def set_format(self, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        """The operation for set format.

        Args:
            format (str, optional): The format of log. Defaults to '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.

        Returns:
            Formatter: The log formatter.
        """
        return logging.Formatter(format)

    def add_stream_handler(self, level=3):
        """The operation for add stream handler.

        Args:
            level (int, optional): The stream handler log level. Defaults to 3.
        """
        stream_handler = logging.StreamHandler()

        if level > 5:
            warnings.warn(
                'If log level larger than 5, you can not stream log anything.')

        stream_handler.setLevel(level*10)
        stream_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)

    def add_file_handler(self, level=3, log_dir=None):
        """The operation for add file handler.

        Args:
            level (int, optional): The file handler log level. Defaults to 3.
            log_dir (_type_, optional): The log file path. Defaults to None.
        """
        if log_dir is not None and os.path.isfile(log_dir):
            file_handler = logging.FileHandler(log_dir)

            if level > 5:
                warnings.warn(
                    'If log level larger than 5, you can not file log anything.')

            file_handler.setLevel(level*10)
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)
