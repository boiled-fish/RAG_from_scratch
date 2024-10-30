import logging
import os
import datetime
import sys

# Enhanced CustomLogger with timestamp and better formatting
class CustomLogger:
    def __init__(self, log_dir_base, logger_name):
        self.log_dir_base = log_dir_base
        self.logger_name = logger_name
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        if not os.path.exists(log_dir_base):
            os.makedirs(log_dir_base)

        self.log_dir = f"{log_dir_base}/{logger_name}"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Prevent adding multiple handlers if logger is already configured
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

            # Create file handler
            fh = logging.FileHandler(f"{self.log_dir}/{logger_name}.log")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Create console handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    def get_logger(self):
        return self.logger

    def get_log_dir(self):
        return self.log_dir