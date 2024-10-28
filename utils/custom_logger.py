import logging
import os
import datetime

class CustomLogger:
    def __init__(self, log_dir_base="../log", log_file_name="rag_training.log"):
        current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
        self.log_dir = os.path.join(log_dir_base, current_time)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file_path = os.path.join(self.log_dir, log_file_name)
        self.setup_logging()

    def setup_logging(self):
        # Set up logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(self.log_file_path),
                logging.StreamHandler()
            ]
        )

    def get_log_dir(self):
        return self.log_dir

    def get_logger(self):
        return logging.getLogger()  # Return the standard logger instance