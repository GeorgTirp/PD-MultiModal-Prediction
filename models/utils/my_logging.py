import logging
from datetime import datetime


class Logging:
    def __init__(self, log_dir='logs'):
        """
        Set up logging for both console and file output.
        """
        # Create a logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)  # Set the log level to DEBUG or any level you'd prefer

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set level for console output (INFO or higher)

        # Create file handler
        log_filename = datetime.now().strftime(f"{log_dir}/pipeline_%Y-%m-%d_%H-%M-%S.log")
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)  # Set level for file output (DEBUG or higher)

        # Create log formatters
        time_format = '%Y-%m-%d %H:%M:%S'
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt=time_format)
        # Modify level names
        logging.addLevelName(logging.DEBUG, "🐛   DEBUG")
        logging.addLevelName(logging.INFO, "ℹ️    INFO")
        logging.addLevelName(logging.WARNING, "⚠️ WARNING")
        logging.addLevelName(logging.ERROR, "❌   ERROR")
        logging.addLevelName(logging.CRITICAL, "❌❌❌ CRITICAL")

        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt=time_format)

        # Add formatters to handlers
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)

        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        """
        Retrieve the configured logger.
        """
        return self.logger