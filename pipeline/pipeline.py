import functools
import logging

import colorlog
import pandas as pd

# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.INFO)

# Create a formatter
formatter = colorlog.ColoredFormatter(
    "%(white)s%(asctime)s - %(log_color)s%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'reset',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)

# Create a StreamHandler and set the formatter
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Add the StreamHandler to the logger
logger.addHandler(stream_handler)


def log_method_execution(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        logger.info(f"{self.name} {method.__name__} started")
        result = method(self, *args, **kwargs)
        logger.info(f"{self.name} {method.__name__} finished")
        return result

    return wrapper


class PipelineStage:
    def __init__(self, input_data=None, import_data=False, export_data=False, file_path=None):
        self.name = type(self).__name__
        self.input_data = input_data
        self.output_data = None
        self.import_data = import_data
        self.export_data = export_data
        self.file_path = file_path

    @log_method_execution
    def import_output(self):
        if not self.import_data:
            return

        self.output_data = pd.read_csv(self.file_path)
        return self.output_data

    @log_method_execution
    def export_output(self):
        if not self.export_data:
            return

        if self.output_data is None:
            raise Exception("Output data is not ready for exporting")

        self.output_data.to_csv(self.file_path, index=False)

    @log_method_execution
    def process(self):
        pass


class Pipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self):
        data = None
        for stage in self.stages:
            if stage.import_data:
                data = stage.import_output()
                continue

            stage.input_data = data
            stage.process()
            data = stage.output_data

            if stage.export_data:
                stage.export_output()

        return data
