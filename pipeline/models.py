import logging
from enum import Enum
import colorlog

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
        'INFO': 'green',
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


class StageData(dict):
    class Keys(Enum):
        LINE_LEVEL_DF = 'line_level_df'

        EMBEDDING_MODEL = 'embedding_model'
        EMBEDDING = 'embedding'

        FILE_LEVEL_DF = 'file_level_df'

        CLASSIFIER_MODEL = 'classifier_model'
        PREDICTION_RESULT_DF = 'prediction_result_df'

        INDEX_TO_VEC_MATRIX = 'embedding_matrix'

    def combine_with(self, another_stage_data: dict):
        if another_stage_data is None:
            return

        for key, val in another_stage_data.items():
            self[key] = val


class PipelineStage:
    def __init__(self, stage_data=None, perform_export=False):
        self.name = type(self).__name__
        self.stage_data: StageData = StageData() if stage_data is None else stage_data
        self.perform_export = perform_export
        self.result = None

    def _export_result(self):
        if self.perform_export:
            logger.info(f" {self.name} export_result started ".center(40, "*"))
            self.export_result()
            logger.info(f" {self.name} export_result finished ".center(40, "*"))

    def export_result(self):
        raise NotImplementedError()

    def _process(self):
        logger.info(f" {self.name} process started ".center(40, "*"))
        self.process()
        logger.info(f" {self.name} process finished ".center(40, "*"))

    def process(self):
        raise NotImplementedError()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Pipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self, initial_data: StageData = None) -> StageData:
        data = initial_data
        for stage in self.stages:
            stage.stage_data.combine_with(data)
            stage._process()
            data = stage.stage_data

            if stage.perform_export:
                stage._export_result()

        return data

    def __str__(self):
        return f"Pipeline[{str.join(', ', [str(stage) for stage in self.stages])}]"

    def __repr__(self):
        return str(self)
