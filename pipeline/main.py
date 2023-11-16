# Define your stages
from config import PREPROCESSED_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR
from data.models import Project
from pipeline.pipeline import PipelineStage, Pipeline


class Stage1(PipelineStage):
    def process(self):
        if self.input_data is not None:
            self.output_data = self.input_data.copy()
            self.output_data['Stage1_Column'] = self.output_data['Column_A'] * 2


class Stage2(PipelineStage):
    def process(self):
        if self.input_data is not None:
            self.output_data = self.input_data.copy()
            self.output_data['Stage2_Column'] = self.output_data['Column_B'] + 5


if __name__ == "__main__":
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )

    stages = [
        PipelineStage("")
    ]
    pipeline = Pipeline(stages)
    pipeline.run()
