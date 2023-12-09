from config import PREPROCESSED_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR, METHOD_LEVEL_DATA_SAVE_DIR
from data.models import Project
from pipeline.models import PipelineStage, Pipeline


class Stage1(PipelineStage):

    def process(self):
        self.stage_data['b'] = 'b'


class Stage2(PipelineStage):

    def process(self):
        self.stage_data['a'] = 'a'


if __name__ == "__main__":
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
        method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
    )

    stages = [
        Stage1(),
        Stage2()
    ]
    pipeline = Pipeline(stages)
    print(stages)
    print(pipeline)
    print(pipeline.run())
