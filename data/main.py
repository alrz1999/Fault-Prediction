from config import PREPROCESSED_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR
from data.models import Project
from data.utils import LineLevelDatasetHelper
from pipeline.data.line_level import LineLevelDatasetLoaderStage, LineLevelTokenizerStage
from pipeline.pipeline import Pipeline


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )

    stages = [
        LineLevelDatasetLoaderStage(project.get_train_release().get_line_level_dataset_path()),
        LineLevelTokenizerStage()
    ]
    lines_tokens = Pipeline(stages).run()
    print(lines_tokens[0])


if __name__ == '__main__':
    main()
