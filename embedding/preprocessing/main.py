from config import PREPROCESSED_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR
from data.models import *
from pipeline.datas.file_level import FileLevelDatasetImporterStage, FileLevelTokenizerStage
from pipeline.models import Pipeline, StageData
from token_extraction import *


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )
    tokenizer = CFGTokenExtractor()

    stages = [
        FileLevelDatasetImporterStage(project.get_train_release().get_file_level_dataset_path()),
        FileLevelTokenizerStage(tokenizer)
    ]

    pipeline_data = Pipeline(stages).run()
    files_tokens = pipeline_data[StageData.Keys.FILE_LEVEL_TOKENS]
    print(files_tokens[0])


if __name__ == '__main__':
    main()
