from config import LINE_LEVEL_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR, METHOD_LEVEL_DATA_SAVE_DIR
from data.models import *
from pipeline.datas.file_level import FileLevelDatasetImporterStage
from pipeline.models import Pipeline, StageData
from token_extraction import *


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
        method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
    )
    tokenizer = CFGTokenExtractor()

    stages = [
        FileLevelDatasetImporterStage(project.get_train_release()),
    ]

    pipeline_data = Pipeline(stages).run()[StageData.Keys.FILE_LEVEL_DF.value]
    files_tokens = tokenizer.extract_tokens(pipeline_data['text'].tolist()[0])
    print(files_tokens)


if __name__ == '__main__':
    main()
