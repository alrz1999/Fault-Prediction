from config import PREPROCESSED_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR
from data.models import Project
from embedding.preprocessing.token_extraction import CustomTokenExtractor
from pipeline.datas.line_level import LineLevelDatasetImporterStage
from pipeline.models import Pipeline, StageData


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )
    token_extractor = CustomTokenExtractor(to_lowercase=True, max_seq_len=None)

    stages = [
        LineLevelDatasetImporterStage(project.get_train_release()),
    ]
    pipeline_data = Pipeline(stages).run()[StageData.Keys.LINE_LEVEL_DF]
    print(token_extractor.extract_tokens(pipeline_data['code_line'].tolist()[0]))


if __name__ == '__main__':
    main()
