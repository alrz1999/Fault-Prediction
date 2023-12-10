from config import LINE_LEVEL_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR, METHOD_LEVEL_DATA_SAVE_DIR
from data.models import Project
from embedding.preprocessing.token_extraction import CustomTokenExtractor


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
        method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
    )
    token_extractor = CustomTokenExtractor(to_lowercase=True, max_seq_len=None)

    line_level_dataset = project.get_train_release().get_processed_line_level_dataset()
    print(token_extractor.extract_tokens(line_level_dataset['text'].tolist()[0]))


if __name__ == '__main__':
    main()
