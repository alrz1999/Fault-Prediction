from config import PREPROCESSED_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR
from data.models import *
from token_extraction import *


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=PREPROCESSED_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR
    )

    code = project.get_train_release().get_file_level_dataset().iloc[0]["SRC"]

    tokenizer = CFGTokenExtractor()
    print(tokenizer.extract_tokens(code))


if __name__ == '__main__':
    main()
