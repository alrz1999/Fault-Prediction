from config import LINE_LEVEL_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR, METHOD_LEVEL_DATA_SAVE_DIR
from data.models import Project
from embedding.models import GensimWord2VecModel
from embedding.preprocessing.token_extraction import CustomTokenExtractor


def main():
    project = Project(
        name="activemq",
        line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
        file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
        method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
    )
    train_release = project.get_train_release()
    token_extractor = CustomTokenExtractor(to_lowercase=True, max_seq_len=None)
    embedding_dim = 50

    line_level_dataset = train_release.get_processed_line_level_dataset()
    model = GensimWord2VecModel.train(
        texts=line_level_dataset['text'],
        metadata={
            'token_extractor': token_extractor,
            'embedding_dim': embedding_dim
        }
    )

    print(model.text_to_vec(['if']))


if __name__ == '__main__':
    main()
