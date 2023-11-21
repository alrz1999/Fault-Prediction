import os.path

from config import ORIGINAL_BUGGY_LINES_DATA_DIR
import re
import more_itertools

from embedding.preprocessing.token_extraction import TokenExtractor


def get_buggy_lines_dataset_path(release):
    return os.path.join(ORIGINAL_BUGGY_LINES_DATA_DIR, release + '_defective_lines_dataset.csv')


def is_empty_line(code_line):
    """
        input
            code_line (string)
        output
            boolean value
    """

    if len(code_line.strip()) == 0:
        return True

    return False


class CommentDetector:
    def __init__(self, code_str):
        self.code_str = code_str
        self.comments_list = self.extract_comments()

    def extract_comments(self):
        comments = re.findall(r'(/\*[\s\S]*?\*/)', self.code_str, re.DOTALL)
        comments_str = '\n'.join(comments)
        comments_list = comments_str.split('\n')
        return comments_list

    def is_comment_line(self, code_line):
        """
            input
                code_line (string): source code in a line
                comments_list (list): a list that contains every comments
            output
                boolean value
        """

        code_line = code_line.strip()

        if len(code_line) == 0:
            return False
        elif code_line.startswith('//'):
            return True
        elif code_line in self.comments_list:
            return True

        return False


class LineLevelDatasetHelper:
    def __init__(self, df, token_extractor: TokenExtractor):
        self.df = df
        self.token_extractor = token_extractor

    def get_all_lines_tokens(self):
        file_lines_tokens, _ = self.get_file_lines_tokens_and_labels()
        all_line_tokens = list(more_itertools.collapse(file_lines_tokens[:], levels=1))
        return all_line_tokens

    def get_file_lines_tokens_and_labels(self):
        file_line_tokens = []
        file_labels = []

        for filename, group_df in self.df.groupby('filename'):
            file_label = bool(group_df['file-label'].unique().any())

            lines = list(group_df['code_line'])

            file_code = [self.token_extractor.extract_tokens(line) for line in lines]
            file_line_tokens.append(file_code)
            file_labels.append(file_label)

        return file_line_tokens, file_labels
