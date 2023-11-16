import os.path

from config import ORIGINAL_BUGGY_LINES_DATA_DIR
import re
import more_itertools


def get_buggy_lines_dataset_path(release):
    return os.path.join(ORIGINAL_BUGGY_LINES_DATA_DIR, release + '_defective_lines_dataset.csv')


CHAR_TO_REMOVE = ['+', '-', '*', '/', '=', '++', '--', '\\', '<str>', '<char>', '|', '&', '!']


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


def preprocess_code_line(code_line):
    """
        input
            code_line (string)
    """

    code_line = re.sub("\'\'", "\'", code_line)
    code_line = re.sub("\".*?\"", "<str>", code_line)
    code_line = re.sub("\'.*?\'", "<char>", code_line)
    code_line = re.sub('\b\d+\b', '', code_line)
    code_line = re.sub("\\[.*?]", '', code_line)
    code_line = re.sub("[.|,:;{}()]", ' ', code_line)

    for char in CHAR_TO_REMOVE:
        code_line = code_line.replace(char, ' ')

    code_line = code_line.strip()

    return code_line


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
    def __init__(self, df):
        self.df = df

    def get_all_lines_tokens(self):
        file_lines_tokens, _ = self.get_file_lines_tokens_and_labels(True)
        all_line_tokens = list(more_itertools.collapse(file_lines_tokens[:], levels=1))
        return all_line_tokens

    def get_file_lines_tokens_and_labels(self, to_lowercase=False):
        file_line_tokens = []
        file_labels = []

        for filename, group_df in self.df.groupby('filename'):
            file_label = bool(group_df['file-label'].unique())

            lines = list(group_df['code_line'])

            file_code = self.get_line_tokens(lines, to_lowercase)
            file_line_tokens.append(file_code)
            file_labels.append(file_label)

        return file_line_tokens, file_labels

    def get_line_tokens(self, lines, to_lowercase=False, max_seq_len=50):
        line_tokens = []

        for line in lines:
            line = re.sub('\\s+', ' ', line)

            if to_lowercase:
                line = line.lower()

            tokens = line.strip().split()
            tokens_count = len(tokens)

            tokens = tokens[:max_seq_len]

            if tokens_count < max_seq_len:
                tokens = tokens + ['<pad>'] * (max_seq_len - tokens_count)

            line_tokens.append(tokens)

        return line_tokens
