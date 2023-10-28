import re

from config import FILE_LEVEL_DATA_DIR, LINE_LEVEL_DATA_DIR


def get_file_level_dataset_path(release):
    return FILE_LEVEL_DATA_DIR + release + '_ground-truth-files_dataset.csv'


def get_line_level_dataset_path(release):
    return LINE_LEVEL_DATA_DIR + release + '_defective_lines_dataset.csv'


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
