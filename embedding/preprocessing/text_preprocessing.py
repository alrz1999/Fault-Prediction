import re
import string
from nltk.corpus import stopwords

CHAR_TO_REMOVE = ['+', '-', '*', '/', '=', '++', '--', '\\', '<str>', '<char>', '|', '&', '!']


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


def preprocess_text(input_text):
    # Lowercase the text
    input_text = input_text.lower()

    # Remove punctuation
    input_text = input_text.translate(str.maketrans('', '', string.punctuation))

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = input_text.split()
    input_text = ' '.join([word for word in words if word not in stop_words])

    # Remove extra white spaces
    input_text = ' '.join(input_text.split())

    return input_text
