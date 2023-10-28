from data.models import *
from token_extraction import *


def main():
    project = Project("activemq")
    source_code_files = project.get_train_release().create_source_code_files()

    tokenizer = CFGTokenExtractor()
    print(tokenizer.extract_tokens(source_code_files[1000].code))


if __name__ == '__main__':
    main()
