from config import DATE_SAVE_DIR
from data.models import Project


def main():
    project = Project("activemq")
    all_texts = project.get_train_release().get_all_lines(DATE_SAVE_DIR)
    print(all_texts[101])


if __name__ == '__main__':
    main()
