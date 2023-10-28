from model.BoW.BoW_baseline import *
from data.models import Project


def main():
    project = Project("activemq")
    train_release = project.get_train_release()
    eval_releases = project.get_eval_releases()
    # train_model(train_release)
    predict_defective_files_in_releases(train_release, eval_releases)


if __name__ == '__main__':
    main()
