from config import LINE_LEVEL_DATA_SAVE_DIR, ORIGINAL_FILE_LEVEL_DATA_DIR, METHOD_LEVEL_DATA_SAVE_DIR
from data.models import Project


def generate_method_level_datasets():
    for project_name in Project.releases_by_project_name.keys():
        project = Project(
            name=project_name,
            line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
            file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
            method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
        )
        project.get_train_release().export_method_level_dataset()
        for release in project.get_eval_releases():
            release.export_method_level_dataset()


def generate_line_level_datasets():
    for project_name in Project.releases_by_project_name.keys():
        project = Project(
            name=project_name,
            line_level_dataset_save_dir=LINE_LEVEL_DATA_SAVE_DIR,
            file_level_dataset_dir=ORIGINAL_FILE_LEVEL_DATA_DIR,
            method_level_dataset_dir=METHOD_LEVEL_DATA_SAVE_DIR
        )
        project.get_train_release().export_line_level_dataset()
        for release in project.get_eval_releases():
            release.export_line_level_dataset()


if __name__ == '__main__':
    generate_line_level_datasets()
    # generate_method_level_datasets()
