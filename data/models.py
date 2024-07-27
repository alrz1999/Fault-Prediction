import numpy as np
import pandas as pd
import os
from pathlib import Path

from data.utils import CommentDetector, is_empty_line, get_buggy_lines_dataset_path
from embedding.preprocessing.token_extraction import ASTExtractor


class MethodLevelDatasetImporter:
    def get_method_level_dataset(self):
        raise NotImplementedError()

    def get_processed_method_level_dataset(self):
        df = self.get_method_level_dataset()
        return df.rename(columns={'SRC': 'text', 'Bug': 'label'})


class FileLevelDatasetImporter:
    def get_file_level_dataset(self):
        raise NotImplementedError()

    def get_processed_file_level_dataset(self):
        df = self.get_file_level_dataset()
        return df.rename(columns={'SRC': 'text', 'Bug': 'label'})


class LineLevelDatasetImporter:
    def get_line_level_dataset(self):
        raise NotImplementedError()

    def get_processed_line_level_dataset(self, replace_na_with_empty=True, return_blank_lines=False,
                                         return_test_file_lines=False, return_comment_lines=False):
        df = self.get_line_level_dataset()

        if replace_na_with_empty:
            df = df.fillna('')
        if not return_blank_lines:
            df = df[df['is_blank'] == False]
        if not return_test_file_lines:
            df = df[df['is_test_file'] == False]
        if not return_comment_lines:
            df = df[df['is_comment'] == False]

        df = df.rename(columns={'code_line': 'text', 'line-label': 'label'})
        return df


class AggregatedDatasetImporter(LineLevelDatasetImporter, FileLevelDatasetImporter, MethodLevelDatasetImporter):
    def __init__(self, releases):
        self.releases = releases

    def get_line_level_dataset(self):
        dfs = [rel.get_line_level_dataset() for rel in self.releases if isinstance(rel, LineLevelDatasetImporter)]
        return pd.concat(dfs, ignore_index=True)

    def get_file_level_dataset(self):
        dfs = [rel.get_file_level_dataset() for rel in self.releases if isinstance(rel, FileLevelDatasetImporter)]
        return pd.concat(dfs, ignore_index=True)


class Project(LineLevelDatasetImporter, MethodLevelDatasetImporter, FileLevelDatasetImporter):
    all_train_releases = {
        'activemq': 'activemq-4.0.0', 'new-activemq': 'new-activemq-5.0.0', 'camel': 'camel-1.4.0',
        'derby': 'derby-10.2.1.6', 'groovy': 'groovy-1_5_7', 'hbase': 'hbase-0.94.0', 'hive': 'hive-0.9.0',
        'jruby': 'jruby-1.1', 'lucene': 'lucene-2.3.0', 'wicket': 'wicket-1.3.0-incubating-beta-1',

        'promise_ant': 'ant-1.3', 'promise_camel': 'camel-1.0', 'promise_jedit': 'jedit-3.2',
        'promise_log4j': 'log4j-1.0', 'promise_lucene': 'lucene-2.0', 'promise_poi': 'poi-1.5',
        'promise_synapse': 'synapse-1.0', 'promise_velocity': 'velocity-1.4',
        'promise_xalan': 'xalan-2.4', 'promise_xerces': 'xerces-1.1',
    }

    all_eval_releases = {
        'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
        'new-activemq': ['new-activemq-5.1.0', 'new-activemq-5.2.0', 'new-activemq-5.3.0', 'new-activemq-5.8.0'],
        'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
        'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'],
        'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
        'hbase': ['hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.10.0', 'hive-0.12.0'],
        'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
        'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
        'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3'],

        'promise_ant': ['ant-1.4', 'ant-1.5', 'ant-1.6', 'ant-1.7'],
        'promise_camel': ['camel-1.2', 'camel-1.4', 'camel-1.6'],
        'promise_jedit': ['jedit-4.0', 'jedit-4.1', 'jedit-4.2', 'jedit-4.3'],
        'promise_log4j': ['log4j-1.1', 'log4j-1.2'],
        'promise_lucene': ['lucene-2.2', 'lucene-2.4'],
        'promise_poi': ['poi-2.0', 'poi-2.5', 'poi-3.0'],
        'promise_synapse': ['synapse-1.1', 'synapse-1.2'],
        'promise_velocity': ['velocity-1.5', 'velocity-1.6'],
        'promise_xalan': ['xalan-2.5', 'xalan-2.6', 'xalan-2.7'],
        'promise_xerces': ['xerces-1.2', 'xerces-1.3', 'xerces-1.4.4'],
    }

    releases_by_project_name = {
        'activemq': ['activemq-4.0.0', 'activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0',
                     'activemq-5.8.0'],
        'new-activemq': ['new-activemq-5.0.0', 'new-activemq-5.1.0', 'new-activemq-5.2.0', 'new-activemq-5.3.0',
                         'new-activemq-5.8.0'],
        'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
        'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'],
        'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
        'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'],
        'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'],
        'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
        'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
        'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3'],

        'promise_ant': ['ant-1.3', 'ant-1.4', 'ant-1.5', 'ant-1.6', 'ant-1.7'],
        'promise_camel': ['camel-1.0', 'camel-1.2', 'camel-1.4', 'camel-1.6'],
        'promise_jedit': ['jedit-3.2', 'jedit-4.0', 'jedit-4.1', 'jedit-4.2', 'jedit-4.3'],
        'promise_log4j': ['log4j-1.0', 'log4j-1.1', 'log4j-1.2'],
        'promise_lucene': ['lucene-2.0', 'lucene-2.2', 'lucene-2.4'],
        'promise_poi': ['poi-1.5', 'poi-2.0', 'poi-2.5', 'poi-3.0'],
        'promise_synapse': ['synapse-1.0', 'synapse-1.1', 'synapse-1.2'],
        'promise_velocity': ['velocity-1.4', 'velocity-1.5', 'velocity-1.6'],
        'promise_xalan': ['xalan-2.4', 'xalan-2.5', 'xalan-2.6', 'xalan-2.7'],
        'promise_xerces': ['xerces-1.1', 'xerces-1.2', 'xerces-1.3', 'xerces-1.4.4'],
    }

    def __init__(self, name, line_level_dataset_save_dir, file_level_dataset_dir, method_level_dataset_dir):
        self.line_level_dataset_save_dir = line_level_dataset_save_dir
        self.file_level_dataset_dir = file_level_dataset_dir
        self.method_level_dataset_dir = method_level_dataset_dir
        self.name = name

    @staticmethod
    def get_project_releases(project_name, line_level_dataset_save_dir, file_level_dataset_save_dir,
                             method_level_dataset_dir):
        project_releases = []
        for release_name in Project.releases_by_project_name[project_name]:
            project_release = ProjectRelease(
                project_name=project_name,
                release_name=release_name,
                line_level_bug_repository=LineLevelBugRepository(release_name),
                line_level_dataset_save_dir=line_level_dataset_save_dir,
                file_level_dataset_save_dir=file_level_dataset_save_dir,
                method_level_dataset_dir=method_level_dataset_dir
            )
            project_releases.append(project_release)
        return project_releases

    def get_line_level_dataset(self):
        all_dataframes = []
        project_releases = Project.get_project_releases(
            project_name=self.name,
            line_level_dataset_save_dir=self.line_level_dataset_save_dir,
            file_level_dataset_save_dir=self.file_level_dataset_dir,
            method_level_dataset_dir=self.method_level_dataset_dir
        )

        for project_release in project_releases:
            df = project_release.get_line_level_dataset()
            all_dataframes.append(df)

        aggregated_dataframe = pd.concat(all_dataframes, ignore_index=True)

        return aggregated_dataframe

    def get_line_level_dataset_path(self):
        return os.path.join(self.line_level_dataset_save_dir, self.name + ".csv")

    def get_file_level_dataset_path(self):
        # TODO
        return os.path.join(self.file_level_dataset_dir, )

    def get_train_release(self):
        train_release = Project.all_train_releases[self.name]
        return ProjectRelease(
            line_level_dataset_save_dir=self.line_level_dataset_save_dir,
            project_name=self.name,
            release_name=train_release,
            line_level_bug_repository=LineLevelBugRepository(train_release),
            file_level_dataset_save_dir=self.file_level_dataset_dir,
            method_level_dataset_dir=self.method_level_dataset_dir
        )

    def get_validation_release(self):
        return self.get_eval_releases()[0]

    def get_eval_releases(self):
        eval_releases = Project.all_eval_releases[self.name]
        output = []
        for release in eval_releases:
            output.append(
                ProjectRelease(
                    line_level_dataset_save_dir=self.line_level_dataset_save_dir,
                    file_level_dataset_save_dir=self.file_level_dataset_dir,
                    project_name=self.name,
                    release_name=release,
                    line_level_bug_repository=LineLevelBugRepository(release),
                    method_level_dataset_dir=self.method_level_dataset_dir
                )
            )

        return output


class ProjectRelease(LineLevelDatasetImporter, MethodLevelDatasetImporter, FileLevelDatasetImporter):
    def __init__(self, project_name, release_name, line_level_dataset_save_dir=None, file_level_dataset_save_dir=None,
                 line_level_bug_repository=None, file_level_bug_repository=None, method_level_dataset_dir=None):

        self.line_level_dataset_save_dir = line_level_dataset_save_dir
        self.file_level_dataset_dir = file_level_dataset_save_dir
        self.project_name = project_name
        self.release_name = release_name
        self.line_level_bug_repository = line_level_bug_repository
        self.file_level_bug_repository = file_level_bug_repository
        self.method_level_dataset_dir = method_level_dataset_dir

    def get_file_level_dataset(self):
        file_path = self.get_file_level_dataset_path()
        return pd.read_csv(file_path, encoding='latin')

    def get_method_level_dataset(self):
        file_path = self.get_method_level_dataset_path()
        return pd.read_csv(file_path, encoding='latin')

    def get_line_level_dataset(self):
        file_path = self.get_line_level_dataset_path()
        return pd.read_csv(file_path, encoding='latin')

    def export_line_level_dataset(self):
        preprocessed_df_list = []
        source_code_files = SourceCodeFile.from_file_level_dataset(
            file_level_dataset=self.get_file_level_dataset(),
            line_level_dataset_save_dir=self.line_level_dataset_save_dir,
            line_level_bug_repository=self.line_level_bug_repository
        )

        for source_code_file in source_code_files:
            source_code_file: SourceCodeFile
            code_df = source_code_file.get_line_level_dataset()
            if len(code_df) != 0:
                preprocessed_df_list.append(code_df)

        all_df = pd.concat(preprocessed_df_list)
        all_df.to_csv(self.get_line_level_dataset_path(), index=False)
        print('finish release {}'.format(self.release_name))

    def export_method_level_dataset(self):
        preprocessed_df_list = []
        source_code_files = SourceCodeFile.from_file_level_dataset(
            file_level_dataset=self.get_file_level_dataset(),
            line_level_dataset_save_dir=self.line_level_dataset_save_dir,
            line_level_bug_repository=self.line_level_bug_repository
        )

        for source_code_file in source_code_files:
            source_code_file: SourceCodeFile
            code_df = source_code_file.get_method_level_dataset()
            if len(code_df) != 0:
                preprocessed_df_list.append(code_df)

        all_df = pd.concat(preprocessed_df_list)
        all_df.to_csv(self.get_method_level_dataset_path(), index=False)
        print('finish release {}'.format(self.release_name))

    def get_line_level_dataset_path(self):
        Path(self.line_level_dataset_save_dir).mkdir(parents=True, exist_ok=True)
        return str(os.path.join(self.line_level_dataset_save_dir, self.release_name + ".csv"))

    def get_method_level_dataset_path(self):
        Path(self.method_level_dataset_dir).mkdir(parents=True, exist_ok=True)
        return str(os.path.join(self.method_level_dataset_dir, self.release_name + ".csv"))

    def get_file_level_dataset_path(self):
        Path(self.file_level_dataset_dir).mkdir(parents=True, exist_ok=True)
        return str(os.path.join(self.file_level_dataset_dir, self.release_name + '_ground-truth-files_dataset.csv'))


class FileLevelBugRepository:
    def __init__(self, buggy_filenames):
        self.buggy_filenames = buggy_filenames

    def is_buggy_file(self, filename):
        return filename in self.buggy_filenames


class LineLevelBugRepository:
    def __init__(self, release):
        self.buggy_lines_dataset_path = get_buggy_lines_dataset_path(release)

    def get_file_buggy_lines(self, filename):
        line_level_df = pd.read_csv(self.buggy_lines_dataset_path, encoding='latin')
        return list(
            line_level_df[line_level_df['File'] == filename]['Line_number']
        )


class SourceCodeFile(LineLevelDatasetImporter):
    def __init__(self, filename, code, is_buggy, line_level_dataset_save_dir, line_level_bug_repository=None):
        self.line_level_dataset_save_dir = line_level_dataset_save_dir
        self.filename = filename
        self.code = code
        self.is_buggy = is_buggy
        self.line_level_bug_repository = line_level_bug_repository

    @staticmethod
    def from_file_level_dataset(file_level_dataset, line_level_dataset_save_dir, line_level_bug_repository):
        file_level_dataset = file_level_dataset.fillna('')
        source_code_files = []
        for idx, row in file_level_dataset.iterrows():
            filename = row['File']
            if '.java' not in filename:
                continue
            code = row['SRC']
            is_buggy_file = row['Bug']
            source_code_file = SourceCodeFile(
                filename=filename,
                code=code,
                is_buggy=is_buggy_file,
                line_level_dataset_save_dir=line_level_dataset_save_dir,
                line_level_bug_repository=line_level_bug_repository
            )

            source_code_files.append(source_code_file)
        return source_code_files

    def get_line_level_dataset(self):
        """
            output
                code_df (DataFrame): a dataframe of source code that contains the following columns
                - code_line (str): source code in a line
                - line_number (str): line number of source code line
                - is_comment (bool): boolean which indicates if a line is comment
                - is_blank_line(bool): boolean which indicates if a line is blank
        """

        df = pd.DataFrame()

        code_lines = self.code.splitlines()
        comment_detector = CommentDetector(self.code)

        preprocess_code_lines = []
        is_comments = []
        is_blank_line = []

        for line in code_lines:
            line = line.strip()
            is_comment = comment_detector.is_comment_line(line)
            is_comments.append(is_comment)
            # preprocess code here then check empty line...

            # if not is_comment:
            #     line = preprocess_code_line(line)

            is_blank_line.append(is_empty_line(line))
            preprocess_code_lines.append(line)

        if 'test' in self.filename:
            is_test = True
        else:
            is_test = False

        df['filename'] = [self.filename] * len(code_lines)
        df['is_test_file'] = [is_test] * len(code_lines)
        df['text'] = preprocess_code_lines
        df['line_number'] = np.arange(1, len(code_lines) + 1)
        df['is_comment'] = is_comments
        df['is_blank'] = is_blank_line

        df['file-label'] = [self.is_buggy] * len(code_lines)
        df['label'] = [False] * len(code_lines)

        if self.is_buggy:
            buggy_lines = self.line_level_bug_repository.get_file_buggy_lines(self.filename)
            df['label'] = df['line_number'].isin(buggy_lines)

        return df

    def get_method_level_dataset(self):
        df = pd.DataFrame()

        try:
            methods_data = ASTExtractor(cross_project=False).extract_methods_data(self.code)
        except Exception:
            return df
        if self.is_buggy:
            buggy_lines = self.line_level_bug_repository.get_file_buggy_lines(self.filename)
        else:
            buggy_lines = []

        start_lines = []
        end_lines = []
        method_texts = []
        is_buggy = []
        method_asts = []
        for method_data in methods_data:
            start_line, end_line, method_text, method_ast_tokens = method_data
            is_buggy_method = False
            for buggy_line in buggy_lines:
                if start_line <= buggy_line <= end_line:
                    is_buggy_method = True

            start_lines.append(start_line)
            end_lines.append(end_line)
            method_texts.append(method_text)
            is_buggy.append(is_buggy_method)
            method_asts.append(str.join(',', method_ast_tokens))

        if 'test' in self.filename:
            is_test = True
        else:
            is_test = False

        df['filename'] = [self.filename] * len(method_texts)
        df['is_test_file'] = [is_test] * len(method_texts)
        df['start_line'] = start_lines
        df['end_line'] = end_lines
        df['label'] = is_buggy
        df['text'] = method_texts
        df['ast'] = method_asts

        return df
