import numpy as np
import pandas as pd
import os
import re
import more_itertools

from data.utils import CommentDetector, preprocess_code_line, is_empty_line, get_file_level_dataset_path, \
    get_line_level_dataset_path


class Project:
    all_train_releases = {'activemq': 'activemq-5.0.0', 'camel': 'camel-1.4.0', 'derby': 'derby-10.2.1.6',
                          'groovy': 'groovy-1_5_7', 'hbase': 'hbase-0.94.0', 'hive': 'hive-0.9.0',
                          'jruby': 'jruby-1.1', 'lucene': 'lucene-2.3.0', 'wicket': 'wicket-1.3.0-incubating-beta-1'
                          }

    all_eval_releases = {'activemq': ['activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
                         'camel': ['camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
                         'derby': ['derby-10.3.1.4', 'derby-10.5.1.1'],
                         'groovy': ['groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
                         'hbase': ['hbase-0.95.0', 'hbase-0.95.2'], 'hive': ['hive-0.10.0', 'hive-0.12.0'],
                         'jruby': ['jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
                         'lucene': ['lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
                         'wicket': ['wicket-1.3.0-beta2', 'wicket-1.5.3']
                         }

    releases_by_project_name = {
        'activemq': ['activemq-5.0.0', 'activemq-5.1.0', 'activemq-5.2.0', 'activemq-5.3.0', 'activemq-5.8.0'],
        'camel': ['camel-1.4.0', 'camel-2.9.0', 'camel-2.10.0', 'camel-2.11.0'],
        'derby': ['derby-10.2.1.6', 'derby-10.3.1.4', 'derby-10.5.1.1'],
        'groovy': ['groovy-1_5_7', 'groovy-1_6_BETA_1', 'groovy-1_6_BETA_2'],
        'hbase': ['hbase-0.94.0', 'hbase-0.95.0', 'hbase-0.95.2'],
        'hive': ['hive-0.9.0', 'hive-0.10.0', 'hive-0.12.0'],
        'jruby': ['jruby-1.1', 'jruby-1.4.0', 'jruby-1.5.0', 'jruby-1.7.0.preview1'],
        'lucene': ['lucene-2.3.0', 'lucene-2.9.0', 'lucene-3.0.0', 'lucene-3.1'],
        'wicket': ['wicket-1.3.0-incubating-beta-1', 'wicket-1.3.0-beta2', 'wicket-1.5.3']
    }

    def __init__(self, name):
        self.name = name

    @staticmethod
    def get_project_releases(project_name):
        project_releases = []
        for release in Project.releases_by_project_name[project_name]:
            project_release = ProjectRelease(project_name, release)
            project_releases.append(project_release)
        return project_releases

    def export_line_level_dfs(self, save_dir):
        for release in Project.releases_by_project_name[self.name]:
            project_release = ProjectRelease(self.name, release)
            but_repository = BugRepository(project_release)
            project_release.export_line_level_df(but_repository, save_dir)

    def import_line_level_dfs(self, save_dir):
        df_by_release = {}
        for release in Project.releases_by_project_name[self.name]:
            df_by_release[release] = ProjectRelease.import_line_level_df(release, save_dir)

        return df_by_release

    def import_train_release_line_level_df(self, save_dir):
        train_release = self.get_train_release()
        return train_release.import_line_level_df(save_dir)

    def get_train_release(self):
        train_release = Project.all_train_releases[self.name]
        return ProjectRelease(self.name, train_release)

    def get_eval_releases(self):
        eval_releases = Project.all_eval_releases[self.name]
        output = []
        for release in eval_releases:
            output.append(
                ProjectRelease(self.name, release)
            )

        return output


class ProjectRelease:
    def __init__(self, project_name, release):
        self.project_name = project_name
        self.release = release

    def export_line_level_df(self, bug_repository, save_dir):
        preprocessed_df_list = []
        for source_code_file in self.create_source_code_files():
            source_code_file: SourceCodeFile
            code_df = source_code_file.create_line_level_code_df(bug_repository)
            if len(code_df) != 0:
                preprocessed_df_list.append(code_df)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        all_df = pd.concat(preprocessed_df_list)
        all_df.to_csv(save_dir + self.release + ".csv", index=False)
        print('finish release {}'.format(self.release))

    def create_source_code_files(self):
        return SourceCodeFile.create_source_code_files(
            get_file_level_dataset_path(self.release)
        )

    def import_line_level_df(self, save_dir):
        # TODO
        file_path = os.path.join(save_dir, self.release + ".csv")
        df = pd.read_csv(file_path, encoding='latin')
        df = df.fillna('')

        df = df[df['is_blank'] == False]
        df = df[df['is_test_file'] == False]
        return df

    def get_buggy_filenames(self):
        source_code_files = self.create_source_code_files()
        return {file.filename for file in source_code_files if file.is_buggy}

    def get_all_lines(self, save_dir):
        df = self.import_line_level_df(save_dir)
        df = df.fillna('')
        df = df[df['is_blank'] == False]
        df = df[df['is_test_file'] == False]
        train_code_3d, _ = self.get_code3d_and_label(df, True)
        all_texts = list(more_itertools.collapse(train_code_3d[:], levels=1))
        return all_texts

    def get_code3d_and_label(self, df, to_lowercase=False):
        '''
            input
                df (DataFrame): a dataframe from get_df()
            output
                code3d (nested list): a list of code2d from prepare_code2d()
                all_file_label (list): a list of file-level label
        '''

        code3d = []
        all_file_label = []

        for filename, group_df in df.groupby('filename'):
            file_label = bool(group_df['file-label'].unique())

            code = list(group_df['code_line'])

            code2d = self.prepare_code2d(code, to_lowercase)
            code3d.append(code2d)

            all_file_label.append(file_label)

        return code3d, all_file_label

    def prepare_code2d(self, code_list, to_lowercase=False, max_seq_len=50):
        '''
            input
                code_list (list): list that contains code each line (in str format)
            output
                code2d (nested list): a list that contains list of tokens with padding by '<pad>'
        '''
        code2d = []

        for c in code_list:
            c = re.sub('\\s+', ' ', c)

            if to_lowercase:
                c = c.lower()

            token_list = c.strip().split()
            total_tokens = len(token_list)

            token_list = token_list[:max_seq_len]

            if total_tokens < max_seq_len:
                token_list = token_list + ['<pad>'] * (max_seq_len - total_tokens)

            code2d.append(token_list)

        return code2d


class BugRepository:
    def __init__(self, project_release):
        self.buggy_filenames = project_release.get_buggy_filenames()
        self.line_level_df = pd.read_csv(get_line_level_dataset_path(project_release.release), encoding='latin')

    def is_buggy_file(self, filename):
        return filename in self.buggy_filenames

    def get_file_buggy_lines(self, filename):
        return list(
            self.line_level_df[self.line_level_df['File'] == filename]['Line_number']
        )


class SourceCodeFile:
    def __init__(self, filename, code, is_buggy):
        self.filename = filename
        self.code = code
        self.is_buggy = is_buggy

    @staticmethod
    def create_source_code_files(release_path):
        df = pd.read_csv(release_path, encoding='latin')
        df = df.fillna('')
        source_code_files = []
        for idx, row in df.iterrows():
            filename = row['File']
            if '.java' not in filename:
                continue
            code = row['SRC']
            is_buggy_file = row['Bug']
            source_code_file = SourceCodeFile(
                filename=filename,
                code=code,
                is_buggy=is_buggy_file
            )

            source_code_files.append(source_code_file)
        return source_code_files

    def create_line_level_code_df(self, bug_repository):
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

            if not is_comment:
                line = preprocess_code_line(line)

            is_blank_line.append(is_empty_line(line))
            preprocess_code_lines.append(line)

        if 'test' in self.filename:
            is_test = True
        else:
            is_test = False

        df['filename'] = [self.filename] * len(code_lines)
        df['is_test_file'] = [is_test] * len(code_lines)
        df['code_line'] = preprocess_code_lines
        df['line_number'] = np.arange(1, len(code_lines) + 1)
        df['is_comment'] = is_comments
        df['is_blank'] = is_blank_line

        df['file-label'] = [self.is_buggy] * len(code_lines)
        df['line-label'] = [False] * len(code_lines)

        if bug_repository.is_buggy_file(self.filename):
            buggy_lines = bug_repository.get_file_buggy_lines(self.filename)
            df['line-label'] = df['line_number'].isin(buggy_lines)

        return df
