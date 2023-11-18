from data.utils import LineLevelDatasetHelper
from pipeline.pipeline import PipelineStage, log_method_execution
import pandas as pd


class LineLevelDatasetLoaderStage(PipelineStage):
    def __init__(self, file_path, replace_na_with_empty=True, return_blank_lines=False,
                 return_test_file_lines=False, return_comment_lines=False):
        super().__init__(None, True, False, file_path)
        self.return_comment_lines = return_comment_lines
        self.replace_na_with_empty = replace_na_with_empty
        self.return_blank_lines = return_blank_lines
        self.return_test_file_lines = return_test_file_lines

    @log_method_execution
    def import_output(self):
        df = pd.read_csv(self.file_path, encoding='latin')

        if self.replace_na_with_empty:
            df = df.fillna('')
        if not self.return_blank_lines:
            df = df[df['is_blank'] == False]
        if not self.return_test_file_lines:
            df = df[df['is_test_file'] == False]
        if not self.return_comment_lines:
            df = df[df['is_comment'] == False]

        self.output_data = df
        return df


class LineLevelTokenizerStage(PipelineStage):
    def __init__(self, input_data=None):
        super().__init__(input_data, False, False, None)

    def import_output(self):
        raise Exception()

    def export_output(self):
        raise Exception()

    @log_method_execution
    def process(self):
        df = self.input_data
        helper = LineLevelDatasetHelper(df)
        tokens = helper.get_all_lines_tokens()
        self.output_data = tokens
        return tokens
