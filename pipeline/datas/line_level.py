from data.utils import LineLevelDatasetHelper
from pipeline.models import PipelineStage, StageData
import pandas as pd


class LineLevelDatasetImporterStage(PipelineStage):
    def __init__(self, file_path, replace_na_with_empty=True, return_blank_lines=False, return_test_file_lines=False,
                 return_comment_lines=False):
        super().__init__()
        # TODO can get a datasetGenerator as input instead of file path
        self.file_path = file_path
        self.return_comment_lines = return_comment_lines
        self.replace_na_with_empty = replace_na_with_empty
        self.return_blank_lines = return_blank_lines
        self.return_test_file_lines = return_test_file_lines

    def import_dataset(self):
        df = pd.read_csv(self.file_path, encoding='latin')

        if self.replace_na_with_empty:
            df = df.fillna('')
        if not self.return_blank_lines:
            df = df[df['is_blank'] == False]
        if not self.return_test_file_lines:
            df = df[df['is_test_file'] == False]
        if not self.return_comment_lines:
            df = df[df['is_comment'] == False]

        return df

    def process(self):
        self.result = self.import_dataset()
        self.stage_data[StageData.Keys.LINE_LEVEL_DF] = self.result


class LineLevelTokenizerStage(PipelineStage):
    def __init__(self, to_lowercase=True, max_seq_len=None):
        super().__init__()
        self.to_lowercase = to_lowercase
        self.max_seq_len = max_seq_len

    def process(self):
        df = self.stage_data[StageData.Keys.LINE_LEVEL_DF]
        helper = LineLevelDatasetHelper(df)
        tokens = helper.get_all_lines_tokens(
            to_lowercase=self.to_lowercase,
            max_seq_len=self.max_seq_len
        )

        self.result = tokens
        self.stage_data[StageData.Keys.LINE_LEVEL_TOKENS] = tokens
