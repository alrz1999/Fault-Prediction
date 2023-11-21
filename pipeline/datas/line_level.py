from data.models import LineLevelDatasetImporter
from data.utils import LineLevelDatasetHelper
from pipeline.models import PipelineStage, StageData


class LineLevelDatasetImporterStage(PipelineStage):
    def __init__(self, line_level_dataset_importer, replace_na_with_empty=True, return_blank_lines=False,
                 return_test_file_lines=False,
                 return_comment_lines=False):
        super().__init__()
        self.line_level_dataset_importer: LineLevelDatasetImporter = line_level_dataset_importer
        self.return_comment_lines = return_comment_lines
        self.replace_na_with_empty = replace_na_with_empty
        self.return_blank_lines = return_blank_lines
        self.return_test_file_lines = return_test_file_lines

    def import_dataset(self):
        return self.line_level_dataset_importer.get_processed_line_level_dataset(
            replace_na_with_empty=self.replace_na_with_empty,
            return_blank_lines=self.return_blank_lines,
            return_test_file_lines=self.return_test_file_lines,
            return_comment_lines=self.return_comment_lines
        )

    def process(self):
        self.result = self.import_dataset()
        self.stage_data[StageData.Keys.LINE_LEVEL_DF] = self.result


class LineLevelTokenizerStage(PipelineStage):
    def __init__(self, token_extractor):
        super().__init__()
        self.token_extractor = token_extractor

    def process(self):
        df = self.stage_data[StageData.Keys.LINE_LEVEL_DF]
        helper = LineLevelDatasetHelper(df, self.token_extractor)
        tokens = helper.get_all_lines_tokens()

        self.result = tokens
        self.stage_data[StageData.Keys.LINE_LEVEL_TOKENS] = self.result
