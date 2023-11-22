from data.models import LineLevelDatasetImporter
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
        df = self.line_level_dataset_importer.get_processed_line_level_dataset(
            replace_na_with_empty=self.replace_na_with_empty,
            return_blank_lines=self.return_blank_lines,
            return_test_file_lines=self.return_test_file_lines,
            return_comment_lines=self.return_comment_lines
        )
        return df.rename(columns={'code_line': 'text', 'line-label': 'label'})


    def process(self):
        self.result = self.import_dataset()
        self.stage_data[StageData.Keys.LINE_LEVEL_DF.value] = self.result
