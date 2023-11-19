from embedding.preprocessing.token_extraction import TokenExtractor
from classification.utils import LineLevelToFileLevelDatasetMapper
from pipeline.models import PipelineStage, StageData
import pandas as pd


class FileLevelDatasetImporterStage(PipelineStage):

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def import_df(self):
        return pd.read_csv(self.file_path, encoding='latin')

    def process(self):
        self.result = self.import_df()
        self.stage_data[StageData.Keys.FILE_LEVEL_DF] = self.result


class FileLevelTokenizerStage(PipelineStage):
    def __init__(self, token_extractor: TokenExtractor):
        super().__init__()
        self.token_extractor = token_extractor

    def process(self):
        df = self.stage_data[StageData.Keys.FILE_LEVEL_DF]
        files_tokens = []
        files_source_codes = df["SRC"]
        for source_code in files_source_codes:
            tokens = self.token_extractor.extract_tokens(source_code)
            files_tokens.append(tokens)

        self.result = files_tokens
        self.stage_data[StageData.Keys.FILE_LEVEL_TOKENS] = self.result


class LineLevelToFileLevelDatasetMapperStage(PipelineStage):

    def __init__(self, to_lower_case=True):
        super().__init__()
        self.to_lower_case = to_lower_case

    def process(self):
        line_level_df = self.stage_data[StageData.Keys.LINE_LEVEL_DF]
        train_code, train_label = LineLevelToFileLevelDatasetMapper().prepare_data(line_level_df, self.to_lower_case)
        data = {'SRC': train_code, 'Bug': train_label}
        file_level_df = pd.DataFrame(data)
        self.result = file_level_df
        self.stage_data[StageData.Keys.FILE_LEVEL_DF] = self.result
