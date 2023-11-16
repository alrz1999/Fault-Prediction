from embedding.preprocessing.token_extraction import TokenExtractor
from classification.utils import LineLevelToFileLevelDatasetMapper
from pipeline.pipeline import PipelineStage, log_method_execution
import pandas as pd


class FileLevelDatasetLoaderStage(PipelineStage):

    def __init__(self, file_path):
        super().__init__(None, True, False, file_path)

    @log_method_execution
    def import_output(self):
        self.output_data = pd.read_csv(self.file_path, encoding='latin')
        return self.output_data


class FileLevelTokenizerStage(PipelineStage):
    def __init__(self, token_extractor: TokenExtractor, input_data=None):
        super().__init__(input_data, False, False, None)
        self.token_extractor = token_extractor

    @log_method_execution
    def process(self):
        files_tokens = []
        files_source_codes = self.input_data["SRC"]
        for source_code in files_source_codes:
            tokens = self.token_extractor.extract_tokens(source_code)
            files_tokens.append(tokens)

        self.output_data = files_tokens
        return files_tokens


class LineLevelToFileLevelDatasetMapperStage(PipelineStage):
    def __init__(self, input_data=None):
        super().__init__(input_data, False, False, None)

    @log_method_execution
    def process(self):
        file_level_df = self.input_data
        train_code, train_label = LineLevelToFileLevelDatasetMapper().prepare_data(file_level_df, True)
        data = {'SRC': train_code, 'Bug': train_label}
        line_level_df = pd.DataFrame(data)
        self.output_data = line_level_df
        return line_level_df
