from classification.utils import LineLevelToFileLevelDatasetMapper
from pipeline.models import PipelineStage, StageData
import pandas as pd


class FileLevelDatasetImporterStage(PipelineStage):

    def __init__(self, file_level_dataset_importer):
        super().__init__()
        self.file_level_dataset_importer = file_level_dataset_importer

    def import_df(self):
        df = self.file_level_dataset_importer.get_file_level_dataset()
        return df.rename(columns={'SRC': 'text', 'Bug': 'label'})

    def process(self):
        self.result = self.import_df()
        self.stage_data[StageData.Keys.FILE_LEVEL_DF.value] = self.result


class LineLevelToFileLevelDatasetMapperStage(PipelineStage):

    def __init__(self, to_lower_case=True):
        super().__init__()
        self.to_lower_case = to_lower_case

    def process(self):
        line_level_df = self.stage_data[StageData.Keys.LINE_LEVEL_DF.value]
        train_code, train_label = LineLevelToFileLevelDatasetMapper().prepare_data(line_level_df, self.to_lower_case)
        data = {'text': train_code, 'label': train_label}
        file_level_df = pd.DataFrame(data)
        self.result = file_level_df
        self.stage_data[StageData.Keys.FILE_LEVEL_DF.value] = self.result
