class LineLevelToFileLevelDatasetMapper:
    @classmethod
    def get_code_str(cls, code, to_lowercase):
        code_str = '\n'.join(code)

        if to_lowercase:
            code_str = code_str.lower()

        return code_str

    @classmethod
    def prepare_data(cls, df, to_lowercase=False):
        all_code_str = []
        all_file_label = []

        for filename, group_df in df.groupby('filename'):
            file_label = bool(group_df['file-label'].unique().any())

            code = list(group_df['text'])

            code_str = cls.get_code_str(code, to_lowercase)

            all_code_str.append(code_str)

            all_file_label.append(file_label)

        return all_code_str, all_file_label
