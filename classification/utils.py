import tensorflow as tf


class LineLevelToFileLevelDatasetMapper:
    def get_code_str(self, code, to_lowercase):
        """
            input
                code (list): a list of code lines from dataset
                to_lowercase (bool)
            output
                code_str: a code in string format
        """

        code_str = '\n'.join(code)

        if to_lowercase:
            code_str = code_str.lower()

        return code_str

    def prepare_data(self, df, to_lowercase=False):
        '''
            input
                df (DataFrame): input data from get_df() function
            output
                all_code_str (list): a list of source code in string format
                all_file_label (list): a list of label
        '''
        all_code_str = []
        all_file_label = []

        for filename, group_df in df.groupby('filename'):
            file_label = bool(group_df['file-label'].unique())

            code = list(group_df['code_line'])

            code_str = self.get_code_str(code, to_lowercase)

            all_code_str.append(code_str)

            all_file_label.append(file_label)

        return all_code_str, all_file_label


def create_tensorflow_dataset(data, batch_size=None, shuffle=False, key_column='SRC'):
    if key_column == 'embedding':
        dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(data[key_column].tolist(), dtype=tf.int64), data['Bug'].values))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((data[key_column].values, data['Bug'].values))
    if shuffle:
        dataset = dataset.shuffle(len(data))
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset
