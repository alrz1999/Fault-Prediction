from imblearn.over_sampling import SMOTE


class BalanceData:
    def __init__(self, sampling_strategy='auto', random_state=None):
        self.sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

    def resample(self, X, y):
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        return X_resampled, y_resampled
