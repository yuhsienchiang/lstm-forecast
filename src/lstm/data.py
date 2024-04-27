from collections import namedtuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LSTM_Sample = namedtuple("LSTM_Sample", ["label", "instance"])
Batch_LSTM_Sample = namedtuple("Batch_LSTM_Sample", ["label", "instance"])


# keys to be used in as attrubutes
INSTANCE_KEYS = [
    "price",
    "demand",
    "temp_air",
    "pv_power",
    "pv_power_forecast_1h",
    "pv_power_forecast_2h",
    "pv_power_forecast_24h",
    "pv_power_basic",
]

# key to be predicted
LABEL_KEYS = ["price"]

# Keys to be used in the dataset
DATA_KEYS = INSTANCE_KEYS if LABEL_KEYS[0] in INSTANCE_KEYS else INSTANCE_KEYS + LABEL_KEYS


class LSTMDataset(Dataset):
    """
    Dataset class for preparing data for training LSTM models.

    Attributes:
    -----------
    raw_data: pd.DataFrame
        The raw data.
    instances: torch.Tensor
        The input sequences.
    labels: torch.Tensor
        The target labels.

    Methods:
    --------
    _null_value_handler(data: pd.DataFrame) -> pd.DataFrame:
        Handle null values in the data.

    min_max_normalize(data: pd.DataFrame) -> pd.DataFrame:
        Perform min-max normalization on the data.

    min_max_denormalize(data: float | list[float] | pd.Series, labels: str | list[str]) -> float | list[float] | pd.Series:
        Perform min-max denormalization on the data.

    collate_fn(batch) -> Batch_LSTM_Sample:
        Function passed to DataLoader for batching data.

    """

    def __init__(self, data_src: str, window_size: int) -> None:
        """
        Initialize the LSTMDataset.

        Parameters:
        -----------
        data_src: str or pd.DataFrame
            The data source. It can be either a file path (str) or a pandas DataFrame.
        window_size: int
            The window size for creating sequences of data.

        """
        super(LSTMDataset).__init__()

        if isinstance(data_src, str):
            self.raw_data = pd.read_csv(data_src)
        elif isinstance(data_src, pd.DataFrame):
            self.raw_data = data_src
        else:
            raise ValueError("data_src must be a string or a pandas DataFrame")

        self.window_size = window_size

        data = self._null_value_handler(self.raw_data.copy(deep=True))

        # get data size after removing unuseful data
        self.data_size = len(data) - self.window_size

        # normalise data
        data = self._min_max_normalization(data[DATA_KEYS].copy(deep=True))

        # setup data set in a format that can be used to train LSTM
        instances_np = data[INSTANCE_KEYS].to_numpy(dtype=np.float32)
        labels_np = data[LABEL_KEYS].to_numpy(dtype=np.float32)

        instances = np.zeros((self.data_size, self.window_size, len(INSTANCE_KEYS)), dtype=np.float32)
        labels = np.zeros((self.data_size, len(LABEL_KEYS)), dtype=np.float32)

        for i in range(self.data_size):
            instances[i] = [x for x in instances_np[i : i + self.window_size]]
            labels[i] = labels_np[i + self.window_size]

        self.instances = torch.from_numpy(instances)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return LSTM_Sample(label=self.labels[idx], instance=self.instances[idx])

    def _null_value_handler(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle null values in the data.

        Parameters:
        -----------
        data: pd.DataFrame
            The input data.

        Returns:
        --------
        pd.DataFrame:
            The cleaned data.

        """
        null_demand_idx = data[data["demand"].isnull()].index.tolist()
        null_temp_air_idx = data[data["temp_air"].isnull()].index.tolist()
        null_pv_power_idx = data[data["pv_power"].isnull()].index.tolist()
        null_pv_power_basic_idx = data[data["pv_power_basic"].isnull()].index.tolist()

        if (len(null_temp_air_idx) > 0 or len(null_pv_power_idx) > 0 or len(null_pv_power_basic_idx) > 0):
            new_start_idx = max(null_temp_air_idx[-1], null_pv_power_idx[-1], null_pv_power_basic_idx[-1])
            data = data.iloc[new_start_idx + 1 :]
        if len(null_demand_idx) > 0:
            data.loc[:, "demand"] = data["demand"].interpolate(method="linear")

        return data

    def min_max_normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform min-max normalization on the data.

        Parameters:
        -----------
        data: pd.DataFrame
            The input data.

        Returns:
        --------
        pd.DataFrame:
            The normalized data.

        """
        # min max normalization
        self.data_min = data.min()
        self.data_max = data.max()
        data = (data - self.data_min) / (self.data_max - self.data_min)

        return data

    def min_max_denormalize(
        self, data: float | list[float] | pd.Series, *, labels: str | list[str]
    ) -> float | list[float] | pd.Series:
        """
        Perform min-max denormalization on the data.

        Parameters:
        -----------
        data: float, list[float], pd.Series
            The data to be denormalized.
        labels: str or list[str]
            The labels to be denormalized.

        Returns:
        --------
        float or list[float] or pd.Series:
            The denormalized data.

        """

        if labels == "all" or "all" in labels:
            return data * (self.data_max - self.data_min) + self.data_min

        return data * (self.data_max[labels] - self.data_min[labels]) + self.data_min[labels]

    def collate_fn(self, batch):
        """
        Function passed to DataLoader for batching data
        """
        batch_instances = []
        batch_labels = []

        for label, instance in batch:
            batch_labels.append(label)
            batch_instances.append(instance)

        return Batch_LSTM_Sample(
            label=torch.stack(batch_labels, dim=0),
            instance=torch.stack(batch_instances, dim=1),
        )
