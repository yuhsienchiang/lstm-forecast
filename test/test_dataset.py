import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.lstm.data import (
    INSTANCE_KEYS,
    LABEL_KEYS,
    Batch_LSTM_Sample,
    LSTM_Sample,
    LSTMDataset,
)


def test_str_data_src():
    data_src = "./data/training_data.csv"
    window_size = 10
    dataset = LSTMDataset(data_src=data_src, window_size=window_size)

    assert isinstance(dataset.instances, torch.Tensor)
    assert isinstance(dataset.labels, torch.Tensor)
    assert len(dataset.instances) != 0
    assert len(dataset.labels) != 0


def test_pd_data_src():
    data_src = pd.read_csv("./data/training_data.csv")
    window_size = 10
    dataset = LSTMDataset(data_src=data_src, window_size=window_size)

    assert isinstance(dataset.instances, torch.Tensor)
    assert isinstance(dataset.labels, torch.Tensor)
    assert len(dataset.instances) != 0
    assert len(dataset.labels) != 0


def test_data_shape():
    data_src = "./data/training_data.csv"
    window_size = 10

    dataset = LSTMDataset(data_src=data_src, window_size=window_size)
    data_size = len(dataset)

    assert dataset.instances.shape == (data_size, window_size, len(INSTANCE_KEYS))
    assert dataset.labels.shape == (data_size, len(LABEL_KEYS))


def test_get_iter():
    data_src = "./data/training_data.csv"
    window_size = 10
    dataset = LSTMDataset(data_src=data_src, window_size=window_size)

    data_sample = next(iter(dataset))

    assert isinstance(data_sample, LSTM_Sample)
    assert isinstance(data_sample.instance, torch.Tensor)
    assert isinstance(data_sample.label, torch.Tensor)


def test_data_loader():
    data_src = "./data/training_data.csv"
    window_size = 10
    batch_size = 8

    dataset = LSTMDataset(data_src=data_src, window_size=window_size)

    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    batch_data_sample = next(iter(train_dataloader))

    assert isinstance(batch_data_sample, Batch_LSTM_Sample)

    batch_labels = batch_data_sample.label
    batch_instance = batch_data_sample.instance

    assert batch_labels.shape == (batch_size, len(LABEL_KEYS))
    assert batch_instance.shape == (window_size, batch_size, len(INSTANCE_KEYS))
