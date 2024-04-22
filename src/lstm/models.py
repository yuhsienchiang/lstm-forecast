import os
from copy import deepcopy

import torch
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(self, features_dim: int, device: str) -> None:
        super(Attention, self).__init__()
        self.features_dim = features_dim
        self.device = self._set_device(device)
        self.fc = nn.Linear(in_features=features_dim, out_features=1, device=self.device)

    def forward(self, inputs: Tensor) -> Tensor:
        # calculate weights
        a_c = self.fc(inputs)
        weights = torch.softmax(a_c, dim=1, dtype=torch.float32)
        # return weighted sum
        context = torch.sum(weights * inputs, dim=1, dtype=torch.float32)
        return context

    def _set_device(self, device_type: str) -> torch.device:
        if device_type.lower() == "mps":
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif device_type.lower() == "cuda":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        return device


class LSTMNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        fc_dim: int = 16,
        output_dim: int = 1,
        attn_layer: bool = False,
        stateful: bool = False,
        device: str = "cpu",
    ) -> None:
        """
        Parameters:
        -----------
        stateful: bool
            - hidden state and cell state are passed from one batch to the next
            - to match up the hidden state and cell state with the next batch, the batch size should be the same as time serice window size
        """
        super(LSTMNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc_dim = fc_dim
        self.output_dim = output_dim
        self.attn_layer = attn_layer
        self.stateful = stateful
        self.device = self._set_device(device)

        self.lstm_cell = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim, device=self.device)

        if self.attn_layer:
            self.attn = Attention(features_dim=hidden_dim, device=device)

        self.output = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=fc_dim, device=self.device),
            nn.Tanh(),
            nn.Linear(in_features=fc_dim, out_features=fc_dim, device=self.device),
            nn.Tanh(),
            nn.Linear(in_features=fc_dim, out_features=fc_dim // 2, device=self.device),
            nn.Tanh(),
            nn.Linear(in_features=fc_dim // 2, out_features=output_dim, device=self.device),
        )

    def forward(
        self, inputs: Tensor, hidden_init: Tensor = None, cell_init: Tensor = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters:
        -----------
        inputs: Tensor (window_size, batch_size, input_dim)
        hidden_init: Tensor (batch_size, hidden_dim)
        cell_init: Tensor (batch_size, hidden_dim)
        """

        batch = inputs.shape[1]
        if self.attn_layer:
            hiddens = []

        # initialise hidden state and cell state
        hidden, cell = self._init_hidden(hidden_init, cell_init, batch)

        # pass through LSTM layer
        for input in inputs:
            hidden, cell = self.lstm_cell(input, (hidden, cell))

            # record hidden states for attention layer
            if self.attn_layer:
                hiddens.append(hidden)

        if self.attn_layer:
            # forward pass through attention layer and fully connected layer
            output = self.output(self.attn(torch.stack(hiddens, dim=1)))
        else:
            # forward pass through fully connected layer
            output = self.output(hidden)

        return output, hidden, cell

    def _init_hidden(
        self, hidden: Tensor, cell: Tensor, batch: int
    ) -> tuple[Tensor, Tensor]:
        if hidden is not None and cell is not None:
            hidden = hidden
            cell = cell
        else:
            hidden = torch.zeros(batch, self.hidden_dim, device=self.device)
            cell = torch.zeros(batch, self.hidden_dim, device=self.device)

        return hidden, cell

    def _set_device(self, device_type: str) -> torch.device:
        if device_type.lower() == "mps":
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif device_type.lower() == "cuda":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        return device

    def record_model(self):
        self.best_model = deepcopy(self.state_dict())

    def save_model(self, file_name: str = "lstm.pth"):
        model_folder_path = "./trained_model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)

        target_state_dict = getattr(self, "best_model", self.state_dict())

        torch.save(target_state_dict, file_name)
