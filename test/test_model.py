import torch

from src.lstm.models import Attention, LSTMNetwork


def test_lstm():

    # Test LSTM model
    batch_size = 8
    window_size = 12

    # model parameters
    input_dim = 10  # feature size
    hidden_dim = 64
    fc_dim = 16
    attn_layer = False
    output_dim = 1
    device = "cpu"

    lstm = LSTMNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        fc_dim=fc_dim,
        output_dim=output_dim,
        attn_layer=attn_layer,
        stateful=False,
        device=device,
    )

    # batch
    input = torch.randn(window_size, batch_size, input_dim)
    hidden_in = torch.randn(batch_size, hidden_dim)
    cell_in = torch.randn(batch_size, hidden_dim)

    output, hidden_out, cell_out = lstm(input, hidden_in, cell_in)

    assert output.shape == (batch_size, output_dim)
    assert hidden_out.shape == (batch_size, hidden_dim)
    assert cell_out.shape == (batch_size, hidden_dim)

    # unbatch
    input = torch.randn(window_size, input_dim)
    hidden_in = torch.randn(hidden_dim)
    cell_in = torch.randn(hidden_dim)

    output, hidden_out, cell_out = lstm(input, hidden_in, cell_in)

    assert output.shape == (output_dim,)
    assert hidden_out.shape == (hidden_dim,)
    assert cell_out.shape == (hidden_dim,)


def test_attention():
    window_size = 10
    batch_size = 8
    hidden_dim = 64
    device = "cpu"

    attention_layer = Attention(features_dim=hidden_dim, device=device)

    # batch
    h_s_batch = torch.randn(batch_size, window_size, hidden_dim)
    context_batch = attention_layer(h_s_batch)
    assert context_batch.shape == (batch_size, hidden_dim)

    # unbatch
    h_s_unbatch = torch.randn(window_size, hidden_dim)
    context_unbatch = attention_layer(h_s_unbatch)
    assert context_unbatch.shape == (hidden_dim,)


def test_lstm_attn():
    window_size = 48
    batch_size = 8
    input_dim = 10  # feature size
    hidden_dim = 64
    fc_dim = 16
    output_dim = 1
    attn_layer = True
    device = "cpu"

    lstm = LSTMNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        fc_dim=fc_dim,
        output_dim=output_dim,
        attn_layer=attn_layer,
        stateful=False,
        device=device,
    )

    # batch
    input = torch.randn(window_size, batch_size, input_dim)
    hidden_in = torch.randn(batch_size, hidden_dim)
    cell_in = torch.randn(batch_size, hidden_dim)

    output, hidden_out, cell_out = lstm(input, hidden_in, cell_in)

    assert output.shape == (batch_size, output_dim)
    assert hidden_out.shape == (batch_size, hidden_dim)
    assert cell_out.shape == (batch_size, hidden_dim)

    # unbatch
    input = torch.randn(window_size, input_dim)
    hidden_in = torch.randn(hidden_dim)
    cell_in = torch.randn(hidden_dim)

    output, hidden, cell = lstm(input, hidden_in, cell_in)

    assert output.shape == (output_dim,)
    assert hidden.shape == (hidden_dim,)
    assert cell.shape == (hidden_dim,)
