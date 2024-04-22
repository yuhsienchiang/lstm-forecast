import matplotlib.pyplot as plt
from models import LSTMNetwork
from trainer import LSTMTrainer

from data import INSTANCE_KEYS, LSTMDataset


def run():
    # data
    train_data_src = "../data/training_data.csv"
    valid_data_src = "../data/validation_data.csv"
    window_size = 48  # 4 hour ->48 5-min intervals

    # model architecture
    input_dim = len(INSTANCE_KEYS)
    hidden_dim = 64
    fc_dim = 16
    output_dim = 1
    attn_layer = False
    stateful = True # if True, remember to set shuffle to False and batch_size to value equal to window_size
    device = "mps"

    # training parameters
    batch_size = 48
    drop_last = True
    learn_rate = 1e-6
    epochs = 25
    optimizer = "adamw"
    loss_fn = "mse"
    shuffle = False
    quiet = False
    model_file_name = "lstm_stateful.pth"

    # dataset
    train_dataset = LSTMDataset(data_src=train_data_src, window_size=window_size)
    valid_dataset = LSTMDataset(data_src=valid_data_src, window_size=window_size)

    # model
    lstm = LSTMNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        fc_dim=fc_dim,
        output_dim=output_dim,
        attn_layer=attn_layer,
        stateful=stateful,
        device=device,
    )

    # trainer
    trainer = LSTMTrainer(
        model=lstm,
        optimizer=optimizer,
        loss_fn=loss_fn,
        learn_rate=learn_rate,
        file_name=model_file_name,
    )

    # training
    train_history, valid_history = trainer.train(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        epochs=epochs,
        drop_last=drop_last,
        quiet=quiet,
    )

    # plot outcome
    plt.figure(figsize=(32, 18), dpi=300)
    plt.plot(range(len(train_history)), train_history, label="Train Loss")
    plt.legend()
    plt.savefig("loss_stateful_5.png")

    plt.figure(figsize=(32, 18), dpi=300)
    plt.plot(range(len(valid_history)), valid_history, label="Valid Loss")
    plt.legend()
    plt.savefig("valid_loss_stateful_5.png")

if __name__ == "__main__":
    run()
