"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import argparse

import torch

from torchvision import transforms

import data_setup, engine, model_builder, utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dir",
        default="data/pizza_steak_sushi/train",
        type=str,
        help="training directory",
    )
    parser.add_argument(
        "--test_dir",
        default="data/pizza_steak_sushi/test",
        type=str,
        help="testing directory",
    )
    parser.add_argument(
        "--batch_size", default=32, type=int, help="batch size of the data"
    )
    parser.add_argument(
        "--learning_rate",
        default=0.01,
        type=float,
        help="learning rate of the optimizer",
    )
    parser.add_argument(
        "--num_epochs",
        default=5,
        type=int,
        help="number of epochs (iterations) the model will train",
    )
    parser.add_argument(
        "--hidden_units",
        default=10,
        type=int,
        help="number of hidden units in the model architecture",
    )

    args = parser.parse_args()

    # setup hyperparameters
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate

    # setup directories
    train_dir = args.train_dir
    test_dir = args.test_dir

    # setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # create transforms
    data_transform = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    )

    # create dataloaders with the help of data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE,
    )

    # create model with the help of model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3, hidden_units=HIDDEN_UNITS, output_shape=len(class_names)
    ).to(device)

    # set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # start training with help from engine.py
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device,
    )

    # save the model with help from utils.py
    utils.save_model(
        model=model, target_dir="models", model_name="going_modular_model.pth"
    )
