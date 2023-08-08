import cProfile
import pstats
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from clearml import Logger, Task
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy, FSDPStrategy
from simple_parsing import ArgumentParser
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split
from torchsummary import summary
from tqdm import tqdm

from config import Args, config, logger
from datasets import *
from siamese import EfficientNet, MobileNet, MyNet, TripletLoss


# Create train and eval functions
def train(model, dataloader, optimizer, criterion, fabric):
    model.train()
    total_loss = 0.0

    for a, p, n in tqdm(dataloader):
        a_emb = model(a)
        p_emb = model(p)
        n_emb = model(n)
        optimizer.zero_grad()
        loss = criterion(a_emb, p_emb, n_emb)
        fabric.backward(loss)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss


def eval(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for a, p, n in tqdm(dataloader):
            a_emb = model(a)
            p_emb = model(p)
            n_emb = model(n)
            loss = criterion(a_emb, p_emb, n_emb)
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)

        return avg_loss


# Sanity check subset function
def create_subset(trainset, valset):
    train_subset = Subset(trainset, range(50))
    val_subset = Subset(valset, range(10))

    return train_subset, val_subset


if __name__ == "__main__":
    # device = set_device()
    fabric = Fabric(accelerator="cuda", devices=2, strategy="deepspeed")
    fabric.launch()
    fabric.seed_everything(17)

    # Initialize model
    logger.info("Initializing model...")
    model = MyNet()
    # model.to(device)
    x = torch.rand(1, 3, 256, 128)
    summary(model, x)

    # Read args
    logger.info("Reading arguments...")
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="options")
    args_namespace = parser.parse_args()
    args = args_namespace.options
    logger.info(args)

    # Define loss and optimizer
    # criterion = TripletLoss(margin=1.0, soft=True)
    criterion = nn.TripletMarginLoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    torch.autograd.set_detect_anomaly(True)

    # Prepare dataset
    logger.info("Preparing dataset...")

    # Define transforms
    color_jitter = transforms.ColorJitter(hue=0.05, saturation=0.05)
    random_horizontal_flip = transforms.RandomHorizontalFlip()
    random_rotation = transforms.RandomRotation(
        20, interpolation=transforms.InterpolationMode.BILINEAR
    )

    transforms = transforms.Compose(
        [
            transforms.Resize((256, 128)),
            # transforms.RandomApply([color_jitter], p=0.2),
            # transforms.RandomApply([random_horizontal_flip], p=0.2),
            # transforms.RandomApply([random_rotation], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset_msmt17 = SiameseMSMT17Dataset(
        data_dir=config.msmt17_dir, transform=transforms
    )
    logger.info(f"Total MSMT17 images: {len(dataset_msmt17)}")

    dataset_market1501 = SiameseMarket1501Dataset(
        data_dir=config.market1501_dir, transform=transforms
    )
    logger.info(f"Total Market1501 images: {len(dataset_market1501)}")

    dataset_cuhk03 = SiameseCuhk03Dataset(
        data_dir=config.cuhk03_dir, transform=transforms
    )
    logger.info(f"Total CUHK03 images: {len(dataset_cuhk03)}")

    dataset_dukemtmcreid = SiameseDukemtmcreidDataset(
        data_dir=config.dukemtmcreid_dir, transform=transforms
    )
    logger.info(f"Total DukeMTMCReID images: {len(dataset_dukemtmcreid)}")
    combined_dataset = ConcatDataset(
        [
            dataset_msmt17,
            dataset_market1501,
            dataset_cuhk03,
            dataset_dukemtmcreid,
        ]
    )

    train_set, val_set = random_split(combined_dataset, [0.95, 0.05])

    # Create subset (sanity check)
    train_set, val_set = create_subset(train_set, val_set)

    logger.info(f"Total training images: {len(train_set)}")
    logger.info(f"Total validation images: {len(val_set)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(
        train_loader, val_loader
    )

    # Create clearml task
    logger.info("Initializing ClearML Task...")
    task = Task.init(
        project_name="3i-Siamese",
        task_name=f"siamese-mynet-combined-{datetime.now()}",
        output_uri="/home/jovyan/haseeb-rnd/haseeb-data/artifacts/siamese/",
    )
    task.connect(args, name="hyperparameters")  # add args to clearml logging

    # Star training loop
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    logger.info(f"Starting training...")
    for i in range(args.epochs):
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        logger.info(f"date and time = {dt_string}")
        train_loss = train(model, train_loader, optimizer, criterion, fabric)
        logger.info(f"EPOCHS: {i+1} train_loss: {train_loss}")
        if i % args.val_interval == 0:
            val_loss = eval(model, val_loader, criterion)
            logger.info("Evaluation...")
            logger.info(f"EPOCHS: {i+1} val_loss: {val_loss}")

            state = {"model": model, "optimizer": optimizer, "iteration": i+1}

            # Check and save best model
            if val_loss + args.min_delta < best_val_loss:
                fabric.save("checkpoint", state)
                best_val_loss = val_loss
                epochs_without_improvement = 0
                logger.info("Saved weights successfully!")
            else:
                epochs_without_improvement += 1

            # Check if early stopping criteria are met
            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping at epoch {i + 1}")
                break

        scheduler.step()
        logger.info(f"learning rate: {optimizer.param_groups[0]['lr']}")

        Logger.current_logger().report_scalar(
            "train", "loss", iteration=i, value=train_loss
        )
        Logger.current_logger().report_scalar(
            "valid",
            "loss",
            iteration=i,
            value=val_loss,
        )
