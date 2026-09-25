import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from path_manager import NEW_NORMALIZED_DATASET, NEW_DATASET_PATH, CONVERTED_DATASET_PATH
from src.ai.config import LEARNING_RATE, BATCH_SIZE, EPOCHS, TRAIN_TEST_SPLIT, MAX_SHOTS, SHOT
from src.ai.criterion import ArrowCriterion
from src.ai.dataset import ArcheryDataset
from src.ai.model import ArcheryResNet
from src.ai.splits import expanded_sample_indices
from src.ai.transform import CustomAugmentation
from src.ai.training_config import TrainingConfig, load_training_config, set_training_seed
from src.ai.ui import show_history
from src.ai.utils import get_device, collate_fn, save_model


# ==== Train ====
def train(data_dir, json_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, config: TrainingConfig | None = None):
    config = config or TrainingConfig(epochs=epochs, batch_size=batch_size, learning_rate=lr)
    set_training_seed(config.seed)

    dataset = ArcheryDataset(data_dir, json_dir, aug_transform=CustomAugmentation(), num_aug=config.num_aug)
    print("Всего изображений:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError(f"В '{data_dir}' нет картинок!")

    train_base_idx, test_base_idx = train_test_split(
        range(dataset.base_len),
        test_size=config.train_test_split,
        random_state=config.seed
    )
    train_idx = expanded_sample_indices(train_base_idx, dataset.base_len, dataset.num_aug, include_augmented=True)
    test_idx = expanded_sample_indices(test_base_idx, dataset.base_len, dataset.num_aug, include_augmented=False)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    loader_train = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    loader_val = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    device = get_device()
    model = ArcheryResNet().to(device)
    criterion = ArrowCriterion(alpha=config.loss_alpha, beta=config.loss_beta)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_history = []
    test_history = []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for imgs, coords in loader_train:
            imgs, coords = imgs.to(device), coords.to(device)
            for j in range(len(coords)):
                for i in range(0, len(coords[j]), 3):
                    if coords[j][i] < 0.5 * SHOT:
                        coords[j][i + 1] = 0
                        coords[j][i + 2] = 0
                    else:
                        coords[j][i] = SHOT
            preds = model(imgs)
            loss = criterion(preds, coords)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_history.append(total_loss / len(loader_train))

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, coords in loader_val:
                imgs, coords = imgs.to(device), coords.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, coords).item()

        test_history.append(val_loss / len(loader_val))
        print(f"[Val]   Epoch {epoch + 1}/{config.epochs}, Loss: {val_loss / len(loader_val):.4f}")

    save_model(model)

    show_history(train_history, test_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    train_config = load_training_config(args.config) if args.config else None
    train(CONVERTED_DATASET_PATH, NEW_NORMALIZED_DATASET, config=train_config)
