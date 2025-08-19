import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from path_manager import NEW_NORMALIZED_DATASET, MODELS
from src.ai.config import LEARNING_RATE, BATCH_SIZE, EPOCHS, TRAIN_TEST_SPLIT, MAX_SHOTS
from src.ai.dataset import ArcheryDataset
from src.ai.model import ArcheryResNet
from src.ai.transform import CustomAugmentation
from src.ai.ui import show_history
from src.ai.utils import get_device, collate_fn, save_model


# ==== Train ====
def train(data_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    dataset = ArcheryDataset(data_dir, transform=CustomAugmentation())
    print("Всего изображений:", len(dataset))

    if len(dataset) == 0:
        raise RuntimeError(f"В '{data_dir}' нет картинок!")

    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=TRAIN_TEST_SPLIT, random_state=42)
    train_set = torch.utils.data.Subset(dataset, train_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)

    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    loader_val = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    output_dim = MAX_SHOTS * 2
    device = get_device()
    model = ArcheryResNet(output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_history = []
    test_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, coords in loader_train:
            imgs, coords = imgs.to(device), coords.to(device)
            preds = model(imgs)
            loss = criterion(preds, coords)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_history.append(total_loss / len(loader_train))
        # print(f"[Train] Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader_train):.4f}")

        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, coords in loader_val:
                imgs, coords = imgs.to(device), coords.to(device)
                preds = model(imgs)
                val_loss += criterion(preds, coords).item()

        test_history.append(val_loss / len(loader_val))
        print(f"[Val]   Epoch {epoch + 1}/{epochs}, Loss: {val_loss / len(loader_val):.4f}")

    save_model(model)

    show_history(train_history, test_history)


if __name__ == "__main__":
    train(NEW_NORMALIZED_DATASET)
