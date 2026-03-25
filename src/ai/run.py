from path_manager import NEW_NORMALIZED_DATASET, CONVERTED_DATASET_PATH
from src.ai.config import OUTPUT_DIM
from src.ai.convert_pth import convert
from src.ai.dataset import ArcheryDataset
from src.ai.model import ArcheryResNet
from src.ai.test import visualize_model
from src.ai.train import train
from src.ai.transform import CustomAugmentation
from src.ai.utils import get_device, load_model

if __name__ == '__main__':
    train(NEW_NORMALIZED_DATASET, NEW_NORMALIZED_DATASET)
    device = get_device()
    dataset = ArcheryDataset(NEW_NORMALIZED_DATASET, NEW_NORMALIZED_DATASET,
                             aug_transform=CustomAugmentation(), num_aug=4)
    model = ArcheryResNet(OUTPUT_DIM).to(device)
    load_model(model, device)

    visualize_model(model, dataset, device)
    convert()
