from path_manager import NEW_NORMALIZED_DATASET, CONVERTED_DATASET_PATH
from neural_network.config import OUTPUT_DIM
from model_converter.convert_pth import convert
from neural_network.dataset import ArcheryDataset
from neural_network.model import ArcheryResNet
from neural_network.visualize import visualize_model
from neural_network.train import train
from neural_network.transform import CustomAugmentation
from neural_network.utils import get_device, load_model

if __name__ == '__main__':
    train(NEW_NORMALIZED_DATASET, NEW_NORMALIZED_DATASET)
    device = get_device()
    dataset = ArcheryDataset(NEW_NORMALIZED_DATASET, NEW_NORMALIZED_DATASET,
                             aug_transform=CustomAugmentation(), num_aug=4)
    model = ArcheryResNet(OUTPUT_DIM).to(device)
    load_model(model, device)

    visualize_model(model, dataset, device)
    convert()
