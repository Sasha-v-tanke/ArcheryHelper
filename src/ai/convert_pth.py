import torch
from PIL import Image
from torchvision import transforms

from path_manager import MODELS, CONVERTED_DATASET_PATH
from src.ai.config import OUTPUT_DIM
from src.ai.model import ArcheryResNet
from src.ai.model_metadata import write_model_metadata, write_sha256
from src.ai.utils import load_model


def convert():
    device = torch.device('cpu')

    model = ArcheryResNet(OUTPUT_DIM)
    load_model(model, device)
    model.eval()

    img = Image.open(CONVERTED_DATASET_PATH + "/30.jpeg").convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    traced = torch.jit.trace(model, img_tensor)

    traced.save(MODELS + "/model.pt")
    # optimized_traced = optimize_for_mobile(traced)

    # Сохраняем Lite модель
    model_path = MODELS + "/model.ptl"
    traced._save_for_lite_interpreter(model_path)
    write_model_metadata(MODELS)
    write_sha256(model_path, MODELS)


if __name__ == '__main__':
    convert()
