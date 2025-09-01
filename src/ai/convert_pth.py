import torch
from path_manager import MODELS
from src.ai.config import OUTPUT_DIM, IMG_SIZE
from src.ai.model import ArcheryResNet
from src.ai.utils import load_model, get_device
from torch.utils.mobile_optimizer import optimize_for_mobile


def convert():
    device = torch.device('cpu')

    model = ArcheryResNet(OUTPUT_DIM)
    load_model(model, device)
    model.eval()

    example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    traced = torch.jit.trace(model, example_input)

    optimized_traced = optimize_for_mobile(traced)

    # Сохраняем Lite модель
    optimized_traced._save_for_lite_interpreter(MODELS + "/model.ptl")


if __name__ == '__main__':
    convert()
