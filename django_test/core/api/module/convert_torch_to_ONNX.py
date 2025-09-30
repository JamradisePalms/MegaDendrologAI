from ultralytics import YOLO
from configs.paths import PathConfig
from ML.Classification.torch_lib.ResNetWrapper import ResNetWrapper
from configs.train_config_classification import TrainConfigs
import torch

path_to_yolo_weights = r"C:\Users\shari\Downloads\yolo11m_on_new_data\best.pt"

model = YOLO(path_to_yolo_weights)

model.export(format='onnx', int8=False)


CURRENT_CONFIG = TrainConfigs.TreeClassificationModelWithMultiHeadMLP
task_names = list(CURRENT_CONFIG.TARGET_JSON_FIELD.keys())
num_classes_per_task = CURRENT_CONFIG.TARGET_JSON_FIELD

torch_model = ResNetWrapper(
    resnet_model=CURRENT_CONFIG.MODEL_NAME,
    num_output_features=num_classes_per_task,
    freeze_resnet=False
)

torch_model.load_state_dict(torch.load(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\results\saved_models\hollow_classification_small.pth"))
torch_model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model=torch_model,
    args=dummy_input,
    f=r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\results\saved_models\hollow_classification_small.onnx",
    input_names=['input'],
    output_names=task_names,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},  # Optional: for dynamic batch size
    opset_version=12,
)
