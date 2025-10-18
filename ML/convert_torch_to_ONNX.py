from ultralytics import YOLO
from configs.paths import PathConfig
from ML.Classification.torch_lib.ClassificationWrappers import MultiHeadCNNWrapper
from configs.train_config_classification import TrainConfigs
import torch

# path_to_yolo_weights = r"C:\Users\shari\Downloads\yolo11m_on_new_data\best.pt"

# model = YOLO(path_to_yolo_weights)

# model.export(format='onnx', int8=False)


CURRENT_CONFIG = TrainConfigs.TreeClassificationWithMobileTransformer
task_names = list(CURRENT_CONFIG.TARGET_JSON_FIELD.keys())
num_classes_per_task = CURRENT_CONFIG.TARGET_JSON_FIELD

torch_model = MultiHeadCNNWrapper(
    backbone_model=CURRENT_CONFIG.MODEL_NAME,
    backbone_type=CURRENT_CONFIG.BACKBONE_TYPE,
    num_output_features=num_classes_per_task,
    hidden_size=128,
    dropout=0.4
)

torch_model.load_state_dict(torch.load(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\results\saved_models\best_tree_type_apple_vit_0.35\APPLE_XS_TRANSFORMER_TREE_TYPE_WEB_DATA.pth"))
torch_model.eval()

dummy_input = torch.randn(1, 3, 320, 320)

torch.onnx.export(
    model=torch_model,
    args=dummy_input,
    f=r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\results\saved_models\best_tree_type_apple_vit_0.35\APPLE_XS_TRANSFORMER_TREE_TYPE_WEB_DATA.onnx",
    input_names=['input'],
    output_names=task_names,
    dynamo=True
)
