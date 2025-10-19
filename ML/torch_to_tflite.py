from ML.Classification.torch_lib.ClassificationWrappers import MultiHeadCNNWrapper
from configs.train_config_classification import TrainConfigs
import torch
import ai_edge_torch

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

torch_model.load_state_dict(torch.load(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\results\saved_models\best_all_classes_53\augs_mobilevit_trees_27class.pth"))
torch_model.eval()

dummy_input = torch.randn(1, 3, 320, 320)

edge_model = ai_edge_torch.convert(torch_model, dummy_input)
edge_model.export(r"C:\Users\shari\PycharmProjects\MegaDendrologAI\ML\Classification\results\saved_models\best_all_classes_53\augs_mobilevit_trees_27class.tflite")