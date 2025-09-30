from pydantic import BaseModel


class DetectionTreeAnalysis(BaseModel):
    is_detection_good: bool
    reason: str


class ClassificationTreeAnalysis(BaseModel):
    tree_type: str
    has_hollow: int
    has_cracks: int
    injuries: str
    has_fruits_or_flowers: int
