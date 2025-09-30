from pydantic import BaseModel


class DetectionTreeAnalysis(BaseModel):
    is_detection_good: bool
    reason: str


class ClassificationTreeAnalysis(BaseModel):
    tree_type: str
    has_hollow: int
    has_cracks: int
    injuries: str
    overall_condition: str
    dry_branch_percentage: str
    has_crown_damage: int
    has_trunk_damage: int
    has_rot: int
    has_fruits_or_flowers: int
