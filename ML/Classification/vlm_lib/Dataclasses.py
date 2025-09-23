from pydantic import BaseModel

class TreeAnalysis(BaseModel):
    tree_type: str
    has_hollow: int
    has_cracks: int
    injuries: str
    has_fruits_or_flowers: int