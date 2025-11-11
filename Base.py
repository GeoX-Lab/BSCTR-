from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class ToolNode:
    name: str
    description: str
    inputs: List[str]
    outputs: List[str]
    feedback: List[str] = None
    vector: np.ndarray = None

@dataclass
class ToolEdge:
    start_node: str
    end_node: str
    messages: List[str]
    status: int
    timestamp: str
    weights: float = 0.01
