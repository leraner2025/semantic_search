from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class Entity:
    entity_id: str
    text: str
    emb: Optional[np.ndarray] = None
    sim_to_query: Optional[float] = None
    top_cuis: Optional[List[Tuple[str, float]]] = None

    @property
    def cui(self):
        return self.top_cuis[0][0] if self.top_cuis else None
