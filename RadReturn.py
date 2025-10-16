from datetime import datetime

import numpy as np

class RadReturn():
    LLA: np.ndarray
    t: datetime

    def __init__(self, LLA, t):
        self.LLA = LLA
        self.t = t

    def __repr__(self) -> str:
        return f"[{self.LLA[0]}, {self.LLA[1]}, {self.LLA[2]}], {self.t}"
