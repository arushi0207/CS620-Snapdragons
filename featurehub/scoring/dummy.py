from __future__ import annotations

from typing import Any, Dict, List

from .base import ScoreModel


class DummyScoreModel(ScoreModel):
    """Placeholder scorer that returns a fixed mid-scale score.

    Replace this with a real model that consumes the FaceMap features.
    """

    def predict(self, records: List[Dict[str, Any]]) -> int:
        # TODO: replace with actual scoring logic.
        return 5
