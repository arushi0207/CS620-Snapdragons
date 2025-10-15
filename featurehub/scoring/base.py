from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ScoreModel(ABC):
    """Interface for models that convert extracted features into discrete scores."""

    @abstractmethod
    def predict(self, records: List[Dict[str, Any]]) -> int:
        """Return an integer score between 1 and 10 (inclusive)."""
        raise NotImplementedError
