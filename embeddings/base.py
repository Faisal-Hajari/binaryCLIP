"""Abstract interface for CLIP embedding backends."""

from abc import ABC, abstractmethod

import torch
from torchvision.transforms import Compose


class ClipEmbedder(ABC):
    """Common interface for text/image encoding across CLIP libraries."""

    @property
    @abstractmethod
    def device(self) -> str:
        ...

    @property
    @abstractmethod
    def preprocess(self) -> Compose:
        ...

    @abstractmethod
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text prompts. Returns normalized embeddings of shape (N, d)."""
        ...

    @abstractmethod
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images. Returns normalized embeddings of shape (B, d)."""
        ...
