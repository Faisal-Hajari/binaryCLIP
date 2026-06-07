"""OpenCLIP backend (https://github.com/mlfoundations/open_clip)."""

import torch
import torch.nn.functional as F
import open_clip
from torchvision.transforms import Compose

from .base import ClipEmbedder


class OpenClipEmbedder(ClipEmbedder):
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str | None = None,
    ):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model = self._model.to(self._device).eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)

    @property
    def device(self) -> str:
        return self._device

    @property
    def preprocess(self) -> Compose:
        return self._preprocess

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self._tokenizer(texts).to(self._device)
        embs = self._model.encode_text(tokens)
        return F.normalize(embs, dim=-1)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self._device)
        embs = self._model.encode_image(images)
        return F.normalize(embs, dim=-1)
