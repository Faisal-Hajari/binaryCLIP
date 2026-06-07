"""NegationCLIP backend — OpenAI CLIP architecture with fine-tuned weights."""

import torch
import torch.nn.functional as F
import clip
from torchvision.transforms import Compose

from .base import ClipEmbedder


class NegationClipEmbedder(ClipEmbedder):
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        weights: str = "negationclip_ViT-B32.pth",
        device: str | None = None,
    ):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._preprocess = clip.load(model_name, device=self._device)
        state = torch.load(weights, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

    @property
    def device(self) -> str:
        return self._device

    @property
    def preprocess(self) -> Compose:
        return self._preprocess

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts).to(self._device)
        embs = self._model.encode_text(tokens)
        return F.normalize(embs, dim=-1)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self._device)
        embs = self._model.encode_image(images)
        return F.normalize(embs, dim=-1)
