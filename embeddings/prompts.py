"""Text prompt embedding helpers for binary presence detection."""

import torch
import torch.nn.functional as F

from .base import ClipEmbedder

POSITIVE_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "an image containing a {}.",
    "a {} in the image.",
]

NEGATIVE_TEMPLATES = [
    "a photo with no {}.",
    "a picture without a {}.",
    "an image containing no {}.",
    "no {} in the image.",
]


def build_class_text_embeddings(
    embedder: ClipEmbedder,
    classname: str,
    positive_templates: list[str] | None = None,
    negative_templates: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (pos_emb, neg_emb) each of shape (d,), averaged over prompt templates.
    """
    pos_templates = positive_templates or POSITIVE_TEMPLATES
    neg_templates = negative_templates or NEGATIVE_TEMPLATES

    def encode_templates(templates: list[str]) -> torch.Tensor:
        prompts = [t.format(classname) for t in templates]
        embs = embedder.encode_text(prompts)          # (n_templates, d)
        embs = embs.mean(dim=0)                       # (d,)
        return F.normalize(embs, dim=-1)

    return encode_templates(pos_templates), encode_templates(neg_templates)
