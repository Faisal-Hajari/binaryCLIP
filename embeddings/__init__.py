"""CLIP embedding backends and factory."""

from .base import ClipEmbedder
from .negation_clip_backend import NegationClipEmbedder
from .open_clip_backend import OpenClipEmbedder
from .openai_clip_backend import OpenAIClipEmbedder

BACKENDS: dict[str, type[ClipEmbedder]] = {
    "open_clip": OpenClipEmbedder,
    "openai_clip": OpenAIClipEmbedder,
    "negation_clip": NegationClipEmbedder,
}

DEFAULT_BACKEND = "open_clip"


def load_embedder(
    backend: str = DEFAULT_BACKEND,
    model_name: str | None = None,
    device: str | None = None,
    **kwargs,
) -> ClipEmbedder:
    """
    Load a CLIP embedder from the requested library.

    Args:
        backend: One of "open_clip" or "openai_clip".
        model_name: Model identifier (library-specific). Defaults per backend.
        device: "cuda" or "cpu". Auto-detected if None.
        **kwargs: Extra args passed to the backend constructor
                  (e.g. pretrained="openai" for open_clip).
    """
    if backend not in BACKENDS:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown backend {backend!r}. Available: {available}")

    cls = BACKENDS[backend]
    if model_name is None:
        model_name = "ViT-L-14" if backend == "open_clip" else "ViT-L/14"

    return cls(model_name=model_name, device=device, **kwargs)


__all__ = [
    "ClipEmbedder",
    "NegationClipEmbedder",
    "OpenClipEmbedder",
    "OpenAIClipEmbedder",
    "BACKENDS",
    "DEFAULT_BACKEND",
    "load_embedder",
]
