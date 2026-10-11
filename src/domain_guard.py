"""Lightweight CLIP zero-shot domain gate for app.py -- checks whether an
uploaded image plausibly looks like an aerial/outdoor scene before running
the (aerial-only-trained) waste detector on it. UI-layer sanity check only,
not part of the research pipeline/contribution.
"""

from __future__ import annotations

import dataclasses

import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

CANDIDATE_LABELS = [
    "an aerial or top-down photo of an outdoor area",
    "a photo of a computer desktop or screenshot",
    "a photo of a person or people",
    "an indoor photo",
    "a photo of text or a document",
    "a photo of an animal",
]
AERIAL_LABEL = CANDIDATE_LABELS[0]
DEFAULT_THRESHOLD = 0.4


@dataclasses.dataclass
class ClipBundle:
    model: object
    preprocess: object
    device: str
    text_features: torch.Tensor  # one row per CANDIDATE_LABELS entry, precomputed once


def load_clip_model() -> ClipBundle:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()

    with torch.no_grad():
        tokens = tokenizer(CANDIDATE_LABELS).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return ClipBundle(model=model, preprocess=preprocess, device=device, text_features=text_features)


@dataclasses.dataclass
class DomainResult:
    top_label: str
    top_score: float
    aerial_score: float
    scores: dict[str, float]
    is_aerial: bool


@torch.no_grad()
def classify_domain(
    bundle: ClipBundle, image: Image.Image, threshold: float = DEFAULT_THRESHOLD
) -> DomainResult:
    image_input = bundle.preprocess(image.convert("RGB")).unsqueeze(0).to(bundle.device)
    image_features = bundle.model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ bundle.text_features.T).softmax(dim=-1)[0]
    scores = {label: float(score) for label, score in zip(CANDIDATE_LABELS, similarity)}

    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]
    aerial_score = scores[AERIAL_LABEL]
    is_aerial = top_label == AERIAL_LABEL and aerial_score >= threshold

    return DomainResult(
        top_label=top_label,
        top_score=top_score,
        aerial_score=aerial_score,
        scores=scores,
        is_aerial=is_aerial,
    )
