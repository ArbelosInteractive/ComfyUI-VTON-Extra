import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nodes  # noqa: E402


def load_img(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    return torch.from_numpy(arr[None, ...])


def main():
    base = ROOT / 'examples'
    collage = load_img(base / 'collage.png')
    selfie = load_img(base / 'selfie.png')
    tryon = load_img(base / 'tryon.png')
    loader = nodes.VLMRubricLoadNode()
    rubric, _ = loader.load(collage, str(base / 'vton_vlm_rubric.json'), '')
    face = nodes.VLMFaceIdentityScoreNode().score_face(selfie, tryon, rubric)
    garments = nodes.VLMGarmentScoreNode().score_garments(selfie, collage, tryon, rubric)
    body = nodes.VLMBodyShapeScoreNode().score_body(selfie, tryon, rubric)
    lighting = nodes.VLMLightingScoreNode().score_lighting(selfie, tryon, rubric)
    background = nodes.VLMBackgroundScoreNode().score_background(tryon, rubric)
    aggregate = nodes.VTONAggregateRubricNode().aggregate(rubric, face[0], garments[0], body[0], lighting[0], background[0])
    out = {
        'face_score': face[1],
        'garments_score': garments[1],
        'body_score': body[1],
        'lighting_score': lighting[1],
        'background_score': background[1],
        'final_score': aggregate[0],
        'summary': aggregate[2],
        'garment_debug': garments[0].get('debug', {}),
    }
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
