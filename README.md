# ComfyUI VTON VLM QC

This package replaces the CV-first try-on QC scorers with VLM-based specialist scorers while preserving the same high-level interface:
- selfie image
- collage image
- try-on result image
- rubric JSON

## Nodes
- `VLMRubricLoadNode`
- `VLMFaceIdentityScoreNode`
- `VLMGarmentScoreNode`
- `VLMBodyShapeScoreNode`
- `VLMLightingScoreNode`
- `VLMBackgroundScoreNode`
- `VTONAggregateRubricNode`
- `VTONSaveJSONReportNode`

## Rubric additions
Top-level fields:
- `background_intent`
- `lighting_model`

Override block:
- `vlm_overrides.background`
- `vlm_overrides.lighting`
- `vlm_overrides.body_shape`
- `vlm_overrides.garments_total`
- `vlm_overrides.face_identity`

Each override supports:
- `prompt_override`
- `prompt_append`

## VLM config
In `options.vlm`:
- `model`
- `detail`
- `temperature`
- `max_output_tokens`
- `fail_on_api_error`
- `mock`

Use `mock: true` for local package validation without a live API call.

## Install
Copy this folder under `ComfyUI/custom_nodes/ComfyUI-VTON-VLM` and restart ComfyUI.

## Live use
Ensure your OpenAI API key is available to the ComfyUI process, typically through `OPENAI_API_KEY`.

## Notes
- The garment scorer is inventory-driven.
- If inventory is empty, it tries local OCR fallback on collage labels like `hat`, `shirt`, `pants`, `skirt`, `dress`, `shoes`, and `purse`.
- Missing required garments can hard-fail the final score if enabled in the rubric.
