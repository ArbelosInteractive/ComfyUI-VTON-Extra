
import base64
import copy
import io
import json
import os
import re
import uuid
import difflib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

try:
    import json5  # type: ignore
except Exception:
    json5 = None

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

PACKAGE_CATEGORY = "Arbelos/VTON VLM QC"
EPS = 1e-8

CANONICAL_GARMENT_LABELS = {
    "hat": ["hat", "cap", "cowboy hat", "beanie", "fedora"],
    "shirt": ["shirt", "top", "blouse", "tee", "tshirt", "t-shirt", "jacket"],
    "pants": ["pants", "jeans", "trousers", "leggings"],
    "skirt": ["skirt"],
    "dress": ["dress", "gown"],
    "shoes": ["shoes", "shoe", "boots", "sneakers", "heels", "sandals"],
    "purse": ["purse", "bag", "handbag", "tote"],
}

LOCAL_OCR_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]
KNOWN_LABEL_VARIANTS = sorted(
    {variant for variants in CANONICAL_GARMENT_LABELS.values() for variant in variants}
    | set(CANONICAL_GARMENT_LABELS.keys()),
    key=len,
)

DEFAULT_RUBRIC = {
    "version": "3.2",
    "options": {
        "inventory_mode": "rubric_first_collage_fallback",
        "allow_best_guess_labels": False,
        "use_collage_labels_when_present": True,
        "speed_profile": "balanced",
        "prompt_profile_path": "",
        "recommendation_policy": {
            "accept_min_score": 0.70,
            "review_min_score": 0.60,
            "hard_fail_recommendation": "reject",
        },
        "vlm": {
            "provider": "openai",
            "model": "gpt-5",
            "detail": "high",
            "temperature": 0.0,
            "max_output_tokens": 1400,
            "fail_on_api_error": True,
            "mock": False,
        },
    },
    "background_intent": "",
    "lighting_model": "",
    "vlm_overrides": {
        "background": {"prompt_override": "", "prompt_append": ""},
        "lighting": {"prompt_override": "", "prompt_append": ""},
        "body_shape": {"prompt_override": "", "prompt_append": ""},
        "garments_total": {"prompt_override": "", "prompt_append": ""},
        "face_identity": {"prompt_override": "", "prompt_append": ""},
    },
    "top_level_sections": {
        "face_identity": {"enabled": True, "weight": 0.22},
        "garments_total": {
            "enabled": True,
            "weight": 0.46,
            "hard_fail_on_missing_required": True,
            "default_subsections": {
                "presence": {"enabled": True, "weight": 0.35},
                "shape_structure": {"enabled": True, "weight": 0.30},
                "texture_color_text": {"enabled": True, "weight": 0.25},
                "placement": {"enabled": True, "weight": 0.10},
            },
            "inventory": [],
        },
        "body_shape": {"enabled": True, "weight": 0.14},
        "lighting": {"enabled": True, "weight": 0.10},
        "background": {"enabled": True, "weight": 0.08},
    },
}

DEFAULT_PROMPT_PROFILE = {
    "version": "1.2",
    "sections": {
        "face_identity": {
            "prompt_template": (
                "You are the face consistency scorer for a virtual try-on quality-control system.\n\n"
                "Compare the source selfie and the try-on result only to evaluate whether the try-on preserved the source face region's visual continuity and production quality.\n\n"
                "Return a score from 0.0 to 1.0 for face consistency:\n"
                "- 0.0 = severe facial drift, hallucination, broken structure, or major face-region artifacts\n"
                "- 1.0 = strong preservation of facial structure and appearance continuity with no obvious face-region artifacts\n\n"
                "Evaluate only these factors:\n"
                "- facial structure continuity\n"
                "- facial feature placement stability\n"
                "- skin-tone continuity\n"
                "- preservation of visible face-region details\n"
                "- artifact absence\n"
                "- occlusion and crop quality\n"
                "- whether hairstyle, makeup, accessories, lighting, or expression changes introduced visible facial drift\n\n"
                "Treat this strictly as a try-on face preservation and artifact assessment task.\n\n"
                "Return only valid JSON matching the provided schema."
            )
        },
        "body_shape": {
            "prompt_template": (
                "You are the body shape specialist scorer for a virtual try-on evaluation system.\n\n"
                "Compare the body shape of the person in the selfie and the try-on result.\n\n"
                "Return a score from 0.0 to 1.0:\n"
                "- 0.0 = clearly not the same body shape/body type\n"
                "- 1.0 = clearly the same body shape/body type\n\n"
                "Focus on silhouette, body type, torso/hip/leg proportion cues, and overall body-shape consistency. Ignore pose differences as much as possible and mentally normalize pose before judging silhouette similarity.\n\n"
                "Return only valid JSON matching the provided schema."
            )
        },
        "lighting": {
            "prompt_template": (
                "You are the lighting specialist scorer for a virtual try-on evaluation system.\n\n"
                "Score the lighting of the try-on image against this lighting target:\n{lighting_target}\n\n"
                "Return a score from 0.0 to 1.0:\n"
                "- 0.0 = clearly incorrect\n"
                "- 1.0 = fully matches the intended lighting\n\n"
                "Evaluate light direction, softness/hardness, shadow style, color temperature, contrast, highlight behavior, and subject-scene lighting consistency.\n\n"
                "Return only valid JSON matching the provided schema."
            )
        },
        "background": {
            "prompt_template": (
                "You are the background specialist scorer for a virtual try-on evaluation system.\n\n"
                "Score the background of the try-on image against this background intent:\n{background_target}\n\n"
                "Return a score from 0.0 to 1.0:\n"
                "- 0.0 = not aligned with the required background intent\n"
                "- 1.0 = fully matches the required background intent\n\n"
                "Evaluate scene type, composition, palette, depth/parallax cues, plausibility, and subject-background fit.\n\n"
                "Return only valid JSON matching the provided schema."
            )
        },
        "garments_total": {
            "prompt_template": (
                "You are the garment specialist scorer for a virtual try-on evaluation system.\n\n"
                "Use the collage image as the garment reference source and the try-on image as the output to judge. Score ONLY the garments listed in the rubric inventory below.\n\n"
                "Inventory:\n{inventory_json}\n\n"
                "Missing required garments should trigger hard_fail={hard_fail_enabled} if enabled.\n\n"
                "For each garment, judge the try-on in a perspective-invariant way. Ignore viewpoint mismatch completely. Score garment identity and authenticity based on the reference garment.\n"
                "Evaluate shape, details, affectations, materials, textures, colors, drape, and whether the garment appears on the correct body region.\n"
                "Map those judgments into these four subscores: presence, shape_structure, texture_color_text, placement.\n"
                "Use these subsection weights for the overall garment score: {weights_json}\n\n"
                "Return only valid JSON matching the provided schema."
            )
        },
    },
}


class _PromptSafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _json_dumps(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=False)


def _get_output_dir() -> str:
    if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
        try:
            return folder_paths.get_output_directory()
        except Exception:
            pass
    default_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(default_dir, exist_ok=True)
    return default_dir


def _tensor_to_np(image: torch.Tensor) -> np.ndarray:
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image)
    arr = image.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _normalize_enabled_weights(section_map: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    enabled = []
    for key, section in section_map.items():
        if bool(section.get("enabled", True)) and _safe_float(section.get("weight", 0.0), 0.0) > 0:
            enabled.append((key, _safe_float(section.get("weight", 0.0), 0.0)))
    total = sum(weight for _, weight in enabled)
    if total <= 0:
        return {key: 0.0 for key in section_map.keys()}
    enabled_map = dict(enabled)
    return {key: (enabled_map.get(key, 0.0) / total) for key in section_map.keys()}


def _normalize_subweights(subsections: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    enabled = {k: _safe_float(v.get("weight", 0.0), 0.0) for k, v in subsections.items() if bool(v.get("enabled", True)) and _safe_float(v.get("weight", 0.0), 0.0) > 0}
    total = sum(enabled.values())
    out: Dict[str, float] = {}
    for key in subsections.keys():
        out[key] = enabled.get(key, 0.0) / total if total > 0 else 0.0
    return out


def _report_shell() -> Dict[str, Any]:
    return {
        "final_score": 0.0,
        "hard_fail": False,
        "hard_fail_reasons": [],
        "top_level_scores": {},
        "top_level_confidence": {},
        "top_level_weights_used": {},
        "garments": [],
        "meta": {},
    }


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _normalize_label_text(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", " ", (text or "")).strip().lower()
    if not text:
        return ""
    candidates = []
    for canonical, variants in CANONICAL_GARMENT_LABELS.items():
        if text == canonical:
            return canonical
        if any(v in text for v in variants):
            return canonical
        candidates.extend([canonical] + variants)
    match = difflib.get_close_matches(text, candidates, n=1, cutoff=0.55)
    if match:
        best = match[0]
        for canonical, variants in CANONICAL_GARMENT_LABELS.items():
            if best == canonical or best in variants:
                return canonical
    return text


def _extract_label_tokens(text: str) -> List[str]:
    text = _normalize_label_text(text)
    if not text:
        return []
    out: List[str] = []
    for canonical, variants in CANONICAL_GARMENT_LABELS.items():
        if text == canonical or any(v in text for v in variants):
            out.append(canonical)
    if not out and text:
        out.append(text)
    return out


def _prepare_label_ocr_image(rgb: np.ndarray) -> np.ndarray:
    gray = _rgb_to_gray(rgb)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ys, xs = np.where(bw > 0)
    if len(xs) and len(ys):
        x0, x1 = max(0, int(xs.min()) - 2), min(bw.shape[1], int(xs.max()) + 3)
        y0, y1 = max(0, int(ys.min()) - 2), min(bw.shape[0], int(ys.max()) + 3)
        bw = bw[y0:y1, x0:x1]
    if bw.shape[0] < 18:
        scale = 18.0 / max(1.0, float(bw.shape[0]))
        bw = cv2.resize(bw, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return bw


def _render_text_template(word: str, width: int, height: int) -> np.ndarray:
    best = None
    word = word.lower().strip()
    for font in LOCAL_OCR_FONTS:
        for thickness in (1, 2):
            canvas = np.zeros((height, width), np.uint8)
            scale = 0.2
            best_scale = scale
            while scale <= 3.0:
                (tw, th), baseline = cv2.getTextSize(word, font, scale, thickness)
                if tw <= width - 4 and th + baseline <= height - 4:
                    best_scale = scale
                    scale += 0.05
                else:
                    break
            (tw, th), baseline = cv2.getTextSize(word, font, best_scale, thickness)
            x = max(1, (width - tw) // 2)
            y = max(th + 1, (height + th) // 2 - baseline // 2)
            cv2.putText(canvas, word, (x, y), font, best_scale, 255, thickness, cv2.LINE_AA)
            score = float(canvas.mean())
            if best is None or score > best[0]:
                best = (score, canvas)
    return best[1] if best is not None else np.zeros((height, width), np.uint8)


def _local_known_label_ocr(rgb: np.ndarray) -> str:
    proc = _prepare_label_ocr_image(rgb)
    h, w = proc.shape[:2]
    if h < 6 or w < 6:
        return ""
    best_word = ""
    best_score = -1.0
    proc_bin = (proc > 0).astype(np.uint8)
    for word in KNOWN_LABEL_VARIANTS:
        tpl = _render_text_template(word, w, h)
        tpl_bin = (tpl > 0).astype(np.uint8)
        inter = float(np.logical_and(proc_bin > 0, tpl_bin > 0).sum())
        union = float(np.logical_or(proc_bin > 0, tpl_bin > 0).sum()) + EPS
        iou = inter / union
        corr = cv2.matchTemplate(proc.astype(np.float32), tpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0, 0]
        score = 0.55 * float(max(-1.0, min(1.0, corr))) + 0.45 * iou
        if score > best_score:
            best_score = score
            best_word = word
    return best_word if best_score >= 0.28 else ""


def _connected_components_boxes(mask: np.ndarray, min_area: int = 40) -> List[Tuple[int, int, int, int]]:
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    boxes = []
    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def _detect_label_boxes(collage_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = _rgb_to_gray(collage_rgb)
    bright = (gray > 242).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    boxes = []
    h, w = gray.shape[:2]
    for x0, y0, x1, y1 in _connected_components_boxes(bright, min_area=max(40, (h * w) // 40000)):
        bw = x1 - x0
        bh = y1 - y0
        area = bw * bh
        if bw < 12 or bh < 8:
            continue
        if bw > int(w * 0.35) or bh > int(h * 0.15):
            continue
        border_near = (x0 < w * 0.25 or y0 < h * 0.25 or x1 > w * 0.75 or y1 > h * 0.75)
        crop = collage_rgb[y0:y1, x0:x1]
        dark_ratio = float((_rgb_to_gray(crop) < 120).mean())
        fill_ratio = area / max(1.0, h * w)
        if border_near and dark_ratio > 0.01 and fill_ratio < 0.05:
            pad = 3
            boxes.append((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def _parse_collage_labels(collage_rgb: np.ndarray) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    h, w = collage_rgb.shape[:2]
    for box in _detect_label_boxes(collage_rgb):
        x0, y0, x1, y1 = box
        crop = collage_rgb[y0:y1, x0:x1]
        text = _local_known_label_ocr(crop)
        tokens = _extract_label_tokens(text)
        if not tokens:
            continue
        labels.append({
            "box": box,
            "text": text,
            "canonical_labels": tokens,
            "primary_label": tokens[0],
            "anchor": ((x0 + x1) / 2.0 / max(1.0, w), (y0 + y1) / 2.0 / max(1.0, h)),
        })
    return labels


def _np_to_data_url(rgb: np.ndarray) -> str:
    image = Image.fromarray(rgb)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _get_openai_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed in this environment.")
    return OpenAI()


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_ -]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def _extract_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def _cleanup_json_text(text: str) -> str:
    text = _strip_markdown_fences(text)
    text = text.replace("﻿", "").strip()
    balanced = _extract_balanced_json_object(text)
    if balanced:
        text = balanced
    text = re.sub(r",\s*([}\]])", r"\1", text)
    lines = text.splitlines()
    repaired = []
    for idx, line in enumerate(lines):
        repaired.append(line)
        if idx >= len(lines) - 1:
            continue
        cur = line.rstrip()
        nxt = lines[idx + 1].lstrip()
        if not cur or not nxt:
            continue
        if nxt.startswith('"') and re.match(r'^"[^"\n]+"\s*:', nxt):
            if re.search(r'("|\d|true|false|null|[}\]])\s*$', cur) and not re.search(r'[,:{\[]\s*$', cur):
                repaired[-1] = cur + ','
    return "\n".join(repaired).strip()

def _try_parse_object(text: str) -> Optional[Dict[str, Any]]:
    parsers = [("json", json.loads)]
    if json5 is not None:
        parsers.append(("json5", json5.loads))
    if yaml is not None:
        parsers.append(("yaml", yaml.safe_load))
    for _name, parser in parsers:
        try:
            obj = parser(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def _extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        raise RuntimeError("Model returned an empty response.")
    cleaned = _cleanup_json_text(text)
    obj = _try_parse_object(cleaned)
    if obj is not None:
        return obj
    balanced = _extract_balanced_json_object(cleaned)
    if balanced and balanced != cleaned:
        obj = _try_parse_object(balanced)
        if obj is not None:
            return obj
    raise RuntimeError("Could not parse JSON from model response.")

def _mock_section_response(section_name: str, schema_name: str, prompt: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    if section_name == "face_identity":
        return {
            "enabled": True,
            "score": 0.83,
            "confidence": 0.70,
            "reasoning_summary": "Mock VLM judged strong face-region consistency with moderate confidence.",
            "face_consistency_binary_hint": True,
            "hard_fail": False,
            "debug": {"mode": "mock"},
        }
    if section_name == "body_shape":
        return {"enabled": True, "score": 0.76, "confidence": 0.68, "reasoning_summary": "Mock VLM judged similar body type after ignoring pose.", "debug": {"mode": "mock"}}
    if section_name == "lighting":
        return {"enabled": True, "score": 0.81, "confidence": 0.74, "reasoning_summary": "Mock VLM judged lighting reasonably aligned with target intent.", "lighting_target_used": extra.get("lighting_target", ""), "debug": {"mode": "mock"}}
    if section_name == "background":
        return {"enabled": True, "score": 0.79, "confidence": 0.71, "reasoning_summary": "Mock VLM judged background plausible for the requested intent.", "background_target_used": extra.get("background_target", ""), "debug": {"mode": "mock"}}
    if section_name == "garments_total":
        garments = []
        inv = extra.get("inventory", [])
        for g in inv:
            missing = False
            garments.append({
                "garment_id": g.get("garment_id", g.get("garment_type", "garment")),
                "garment_type": g.get("garment_type", "unknown"),
                "required": bool(g.get("required", True)),
                "detected": not missing,
                "missing": missing,
                "score": 0.77,
                "confidence": 0.72,
                "reasoning_summary": "Mock VLM judged garment present and reasonably authentic.",
                "subscores": {
                    "presence": 1.0,
                    "shape_structure": 0.76,
                    "texture_color_text": 0.75,
                    "placement": 0.78,
                },
                "panel_index": int(g.get("panel_index", 0)),
                "label_source": g.get("label_source", "rubric"),
                "weight_within_garments": _safe_float(g.get("weight_within_garments", 1.0), 1.0),
            })
        score = float(np.mean([g["score"] for g in garments])) if garments else 0.0
        return {
            "enabled": True,
            "score": score,
            "confidence": 0.72,
            "hard_fail": False,
            "hard_fail_reasons": [],
            "weights_used": extra.get("weights_used", {}),
            "garments": garments,
            "debug": {"mode": "mock", "inventory_count": len(inv)},
        }
    raise RuntimeError(f"No mock response defined for section {section_name}.")


def _call_openai_structured(
    section_name: str,
    prompt: str,
    images: Sequence[np.ndarray],
    schema_name: str,
    schema: Dict[str, Any],
    rubric: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extra = extra or {}
    vlm_cfg = rubric.get("options", {}).get("vlm", {})
    if bool(vlm_cfg.get("mock", False)) or os.environ.get("VTON_VLM_MOCK", "0") == "1":
        return _mock_section_response(section_name, schema_name, prompt, extra)

    model = vlm_cfg.get("model", "gpt-5")
    detail = vlm_cfg.get("detail", "high")
    temperature = _safe_float(vlm_cfg.get("temperature", 0.0), 0.0)
    max_output_tokens = int(vlm_cfg.get("max_output_tokens", 1400))

    client = _get_openai_client()
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for rgb in images:
        content.append({"type": "image_url", "image_url": {"url": _np_to_data_url(rgb), "detail": detail}})

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_output_tokens,
        messages=[{"role": "user", "content": content}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        },
    )
    message = response.choices[0].message
    text = message.content or ""
    return _extract_json_from_text(text)


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_prompt_profile(user_prompt_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    profile = copy.deepcopy(DEFAULT_PROMPT_PROFILE)
    if user_prompt_profile:
        _deep_update(profile, user_prompt_profile)
    return profile


def _section_prompt_context(section_name: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {
        "section_name": section_name,
        "background_target": (rubric.get("background_intent", "") or "").strip(),
        "lighting_target": (rubric.get("lighting_model", "") or "").strip() or "Use the selfie image as the lighting target because no explicit lighting_model was specified.",
        "inventory_json": "[]",
        "weights_json": "{}",
        "hard_fail_enabled": "false",
    }
    if section_name == "garments_total":
        section = rubric.get("top_level_sections", {}).get("garments_total", {})
        inventory = section.get("inventory", [])
        ctx["inventory_json"] = json.dumps(inventory, indent=2)
        ctx["weights_json"] = json.dumps(_garment_weights(rubric))
        ctx["hard_fail_enabled"] = str(bool(section.get("hard_fail_on_missing_required", True))).lower()
    return ctx


def _compose_section_prompt(rubric: Dict[str, Any], section_name: str, default_prompt: str) -> str:
    overrides = rubric.get("vlm_overrides", {}).get(section_name, {})
    override = (overrides.get("prompt_override", "") or "").strip()
    append = (overrides.get("prompt_append", "") or "").strip()
    prompt_profile = rubric.get("prompt_profile", {}) or {}
    template = prompt_profile.get("sections", {}).get(section_name, {}).get("prompt_template", "")
    ctx = _PromptSafeDict(_section_prompt_context(section_name, rubric))
    if override:
        out = override.format_map(ctx)
    elif template:
        out = template.format_map(ctx)
    else:
        out = default_prompt
    if append:
        out = f"{out}\n\nAdditional instructions:\n{append}"
    return out


def _connector_prompt_with_schema(section_name: str, rubric: Dict[str, Any], base_prompt: str, schema_name: str, schema: Dict[str, Any], image_order: str) -> str:
    prompt = _compose_section_prompt(rubric, section_name, base_prompt)
    return (
        f"{prompt}\n\n"
        f"Image order for this request:\n{image_order}\n\n"
        "Return ONLY valid JSON. Do not use markdown fences. Do not add prose before or after the JSON.\n\n"
        f"Schema name: {schema_name}\n"
        f"Required JSON schema:\n{json.dumps(schema, indent=2)}"
    )


def _face_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "enabled": {"type": "boolean"},
            "score": {"type": "number"},
            "confidence": {"type": "number"},
            "reasoning_summary": {"type": "string"},
            "face_consistency_binary_hint": {"type": "boolean"},
            "hard_fail": {"type": "boolean"},
            "debug": {"type": "object", "additionalProperties": True},
        },
        "required": ["enabled", "score", "confidence", "reasoning_summary", "face_consistency_binary_hint", "hard_fail", "debug"],
    }


def _simple_schema(name_field: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "enabled": {"type": "boolean"},
            "score": {"type": "number"},
            "confidence": {"type": "number"},
            "reasoning_summary": {"type": "string"},
            name_field: {"type": "string"},
            "debug": {"type": "object", "additionalProperties": True},
        },
        "required": ["enabled", "score", "confidence", "reasoning_summary", name_field, "debug"],
    }


def _body_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "enabled": {"type": "boolean"},
            "score": {"type": "number"},
            "confidence": {"type": "number"},
            "reasoning_summary": {"type": "string"},
            "debug": {"type": "object", "additionalProperties": True},
        },
        "required": ["enabled", "score", "confidence", "reasoning_summary", "debug"],
    }


def _garments_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "enabled": {"type": "boolean"},
            "score": {"type": "number"},
            "confidence": {"type": "number"},
            "hard_fail": {"type": "boolean"},
            "hard_fail_reasons": {"type": "array", "items": {"type": "string"}},
            "weights_used": {"type": "object", "additionalProperties": {"type": "number"}},
            "garments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "garment_id": {"type": "string"},
                        "garment_type": {"type": "string"},
                        "required": {"type": "boolean"},
                        "detected": {"type": "boolean"},
                        "missing": {"type": "boolean"},
                        "score": {"type": "number"},
                        "confidence": {"type": "number"},
                        "reasoning_summary": {"type": "string"},
                        "subscores": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "presence": {"type": "number"},
                                "shape_structure": {"type": "number"},
                                "texture_color_text": {"type": "number"},
                                "placement": {"type": "number"},
                            },
                            "required": ["presence", "shape_structure", "texture_color_text", "placement"],
                        },
                        "panel_index": {"type": "integer"},
                        "label_source": {"type": "string"},
                        "weight_within_garments": {"type": "number"},
                    },
                    "required": [
                        "garment_id", "garment_type", "required", "detected", "missing", "score", "confidence", "reasoning_summary", "subscores", "panel_index", "label_source", "weight_within_garments"
                    ],
                },
            },
            "debug": {"type": "object", "additionalProperties": True},
        },
        "required": ["enabled", "score", "confidence", "hard_fail", "hard_fail_reasons", "weights_used", "garments", "debug"],
    }


def _face_prompt(rubric: Dict[str, Any]) -> str:
    default = DEFAULT_PROMPT_PROFILE["sections"]["face_identity"]["prompt_template"]
    return _compose_section_prompt(rubric, "face_identity", default)


def _body_prompt(rubric: Dict[str, Any]) -> str:
    default = DEFAULT_PROMPT_PROFILE["sections"]["body_shape"]["prompt_template"]
    return _compose_section_prompt(rubric, "body_shape", default)


def _lighting_prompt(rubric: Dict[str, Any], target: str) -> str:
    default = DEFAULT_PROMPT_PROFILE["sections"]["lighting"]["prompt_template"].format_map(_PromptSafeDict({"lighting_target": target}))
    return _compose_section_prompt(rubric, "lighting", default)


def _background_prompt(rubric: Dict[str, Any], target: str) -> str:
    default = DEFAULT_PROMPT_PROFILE["sections"]["background"]["prompt_template"].format_map(_PromptSafeDict({"background_target": target}))
    return _compose_section_prompt(rubric, "background", default)


def _garment_prompt(rubric: Dict[str, Any], inventory: List[Dict[str, Any]], hard_fail_enabled: bool, weights: Dict[str, float]) -> str:
    default = DEFAULT_PROMPT_PROFILE["sections"]["garments_total"]["prompt_template"].format_map(_PromptSafeDict({
        "inventory_json": json.dumps(inventory, indent=2),
        "hard_fail_enabled": str(hard_fail_enabled).lower(),
        "weights_json": json.dumps(weights),
    }))
    return _compose_section_prompt(rubric, "garments_total", default)


def _build_prompt_outputs(rubric: Dict[str, Any]) -> Dict[str, str]:
    return {
        "face_prompt": _connector_prompt_with_schema("face_identity", rubric, _face_prompt(rubric), "face_consistency_section", _face_schema(), "image 1 = source selfie, image 2 = try-on result"),
        "garments_prompt": _connector_prompt_with_schema("garments_total", rubric, _garment_prompt(rubric, rubric.get("top_level_sections", {}).get("garments_total", {}).get("inventory", []), bool(rubric.get("top_level_sections", {}).get("garments_total", {}).get("hard_fail_on_missing_required", True)), _garment_weights(rubric)), "garments_section", _garments_schema(), "image 1 = garment collage/reference, image 2 = try-on result, image 3 = source selfie"),
        "body_prompt": _connector_prompt_with_schema("body_shape", rubric, _body_prompt(rubric), "body_shape_section", _body_schema(), "image 1 = source selfie, image 2 = try-on result"),
        "lighting_prompt": _connector_prompt_with_schema("lighting", rubric, _lighting_prompt(rubric, (rubric.get("lighting_model", "") or "").strip() or "Use the selfie image as the lighting target because no explicit lighting_model was specified."), "lighting_section", _simple_schema("lighting_target_used"), "image 1 = source selfie, image 2 = try-on result"),
        "background_prompt": _connector_prompt_with_schema("background", rubric, _background_prompt(rubric, (rubric.get("background_intent", "") or "").strip()), "background_section", _simple_schema("background_target_used"), "image 1 = try-on result"),
    }


def _section_override_prompt(rubric: Dict[str, Any], section_name: str, default_prompt: str) -> str:
    return _compose_section_prompt(rubric, section_name, default_prompt)


def _validate_rubric(user_rubric: Optional[Dict[str, Any]], collage_rgb: Optional[np.ndarray] = None, prompt_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rubric = copy.deepcopy(DEFAULT_RUBRIC)
    if user_rubric:
        _deep_update(rubric, user_rubric)

    top_sections = rubric.setdefault("top_level_sections", {})
    for key, default_value in DEFAULT_RUBRIC["top_level_sections"].items():
        if key not in top_sections:
            top_sections[key] = copy.deepcopy(default_value)
        elif isinstance(default_value, dict):
            _deep_update(top_sections[key], copy.deepcopy(default_value))
            _deep_update(top_sections[key], user_rubric.get("top_level_sections", {}).get(key, {}) if user_rubric else {})

    rubric.setdefault("vlm_overrides", copy.deepcopy(DEFAULT_RUBRIC["vlm_overrides"]))
    for key, default_value in DEFAULT_RUBRIC["vlm_overrides"].items():
        if key not in rubric["vlm_overrides"]:
            rubric["vlm_overrides"][key] = copy.deepcopy(default_value)
        else:
            merged = copy.deepcopy(default_value)
            _deep_update(merged, rubric["vlm_overrides"][key])
            rubric["vlm_overrides"][key] = merged

    rubric["prompt_profile"] = _resolve_prompt_profile(prompt_profile)

    section = rubric["top_level_sections"]["garments_total"]
    inventory = list(section.get("inventory", []))
    labels: List[Dict[str, Any]] = []
    if collage_rgb is not None:
        labels = _parse_collage_labels(collage_rgb)

    options = rubric.get("options", {})
    inventory_mode = options.get("inventory_mode", "rubric_first_collage_fallback")
    use_collage_labels = bool(options.get("use_collage_labels_when_present", True))

    if not inventory and collage_rgb is not None and use_collage_labels and inventory_mode in {"rubric_first_collage_fallback", "collage_labels_first", "auto_from_collage", "labels_only"}:
        for idx, label in enumerate(labels):
            inventory.append({
                "garment_id": f"{label['primary_label']}_{idx + 1:02d}",
                "garment_type": label["primary_label"],
                "required": True,
                "panel_index": idx,
                "weight_within_garments": 1.0,
                "enabled": True,
                "label_source": "collage_ocr",
                "collage_label": label["primary_label"],
            })

    enabled_inventory = [g for g in inventory if bool(g.get("enabled", True))]
    if enabled_inventory:
        total_w = sum(_safe_float(g.get("weight_within_garments", 1.0), 1.0) for g in enabled_inventory)
        for g in inventory:
            if bool(g.get("enabled", True)) and total_w > 0:
                g["weight_within_garments"] = _safe_float(g.get("weight_within_garments", 1.0), 1.0) / total_w
            else:
                g["weight_within_garments"] = 0.0

    section["inventory"] = inventory
    section["_parsed_labels"] = labels
    return rubric


def _garment_weights(rubric: Dict[str, Any]) -> Dict[str, float]:
    section = rubric["top_level_sections"]["garments_total"]
    return _normalize_subweights(section.get("default_subsections", {}))


def _parse_section_payload(raw_response: str) -> Dict[str, Any]:
    return _extract_json_from_text(raw_response)


def _normalize_face_payload(payload: Dict[str, Any], enabled_default: bool = True) -> Dict[str, Any]:
    payload["enabled"] = bool(payload.get("enabled", enabled_default))
    payload["score"] = _clamp01(payload.get("score", 0.0))
    payload["confidence"] = _clamp01(payload.get("confidence", 0.0))
    payload.setdefault("reasoning_summary", "")
    hint = payload.get("face_consistency_binary_hint", payload.get("same_person_binary_hint", payload["score"] >= 0.5))
    payload["face_consistency_binary_hint"] = bool(hint)
    payload["same_person_binary_hint"] = bool(hint)
    payload["hard_fail"] = bool(payload.get("hard_fail", False))
    payload.setdefault("debug", {})
    return payload


class VLMRubricLoadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "collage_image": ("IMAGE", {}),
                "rubric_path": ("STRING", {"multiline": False, "default": ""}),
                "rubric_json_override": ("STRING", {"multiline": True, "default": ""}),
                "prompt_profile_path": ("STRING", {"multiline": False, "default": ""}),
                "prompt_profile_json_override": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("VTON_RUBRIC", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("rubric", "rubric_json", "face_prompt", "garments_prompt", "body_prompt", "lighting_prompt", "background_prompt")
    FUNCTION = "load"
    CATEGORY = PACKAGE_CATEGORY

    def load(self, collage_image: torch.Tensor, rubric_path: str, rubric_json_override: str, prompt_profile_path: str, prompt_profile_json_override: str):
        loaded: Dict[str, Any] = {}
        if rubric_json_override.strip():
            loaded = json.loads(rubric_json_override)
        elif rubric_path.strip():
            loaded = _load_json_file(rubric_path)

        prompt_profile_loaded: Dict[str, Any] = {}
        if prompt_profile_json_override.strip():
            prompt_profile_loaded = json.loads(prompt_profile_json_override)
        elif prompt_profile_path.strip():
            prompt_profile_loaded = _load_json_file(prompt_profile_path)
        else:
            rubric_prompt_path = str(loaded.get("options", {}).get("prompt_profile_path", "") or "").strip()
            if rubric_prompt_path:
                prompt_profile_loaded = _load_json_file(rubric_prompt_path)

        collage = _tensor_to_np(collage_image)
        rubric = _validate_rubric(loaded, collage, prompt_profile_loaded)
        prompts = _build_prompt_outputs(rubric)
        return (rubric, _json_dumps(rubric), prompts["face_prompt"], prompts["garments_prompt"], prompts["body_prompt"], prompts["lighting_prompt"], prompts["background_prompt"])


class VLMFaceIdentityScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source_image": ("IMAGE", {}), "tryon_image": ("IMAGE", {}), "rubric": ("VTON_RUBRIC", {})}}

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("face_section", "face_score", "face_json")
    FUNCTION = "score_face"
    CATEGORY = PACKAGE_CATEGORY

    def score_face(self, source_image: torch.Tensor, tryon_image: torch.Tensor, rubric: Dict[str, Any]):
        section = rubric["top_level_sections"]["face_identity"]
        if not section.get("enabled", True):
            payload = _normalize_face_payload({"enabled": False, "score": 0.0, "confidence": 0.0, "reasoning_summary": "Section disabled.", "face_consistency_binary_hint": False, "hard_fail": False, "debug": {}}, enabled_default=False)
            return (payload, 0.0, _json_dumps(payload))
        src = _tensor_to_np(source_image)
        dst = _tensor_to_np(tryon_image)
        prompt = _face_prompt(rubric)
        payload = _call_openai_structured("face_identity", prompt, [src, dst], "face_consistency_section", _face_schema(), rubric)
        payload = _normalize_face_payload(payload)
        return (payload, float(payload["score"]), _json_dumps(payload))


class VLMGarmentScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source_image": ("IMAGE", {}), "collage_image": ("IMAGE", {}), "tryon_image": ("IMAGE", {}), "rubric": ("VTON_RUBRIC", {})}}

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("garments_section", "garments_score", "garments_json")
    FUNCTION = "score_garments"
    CATEGORY = PACKAGE_CATEGORY

    def score_garments(self, source_image: torch.Tensor, collage_image: torch.Tensor, tryon_image: torch.Tensor, rubric: Dict[str, Any]):
        section = rubric["top_level_sections"]["garments_total"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "hard_fail": False, "hard_fail_reasons": [], "weights_used": {}, "garments": [], "debug": {}}
            return (payload, 0.0, _json_dumps(payload))

        inventory = section.get("inventory", [])
        collage = _tensor_to_np(collage_image)
        parsed_labels = section.get("_parsed_labels", [])
        if not inventory:
            payload = {"enabled": True, "score": 0.0, "confidence": 0.0, "hard_fail": False, "hard_fail_reasons": [], "weights_used": _garment_weights(rubric), "garments": [], "debug": {"inventory_creation_failed": True, "label_count": len(parsed_labels), "parsed_labels": [label.get("primary_label", "") for label in parsed_labels]}}
            return (payload, 0.0, _json_dumps(payload))

        src = _tensor_to_np(source_image)
        dst = _tensor_to_np(tryon_image)
        weights = _garment_weights(rubric)
        hard_fail_enabled = bool(section.get("hard_fail_on_missing_required", True))
        prompt = _garment_prompt(rubric, inventory, hard_fail_enabled, weights)
        payload = _call_openai_structured("garments_total", prompt, [collage, dst, src], "garments_section", _garments_schema(), rubric, extra={"inventory": inventory, "weights_used": weights})
        payload["enabled"] = bool(payload.get("enabled", True))
        payload["score"] = _clamp01(payload.get("score", 0.0))
        payload["confidence"] = _clamp01(payload.get("confidence", 0.0))
        payload["weights_used"] = weights
        if hard_fail_enabled:
            payload["hard_fail"] = bool(payload.get("hard_fail", False)) or any(g.get("required", True) and bool(g.get("missing", False)) for g in payload.get("garments", []))
            if payload["hard_fail"] and not payload.get("hard_fail_reasons"):
                payload["hard_fail_reasons"] = [f"Missing required garment: {g.get('garment_type', g.get('garment_id', 'garment'))}" for g in payload.get("garments", []) if g.get("required", True) and bool(g.get("missing", False))]
        payload.setdefault("debug", {})
        payload["debug"]["label_count"] = len(parsed_labels)
        payload["debug"]["parsed_labels"] = [label.get("primary_label", "") for label in parsed_labels]
        payload["debug"]["inventory_count"] = len(inventory)
        return (payload, float(payload["score"]), _json_dumps(payload))


class VLMBodyShapeScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source_image": ("IMAGE", {}), "tryon_image": ("IMAGE", {}), "rubric": ("VTON_RUBRIC", {})}}

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("body_section", "body_score", "body_json")
    FUNCTION = "score_body"
    CATEGORY = PACKAGE_CATEGORY

    def score_body(self, source_image: torch.Tensor, tryon_image: torch.Tensor, rubric: Dict[str, Any]):
        section = rubric["top_level_sections"]["body_shape"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "reasoning_summary": "Section disabled.", "debug": {}}
            return (payload, 0.0, _json_dumps(payload))
        src = _tensor_to_np(source_image)
        dst = _tensor_to_np(tryon_image)
        prompt = _body_prompt(rubric)
        payload = _call_openai_structured("body_shape", prompt, [src, dst], "body_shape_section", _body_schema(), rubric)
        payload["score"] = _clamp01(payload.get("score", 0.0))
        payload["confidence"] = _clamp01(payload.get("confidence", 0.0))
        payload["enabled"] = bool(payload.get("enabled", True))
        payload.setdefault("reasoning_summary", "")
        payload.setdefault("debug", {})
        return (payload, float(payload["score"]), _json_dumps(payload))


class VLMLightingScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source_image": ("IMAGE", {}), "tryon_image": ("IMAGE", {}), "rubric": ("VTON_RUBRIC", {})}}

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("lighting_section", "lighting_score", "lighting_json")
    FUNCTION = "score_lighting"
    CATEGORY = PACKAGE_CATEGORY

    def score_lighting(self, source_image: torch.Tensor, tryon_image: torch.Tensor, rubric: Dict[str, Any]):
        section = rubric["top_level_sections"]["lighting"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "reasoning_summary": "Section disabled.", "lighting_target_used": "", "debug": {}}
            return (payload, 0.0, _json_dumps(payload))
        src = _tensor_to_np(source_image)
        dst = _tensor_to_np(tryon_image)
        lighting_target = (rubric.get("lighting_model", "") or "").strip() or "Use the selfie image as the lighting target because no explicit lighting_model was specified."
        prompt = _lighting_prompt(rubric, lighting_target)
        payload = _call_openai_structured("lighting", prompt, [src, dst], "lighting_section", _simple_schema("lighting_target_used"), rubric, extra={"lighting_target": lighting_target})
        payload["score"] = _clamp01(payload.get("score", 0.0))
        payload["confidence"] = _clamp01(payload.get("confidence", 0.0))
        payload["enabled"] = bool(payload.get("enabled", True))
        payload.setdefault("reasoning_summary", "")
        payload.setdefault("debug", {})
        return (payload, float(payload["score"]), _json_dumps(payload))


class VLMBackgroundScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tryon_image": ("IMAGE", {}), "rubric": ("VTON_RUBRIC", {})}}

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("background_section", "background_score", "background_json")
    FUNCTION = "score_background"
    CATEGORY = PACKAGE_CATEGORY

    def score_background(self, tryon_image: torch.Tensor, rubric: Dict[str, Any]):
        section = rubric["top_level_sections"]["background"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "reasoning_summary": "Section disabled.", "background_target_used": "", "debug": {}}
            return (payload, 0.0, _json_dumps(payload))
        dst = _tensor_to_np(tryon_image)
        background_target = (rubric.get("background_intent", "") or "").strip()
        prompt = _background_prompt(rubric, background_target)
        payload = _call_openai_structured("background", prompt, [dst], "background_section", _simple_schema("background_target_used"), rubric, extra={"background_target": background_target})
        payload["score"] = _clamp01(payload.get("score", 0.0))
        payload["confidence"] = _clamp01(payload.get("confidence", 0.0))
        payload["enabled"] = bool(payload.get("enabled", True))
        payload.setdefault("reasoning_summary", "")
        payload.setdefault("debug", {})
        return (payload, float(payload["score"]), _json_dumps(payload))


class VLMFaceParseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"response_text": ("STRING", {"multiline": True, "default": ""}), "rubric": ("VTON_RUBRIC", {})}}

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("face_section", "face_score", "face_json")
    FUNCTION = "parse_response"
    CATEGORY = PACKAGE_CATEGORY

    def parse_response(self, response_text: str, rubric: Dict[str, Any]):
        section = rubric["top_level_sections"]["face_identity"]
        if not section.get("enabled", True):
            payload = _normalize_face_payload({"enabled": False, "score": 0.0, "confidence": 0.0, "reasoning_summary": "Section disabled.", "face_consistency_binary_hint": False, "hard_fail": False, "debug": {}}, enabled_default=False)
            return (payload, 0.0, _json_dumps(payload))
        payload = _normalize_face_payload(_parse_section_payload(response_text))
        return (payload, float(payload["score"]), _json_dumps(payload))


class VLMGarmentParseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"response_text": ("STRING", {"multiline": True, "default": ""}), "rubric": ("VTON_RUBRIC", {})}}

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("garments_section", "garments_score", "garments_json")
    FUNCTION = "parse_response"
    CATEGORY = PACKAGE_CATEGORY

    def parse_response(self, response_text: str, rubric: Dict[str, Any]):
        section = rubric["top_level_sections"]["garments_total"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "hard_fail": False, "hard_fail_reasons": [], "weights_used": {}, "garments": [], "debug": {}}
            return (payload, 0.0, _json_dumps(payload))
        try:
            payload = _parse_section_payload(response_text)
        except Exception as e:
            payload = {
                "enabled": True,
                "score": 0.0,
                "confidence": 0.0,
                "hard_fail": False,
                "hard_fail_reasons": [],
                "weights_used": _garment_weights(rubric),
                "garments": [],
                "debug": {
                    "parse_error": str(e),
                    "raw_response_excerpt": str(response_text)[:1200],
                },
            }
            return (payload, 0.0, _json_dumps(payload))
        payload["enabled"] = bool(payload.get("enabled", True))
        payload["score"] = _clamp01(payload.get("score", 0.0))
        payload["confidence"] = _clamp01(payload.get("confidence", 0.0))
        payload["weights_used"] = _garment_weights(rubric)
        payload.setdefault("garments", [])
        payload.setdefault("hard_fail_reasons", [])
        if bool(section.get("hard_fail_on_missing_required", True)):
            payload["hard_fail"] = bool(payload.get("hard_fail", False)) or any(g.get("required", True) and bool(g.get("missing", False)) for g in payload.get("garments", []))
            if payload["hard_fail"] and not payload.get("hard_fail_reasons"):
                payload["hard_fail_reasons"] = [f"Missing required garment: {g.get('garment_type', g.get('garment_id', 'garment'))}" for g in payload.get("garments", []) if g.get("required", True) and bool(g.get("missing", False))]
        payload.setdefault("debug", {})
        parsed_labels = section.get("_parsed_labels", [])
        payload["debug"]["label_count"] = len(parsed_labels)
        payload["debug"]["parsed_labels"] = [label.get("primary_label", "") for label in parsed_labels]
        payload["debug"]["inventory_count"] = len(section.get("inventory", []))
        return (payload, float(payload["score"]), _json_dumps(payload))


class VLMBodyShapeParseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"response_text": ("STRING", {"multiline": True, "default": ""}), "rubric": ("VTON_RUBRIC", {})}}

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("body_section", "body_score", "body_json")
    FUNCTION = "parse_response"
    CATEGORY = PACKAGE_CATEGORY

    def parse_response(self, response_text: str, rubric: Dict[str, Any]):
        section = rubric["top_level_sections"]["body_shape"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "reasoning_summary": "Section disabled.", "debug": {}}
            return (payload, 0.0, _json_dumps(payload))
        payload = _parse_section_payload(response_text)
        payload["enabled"] = bool(payload.get("enabled", True))
        payload["score"] = _clamp01(payload.get("score", 0.0))
        payload["confidence"] = _clamp01(payload.get("confidence", 0.0))
        payload.setdefault("reasoning_summary", "")
        payload.setdefault("debug", {})
        return (payload, float(payload["score"]), _json_dumps(payload))


class VLMLightingParseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"response_text": ("STRING", {"multiline": True, "default": ""}), "rubric": ("VTON_RUBRIC", {})}}

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("lighting_section", "lighting_score", "lighting_json")
    FUNCTION = "parse_response"
    CATEGORY = PACKAGE_CATEGORY

    def parse_response(self, response_text: str, rubric: Dict[str, Any]):
        section = rubric["top_level_sections"]["lighting"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "reasoning_summary": "Section disabled.", "lighting_target_used": "", "debug": {}}
            return (payload, 0.0, _json_dumps(payload))
        payload = _parse_section_payload(response_text)
        payload["enabled"] = bool(payload.get("enabled", True))
        payload["score"] = _clamp01(payload.get("score", 0.0))
        payload["confidence"] = _clamp01(payload.get("confidence", 0.0))
        payload.setdefault("reasoning_summary", "")
        payload.setdefault("lighting_target_used", (rubric.get("lighting_model", "") or "").strip())
        payload.setdefault("debug", {})
        return (payload, float(payload["score"]), _json_dumps(payload))


class VLMBackgroundParseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"response_text": ("STRING", {"multiline": True, "default": ""}), "rubric": ("VTON_RUBRIC", {})}}

    RETURN_TYPES = ("VTON_SECTION_SCORE", "FLOAT", "STRING")
    RETURN_NAMES = ("background_section", "background_score", "background_json")
    FUNCTION = "parse_response"
    CATEGORY = PACKAGE_CATEGORY

    def parse_response(self, response_text: str, rubric: Dict[str, Any]):
        section = rubric["top_level_sections"]["background"]
        if not section.get("enabled", True):
            payload = {"enabled": False, "score": 0.0, "confidence": 0.0, "reasoning_summary": "Section disabled.", "background_target_used": "", "debug": {}}
            return (payload, 0.0, _json_dumps(payload))
        payload = _parse_section_payload(response_text)
        payload["enabled"] = bool(payload.get("enabled", True))
        payload["score"] = _clamp01(payload.get("score", 0.0))
        payload["confidence"] = _clamp01(payload.get("confidence", 0.0))
        payload.setdefault("reasoning_summary", "")
        payload.setdefault("background_target_used", (rubric.get("background_intent", "") or "").strip())
        payload.setdefault("debug", {})
        return (payload, float(payload["score"]), _json_dumps(payload))


class VLMAggregateRubricNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rubric": ("VTON_RUBRIC", {}),
                "face_section": ("VTON_SECTION_SCORE", {}),
                "garments_section": ("VTON_SECTION_SCORE", {}),
                "body_section": ("VTON_SECTION_SCORE", {}),
                "lighting_section": ("VTON_SECTION_SCORE", {}),
                "background_section": ("VTON_SECTION_SCORE", {}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("final_score", "report_json", "summary_text")
    FUNCTION = "aggregate"
    CATEGORY = PACKAGE_CATEGORY
    OUTPUT_NODE = True

    def aggregate(self, rubric: Dict[str, Any], face_section: Dict[str, Any], garments_section: Dict[str, Any], body_section: Dict[str, Any], lighting_section: Dict[str, Any], background_section: Dict[str, Any]):
        report = _report_shell()
        sections = rubric["top_level_sections"]
        top_section_map = {
            "face_identity": face_section,
            "garments_total": garments_section,
            "body_shape": body_section,
            "lighting": lighting_section,
            "background": background_section,
        }
        weights_used = _normalize_enabled_weights(sections)
        report["top_level_weights_used"] = weights_used
        hard_fail = bool(garments_section.get("hard_fail", False))
        report["hard_fail"] = hard_fail
        report["hard_fail_reasons"] = list(garments_section.get("hard_fail_reasons", []))
        report["garments"] = garments_section.get("garments", [])

        weighted_sum = 0.0
        for key, section_payload in top_section_map.items():
            report["top_level_scores"][key] = float(section_payload.get("score", 0.0))
            report["top_level_confidence"][key] = float(section_payload.get("confidence", 0.0))
            weighted_sum += weights_used.get(key, 0.0) * float(section_payload.get("score", 0.0))

        final_score = 0.0 if hard_fail else _clamp01(weighted_sum)
        report["final_score"] = final_score

        policy = rubric.get("options", {}).get("recommendation_policy", {})
        accept_min = _clamp01(_safe_float(policy.get("accept_min_score", 0.70), 0.70))
        review_min = _clamp01(_safe_float(policy.get("review_min_score", 0.60), 0.60))
        hard_fail_rec = str(policy.get("hard_fail_recommendation", "reject")).strip().lower() or "reject"
        if review_min > accept_min:
            review_min = accept_min
        if hard_fail_rec not in {"accept", "review", "reject"}:
            hard_fail_rec = "reject"

        report["meta"] = {
            "rubric_version": rubric.get("version", "3.2"),
            "speed_profile": rubric.get("options", {}).get("speed_profile", "balanced"),
            "vlm_model": rubric.get("options", {}).get("vlm", {}).get("model", "gpt-5"),
            "background_intent": rubric.get("background_intent", ""),
            "lighting_model": rubric.get("lighting_model", ""),
            "recommendation_policy_used": {
                "accept_min_score": accept_min,
                "review_min_score": review_min,
                "hard_fail_recommendation": hard_fail_rec,
            },
        }
        if hard_fail:
            report["operator_recommendation"] = hard_fail_rec
        elif final_score >= accept_min:
            report["operator_recommendation"] = "accept"
        elif final_score >= review_min:
            report["operator_recommendation"] = "review"
        else:
            report["operator_recommendation"] = "reject"

        summary = (
            f"Final VTON VLM QC score: {final_score:.4f}\n"
            f"Recommendation: {report['operator_recommendation']}\n"
            f"Hard fail: {report['hard_fail']}\n"
            f"Face: {report['top_level_scores'].get('face_identity', 0.0):.4f}\n"
            f"Garments: {report['top_level_scores'].get('garments_total', 0.0):.4f}\n"
            f"Body: {report['top_level_scores'].get('body_shape', 0.0):.4f}\n"
            f"Lighting: {report['top_level_scores'].get('lighting', 0.0):.4f}\n"
            f"Background: {report['top_level_scores'].get('background', 0.0):.4f}"
        )
        return (float(final_score), _json_dumps(report), summary)


class VLMSaveJSONReportNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"report_json": ("STRING", {"multiline": True, "default": ""}), "filename_prefix": ("STRING", {"multiline": False, "default": "vton_vlm_qc_report"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = PACKAGE_CATEGORY
    OUTPUT_NODE = True

    def save(self, report_json: str, filename_prefix: str):
        output_dir = _get_output_dir()
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.json"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_json)
        return (path,)


NODE_CLASS_MAPPINGS = {
    "VLMRubricLoadNode": VLMRubricLoadNode,
    "VLMFaceIdentityScoreNode": VLMFaceIdentityScoreNode,
    "VLMGarmentScoreNode": VLMGarmentScoreNode,
    "VLMBodyShapeScoreNode": VLMBodyShapeScoreNode,
    "VLMLightingScoreNode": VLMLightingScoreNode,
    "VLMBackgroundScoreNode": VLMBackgroundScoreNode,
    "VLMFaceParseNode": VLMFaceParseNode,
    "VLMGarmentParseNode": VLMGarmentParseNode,
    "VLMBodyShapeParseNode": VLMBodyShapeParseNode,
    "VLMLightingParseNode": VLMLightingParseNode,
    "VLMBackgroundParseNode": VLMBackgroundParseNode,
    "VLMAggregateRubricNode": VLMAggregateRubricNode,
    "VLMSaveJSONReportNode": VLMSaveJSONReportNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VLMRubricLoadNode": "VTON VLM QC - Load Rubric",
    "VLMFaceIdentityScoreNode": "VTON VLM QC - Face (Direct)",
    "VLMGarmentScoreNode": "VTON VLM QC - Garments (Direct)",
    "VLMBodyShapeScoreNode": "VTON VLM QC - Body Shape (Direct)",
    "VLMLightingScoreNode": "VTON VLM QC - Lighting (Direct)",
    "VLMBackgroundScoreNode": "VTON VLM QC - Background (Direct)",
    "VLMFaceParseNode": "VTON VLM QC - Face Parse",
    "VLMGarmentParseNode": "VTON VLM QC - Garments Parse",
    "VLMBodyShapeParseNode": "VTON VLM QC - Body Parse",
    "VLMLightingParseNode": "VTON VLM QC - Lighting Parse",
    "VLMBackgroundParseNode": "VTON VLM QC - Background Parse",
    "VLMAggregateRubricNode": "VTON VLM QC - Aggregate",
    "VLMSaveJSONReportNode": "VTON VLM QC - Save JSON",
}