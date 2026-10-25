"""Experiment configuration: classes, prompts, hierarchies, models."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "EuroSAT-SAR"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"

SEED = 0
DEV_FRACTION = 0.2  # stratified split used ONLY for prompt/template selection

# Folder names of EuroSAT-SAR, in label-index order (alphabetical).
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

# Natural-language class names inserted into the prompt templates.
READABLE = [
    "annual crop land", "forest", "herbaceous vegetation", "highway or road",
    "industrial buildings", "pasture land", "permanent crop land",
    "residential buildings", "river", "sea or lake",
]

PROMPT_TEMPLATES = {
    # general
    "generic": "a satellite image of {}.",
    "photo": "a photo of {}.",
    "aerial": "an aerial view of {}.",
    # SAR-specific
    "sar_basic": "radar imagery showing {}.",
    "sar_backscatter": "SAR backscatter pattern of {}.",
    "sar_texture": "grayscale radar texture of {}.",
    # remote-sensing
    "sentinel": "a Sentinel-1 satellite image of {}.",
    "landcover": "land cover classification of {}.",
    # structural
    "structure": "spatial pattern of {} from above.",
    "texture_gray": "grayscale texture pattern of {}.",
}

PROMPT_CATEGORY = {
    "generic": "General", "photo": "General", "aerial": "Spatial",
    "sar_basic": "SAR-specific", "sar_backscatter": "SAR-specific", "sar_texture": "SAR-specific",
    "sentinel": "Remote sensing", "landcover": "Remote sensing",
    "structure": "Structural", "texture_gray": "Structural",
}

# Physics-informed class descriptions (drafted with LLM assistance, then edited by
# the authors). Used with the "aerial" template prefix; listed in full in the paper.
PHYSICS_DESCRIPTIONS = [
    "annual crop land, with regular field parcels of smooth to medium texture and low to moderate radar backscatter",
    "a forest, with a dense, coarse canopy texture and uniformly high radar backscatter from volume scattering",
    "herbaceous vegetation, with an irregular, medium-textured surface and moderate radar backscatter",
    "a highway or road, a thin linear feature with low backscatter crossing brighter surrounding land",
    "industrial buildings, with large bright spots from double-bounce scattering off buildings and metal structures",
    "pasture land, with a smooth, homogeneous surface and low to moderate radar backscatter",
    "permanent crop land such as orchards or vineyards, with a regular row texture and moderate radar backscatter",
    "residential buildings, a bright, granular urban texture caused by double-bounce scattering",
    "a river, a dark sinuous band of very low backscatter from specular reflection on water",
    "a sea or lake, a large dark homogeneous area of very low backscatter from specular reflection on calm water",
]

# Two-level hierarchies. Indices refer to CLASSES.
HIERARCHIES = {
    "H3": {
        "names": ["vegetation", "built-up", "water"],
        "readable": ["vegetated or agricultural land", "built-up area or road", "a water body"],
        "groups": [[0, 1, 2, 5, 6], [3, 4, 7], [8, 9]],
    },
    "H5": {
        "names": ["agriculture", "natural vegetation", "built-up", "transport", "water"],
        "readable": ["agricultural land", "natural vegetation", "urban built-up area",
                     "transportation infrastructure", "a water body"],
        "groups": [[0, 6], [1, 2, 5], [4, 7], [3], [8, 9]],
    },
}

# (key, open_clip architecture, pretrained tag or None, HF checkpoint (repo, file) or None)
MODELS = {
    "clip_b32": ("CLIP ViT-B/32", "ViT-B-32", "openai", None),
    "clip_l14": ("CLIP ViT-L/14", "ViT-L-14", "openai", None),
    "remoteclip_b32": ("RemoteCLIP ViT-B/32", "ViT-B-32", None, ("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt")),
    "remoteclip_l14": ("RemoteCLIP ViT-L/14", "ViT-L-14", None, ("chendelong/RemoteCLIP", "RemoteCLIP-ViT-L-14.pt")),
    "georsclip_b32": ("GeoRSCLIP ViT-B/32", "ViT-B-32", None, ("Zilun/GeoRSCLIP", "ckpt/RS5M_ViT-B-32.pt")),
}

# Input encodings of the dual-pol SAR patch into the 3 channels a VLM expects.
INPUT_MODES = ["vv", "vh", "dual"]
ROTATIONS = [0, 1, 2, 3]  # multiples of 90 degrees (TTA)
