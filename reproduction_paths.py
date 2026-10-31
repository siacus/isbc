"""Resolve generated reference caches without a machine-specific filename hash."""
from pathlib import Path

def qwen_reference_cache(root: Path) -> Path:
    candidates=sorted((root/'runs/agnews_embedding_comparison/_information_reference_cache/qwen').glob('information_reference_codes_*.npz'))
    if len(candidates)!=1:
        raise ValueError(f'Expected one Qwen reference-code cache; found {len(candidates)}. Regenerate the five-backbone reference analysis in a clean output directory or select the intended cache explicitly.')
    return candidates[0]
