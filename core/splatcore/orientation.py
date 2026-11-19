"""Conservative, metadata-based viewer orientation; never modifies splat data.

PLY does not define a world-up convention. Unknown files deliberately return
None rather than guessing a sign from their bounding box. A saved viewer
override takes precedence over this suggestion.
"""
import json
import math
from pathlib import Path


def detect_orientation(path):
    """Recognise Terra's local ENU sidecar beside a PLY or above Block/LOD.

    Only unrotated, positive-scale ENU is accepted. Rotated local frames need
    arbitrary up-vector support, not an incorrect cardinal-axis suggestion.
    Stop at the nearest sidecar, even if it is unsupported or malformed.
    """
    path = Path(path)
    if path.suffix.lower() != '.ply':
        return None
    for parent in list(path.parents)[:3]:
        sidecar = parent / 'geo_desc.json'
        try:
            if not sidecar.is_file():
                continue
            if sidecar.stat().st_size > 65536:
                return None
            data = json.loads(sidecar.read_text(encoding='utf-8-sig'))
            if data.get('cs_type') != 'LOCAL_ENU_CS':
                return None
            rotation = data.get('rvec')
            if not isinstance(rotation, list) or len(rotation) != 3:
                return None
            if not all(math.isfinite(float(v)) and abs(float(v)) < 1e-8 for v in rotation):
                return None
            scale = float(data.get('scale', 1))
            if not math.isfinite(scale) or scale <= 0:
                return None
            return {'up': '+z', 'source': 'ENU metadata (geo_desc.json)'}
        except (OSError, ValueError, TypeError, AttributeError):
            return None
    return None
