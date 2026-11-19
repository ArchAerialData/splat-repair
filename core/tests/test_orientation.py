"""Orientation metadata must never turn an unknown or rotated scan by guesswork."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splatcore.orientation import detect_orientation


with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    scan = root / 'Block000' / 'LOD0' / 'point_cloud.ply'
    scan.parent.mkdir(parents=True)
    scan.touch()
    assert detect_orientation(scan) is None
    sidecar = root / 'geo_desc.json'
    data = {'cs_type': 'LOCAL_ENU_CS', 'rvec': [0, 0, 0], 'scale': 1}
    sidecar.write_text(json.dumps(data))
    assert detect_orientation(scan)['up'] == '+z'
    assert detect_orientation(scan.with_suffix('.spz')) is None
    for bad in [dict(data, rvec=[1, 0, 0]), dict(data, scale=-1),
                dict(data, cs_type='UNKNOWN'), dict(data, rvec=[float('nan'), 0, 0])]:
        sidecar.write_text(json.dumps(bad))
        assert detect_orientation(scan) is None
    sidecar.write_text(json.dumps(data))
    nearest = scan.parent / 'geo_desc.json'
    nearest.write_text('{invalid')
    assert detect_orientation(scan) is None
    nearest.write_text('[]')
    assert detect_orientation(scan) is None
    nearest.write_bytes(b'\xff\xfe\x80')
    assert detect_orientation(scan) is None, 'Malformed encoding cannot break scan loading'
    nearest.write_text(' ' * 65537)
    assert detect_orientation(scan) is None, 'Oversized metadata is ignored'
    nearest.write_text(json.dumps(data), encoding='utf-8-sig')
    assert detect_orientation(scan)['up'] == '+z', 'BOM sidecars are accepted'
    for bad in [dict(data, scale=float('inf')), dict(data, scale=None),
                dict(data, rvec=[0, 0]), dict(data, rvec=None), dict(data, scale=0)]:
        nearest.write_text(json.dumps(bad))
        assert detect_orientation(scan) is None
print('orientation metadata checks passed')
