"""Known flat/sloped surfaces, real faint surface detail, and displaced haze."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splatcore.io import SplatSet
from splatcore.surface_noise import analyse

x, y = np.meshgrid(np.linspace(-2, 2, 81), np.linspace(-2, 2, 81))
ground = np.column_stack([x.ravel(), y.ravel(), x.ravel() * .3])
rng = np.random.default_rng(4)
anchor = ground[rng.choice(len(ground), 200, replace=False)].copy()
detail = anchor.copy()
haze = anchor + [-.03, 0, .1]
far = anchor + [0, 0, 20]
xyz = np.vstack([ground, detail, haze, far]).astype(np.float32)
n = len(xyz)
alpha = np.concatenate([np.full(len(ground), .95), np.full(600, .12)])
rot = np.zeros((n, 4), np.float32); rot[:, 0] = 1
s = SplatSet(xyz, np.zeros((n, 3), np.float32), np.zeros((n, 0, 3), np.float32),
             np.log(alpha / (1-alpha)).astype(np.float32),
             np.full((n, 3), np.log(.04), np.float32), rot)
before = s.xyz.copy()
ids, scores, info = analyse(s)
selected = ids[scores >= .35]
start = len(ground)
assert np.count_nonzero(selected < start + 200) == 0, 'Preserve opaque ground and faint on-surface detail'
assert np.count_nonzero((selected >= start+200) & (selected < start+400)) > 150, 'Find displaced haze'
assert np.count_nonzero(selected >= start+400) == 0, 'Distant floaters belong to the existing detector'
ids, scores, _ = analyse(s, np.arange(start+200, start+210))
assert set(ids).issubset(set(range(start+200, start+210))), 'Respect selection scope'
assert np.array_equal(before, s.xyz), 'Analysis must not mutate geometry'
for angle in [0.8, 2.1]:
    c, t = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0, t], [0, 1, 0], [-t, 0, c]])
    rotated = s.copy(); rotated.xyz = (s.xyz @ rotation.T).astype(np.float32)
    ri, rs, _ = analyse(rotated)
    assert set(ri[rs >= .35]) == set(selected), 'Detection must not depend on the up axis'
print('Surface haze checks passed:', len(selected), 'candidates')

# The protocol returns stable document indices even after deletions.
from server import Session
from splatcore.document import Document
session = Session(); session.doc = Document(s)
def forbidden_copy():
    raise AssertionError('Surface haze must not copy the full radiance field')
session.doc.live_splats = forbidden_copy
session.doc.delete(np.array([0, start+200]))
requested = np.arange(start+200, start+210, dtype='<u4')
reply, payload = session.cmd_surface_noise({'selectionOnly': True, 'threshold': .35}, requested.tobytes())
assert set(reply['indices']) == set(range(start+201, start+210))
assert payload == b''
assert not session.doc.alive[start+200]
try:
    session.cmd_surface_noise({'selectionOnly': True}, b'')
    raise AssertionError('Empty scoped requests must not analyse the whole scene')
except ValueError:
    pass
for request, raw in [({'selectionOnly': True}, np.array([n+100], dtype='<u4').tobytes()),
                     ({'selectionOnly': True}, np.array([start+200], dtype='<u4').tobytes()),
                     ({'selectionOnly': True}, b'bad'),
                     ({'threshold': float('nan')}, b''), ({'threshold': 0}, b'')]:
    try:
        session.cmd_surface_noise(request, raw)
        raise AssertionError('Invalid scope or threshold accepted')
    except ValueError:
        pass
# Appended repair splats keep their document IDs, not compacted live indices.
session.doc.generated = s.take(np.arange(start+200, start+210))
session.doc.alive = np.concatenate([session.doc.alive, np.ones(10, bool)])
reply, _ = session.cmd_surface_noise({'selectionOnly': True}, np.arange(n, n+10, dtype='<u4').tobytes())
assert set(reply['indices']) == set(range(n, n+10))
# A deleted opaque surface must not remain a reference that convicts a faint
# splat; and an empty candidate set must not need reference geometry at all.
ids, scores, info = analyse(s, np.array([], dtype=np.int64), alive=np.zeros(n, bool))
assert not len(ids) and info['tested'] == 0
try:
    analyse(s, alive=np.arange(n) >= start)
    raise AssertionError('Deleted reference surface was used')
except ValueError:
    pass
for cap in [0, 23, 2.5]:
    try:
        analyse(s, reference_limit=cap)
        raise AssertionError('Invalid reference limit accepted')
    except ValueError:
        pass
invalid = s.copy(); invalid.xyz[start+201, 0] = np.nan
ii, ss, info = analyse(invalid)
assert start+201 not in ii and info['invalidSkipped'] == 1
# Nearby slender reference structures are not reliable surface planes.
line = s.copy()
line.xyz[:start, 1:] = 0
li, ls, _ = analyse(line)
assert not np.any(ls), 'Line-like references must not classify surrounding splats as haze'
print('Surface haze protocol checks passed')
