# Classification design and implementation boundary

Status: foundation implemented on `codex/classification`. The standalone
`classification.py` model and its unit tests are available on this branch. There is no Document, engine protocol, viewport,
panel, file-save, or project integration yet. This does not expose a working
classification tool in the app.

## User outcome

Select suspect floaters or surface haze, assign a class, then switch that class
off to inspect the scene without deleting data. Assign other selected regions
to custom classes. Restore visibility or reassign at any time. Export all
alive splats, including hidden classes, unless the user explicitly chooses a
filtered export. Class hiding is not memory unloading.

Classification describes what a splat represents. Later spatial work areas
remain a separate concept, so a splat can be both in Stockpile A and in the
Surface haze class. A work-area implementation must not overload class IDs.

## Foundation API

`Classification` owns a `uint16` class-ID array parallel to Document's stable
indices. This includes deleted splats, whose metadata must survive undo.
IDs 0, 1, and 2 initially mean Unclassified, Floaters, and Surface haze. Custom
IDs increase monotonically and are never silently reused. Class definitions
contain name, RGB color, and visibility. Locking is deferred until both viewer
and engine edit restrictions can enforce it consistently.

The model has no geometry, file-system, Electron, or renderer dependency.

| API | Contract |
| --- | --- |
| `empty(count)` | All splats start Unclassified; default registry. |
| `export_ids(indices=None)` | An owned array, optionally in the exact requested export order. |
| `assign(indices, class_id)` | Validate fully, change only differing entries, return a sparse `AssignmentDelta`; no-op returns `None`. |
| `apply_delta(delta, undo=...)` | Restore/replay only if current values match expected history; otherwise fail without partial writes. |
| `append(count, inherited=None)` | Newly reconstructed splats default to Unclassified; caller may supply exact inherited IDs. |
| `truncate(count)` | Remove an appended geometry tail and return its exact class IDs for redo. |
| `create_class(name, color)` | Create a validated registry entry at the next unused ID. |
| `replace_class(definition)` | Replace an existing entry and return the previous definition for document history. |
| `visibility_mask()` | Per-splat class visibility, independent of alive/crop flags. |
| `to_json(exported_count=...)` | Versioned registry metadata; class IDs remain in PLY. |
| `from_json(ids, text)` | Validate schema, registry, all class references, and point count before accepting data. |
| `recover_without_metadata(ids)` | Preserve PLY class IDs, with visible default/placeholder registry entries. |

Array inputs are validated before integer conversion, so invalid floats,
negative numbers, booleans, and out-of-range IDs cannot wrap into valid IDs.
History deltas own immutable copies. Exported arrays never alias live state.
Names are trimmed, length-limited, and reject ASCII control characters. Colors
must be three integer RGB channels. The JSON reader rejects duplicate fields,
unknown/missing fields, unsupported versions, and metadata over 1 MiB.

The caller must filter assignment indices by alive/visible/locked state before
assigning. The primitive intentionally has no hidden knowledge of geometry.

## Why class IDs are a side array

The current `SplatSet.concat()` intersects `extra` property keys, and generated
repairs can lack arbitrary extras. Keeping the only copy of class IDs there
would silently lose classifications. The engine's `_pack_set()` also currently
transports only standard splat attributes. A dedicated array makes lifecycle
and transport explicit. Attach it as `SplatSet.extra['splat_class']` only at the
PLY export boundary, then read it back into the dedicated state on open.

Memory overhead of the ID array is two bytes per splat (roughly 14 MB for seven
million splats), plus registry and sparse history. Visibility masks and GPU
streams have additional costs. Do not describe this as a memory optimization.

## Document and history integration

1. Instantiate classification alongside `Document.base` and `alive`.
2. Add a metadata edit type to `Edit` that stores `AssignmentDelta` or prior/new
   class definitions. Do not run geometry `alive` toggling for these edits.
3. Include class metadata edits in the existing dirty/save state and history.
4. Every append/truncate must update class IDs at the same boundary. Default
   newly reconstructed repair/fill output to Unclassified. Transforms preserve
   source IDs exactly. Clone output should inherit its donor class only when
   donor-to-output provenance is explicit; otherwise default Unclassified.
5. Save the truncated class tail with the generated geometry tail during undo,
   then append both on redo. Test edits on generated splats before undoing
   their original append.
6. Class creation undo must not recycle IDs. The first UI can keep unused class
   definitions rather than expose class deletion. Registry deletion/remapping
   requires a dedicated later operation with its own history rules.

Do not use UI selection-history snapshots to copy the whole class-ID array on
every brush stroke. Class assignments belong to the engine edit stack, using
sparse deltas; UI selection history should retain only normal selection state.

## Engine protocol and renderer integration

Open returns the registry and class IDs through a typed/binary payload or a
dedicated query, not millions of JSON numbers. Assignment and undo return
changed indices/IDs and updated counts. Appended geometry carries the exact
new class IDs; the existing float geometry packet does not yet do that.

The renderer's SplatDocument mirrors IDs and class definitions. Combine class
visibility with `alive` and the crop mask to derive the working visibility
mask. Reuse the existing hidden/deleted rendering stream for display culling,
but never mutate Document.alive because a checkbox was unchecked.

On visibility changes:

- Remove newly hidden members from selection and invalidate projection,
  surface-depth, clone-picking, and navigation caches.
- Rebuild GPU visibility and selection streams for base and generated splats.
- Honor the same mask in Surface/Through brush, lasso, rectangle, Select All,
  Invert, framing, selection-inspection markers, and cleanup detector results.
- Intersect history-restored selections with current visibility; otherwise a
  subsequent Delete could remove hidden splats unintentionally.
- Give selection/inspection highlighting precedence over class colors.

Geometry edits need engine-side enforcement too. Repair, object growth,
cleanup, and clone logic currently consult the complete live cloud. Hidden
noise should not become a donor or be modified incidentally. Introduce a
working mask or explicit class filter for edit operations while preserving an
all-alive path for unfiltered export. Do not redefine `live_splats()` globally:
that would quietly omit hidden classes from export.

## Persistence

Proposed PLY property: `property ushort splat_class`. Existing PLY I/O already
preserves numeric extra properties; integration must attach the export's IDs
in exactly the same point order as exported geometry.

Proposed sidecar: `<scan filename>.classes.json`, schema version 1, fields:

```json
{"version":1,"property":"splat_class","point_count":3,"next_id":3,
 "classes":[
   {"id":0,"name":"Unclassified","color":[180,190,200],"visible":true},
   {"id":1,"name":"Floaters","color":[255,125,85],"visible":true},
   {"id":2,"name":"Surface haze","color":[235,195,65],"visible":false}
 ]}
```

This is registry/visibility metadata only: it does not contain paths to follow
or executable data. Point count is a useful mismatch check, not a content
fingerprint. A matching point count does not prove a sidecar belongs to a scan.
Registry IDs embedded in PLY remain authoritative for assignments.

Missing metadata uses visible default/placeholder classes, preserving IDs. A
malformed or mismatched sidecar should produce a clear warning and explicit
fallback, rather than silently dropping classification or rejecting geometry.

The file-save integration must design a durable two-file transaction or an
equivalent recovery protocol. A temporary write/rename for each file alone
does not make the pair atomic. Do not mark the document fully saved when only
one part succeeds. A future single project container can avoid this pairing.

The current `.splat` and `.spz` writers do not carry arbitrary class properties.
The first release must warn that these exports lose classifications and class
settings, and recommend PLY plus its sidecar for classified work. Do not write
unbound class arrays into a sidecar for quantized/reordered formats without
defining and testing identity/order checks.

## Required tests before enabling the feature

The standalone tests cover schema validation, sparse assignments and no-ops,
undo/redo preconditions, immutable deltas, generated-tail policies, registry
round trips, subset ordering, visibility independence, missing sidecars, and
invalid inputs. They do not yet prove app-level persistence or UI behavior.

Before exposing controls, add:

- PLY save/reopen with assignment IDs and registry settings.
- Geometry deletion/repair/transform interleaved with classification undo/redo.
- New generated splats' classes and renderer synchronization.
- All selection and picking modes respecting hidden classes.
- Hidden classes protected from incidental edits and repair donor selection.
- All-alive export versus explicitly requested visible-only export.
- Partial-save failure and stale/malformed/missing sidecar recovery.
- `.splat`/`.spz` classification-loss warnings.
- Real viewport class-color and visibility tests with a synthetic fixture.

Large-site streaming, spatial work areas, memory unloading, automatic class
assignment, and LAS/LAZ interoperability are outside this first feature.
