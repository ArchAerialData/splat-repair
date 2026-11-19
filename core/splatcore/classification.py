"""Classification state independent of geometry, rendering and file I/O.

IDs stay parallel to Document's stable indices, including deleted splats.
The caller stores AssignmentDelta in its existing history, pairs metadata
with a PLY export, and supplies the alive/crop masks for visibility. This
module never changes geometry or deletes/unloads a splat.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

PROPERTY_NAME = 'splat_class'
SCHEMA_VERSION = 1
MAX_CLASS_ID = 65535
MAX_METADATA_BYTES = 1024 * 1024


def _integer(value, name, minimum=0, maximum=MAX_CLASS_ID):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}')
    return value


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'{name} has missing or unexpected fields')


@dataclass(frozen=True)
class ClassDefinition:
    id: int
    name: str
    color: tuple[int, int, int]
    visible: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'id', _integer(self.id, 'class ID'))
        if (not isinstance(self.name, str) or not self.name or len(self.name) > 80
                or self.name != self.name.strip() or any(ord(c) < 32 or ord(c) == 127 for c in self.name)):
            raise ValueError('class name must be 1-80 characters, trimmed and without control characters')
        if not isinstance(self.color, (list, tuple)) or len(self.color) != 3:
            raise ValueError('class color must contain three RGB integers')
        object.__setattr__(self, 'color', tuple(_integer(c, 'color channel', maximum=255) for c in self.color))
        if not isinstance(self.visible, bool):
            raise ValueError('class visibility must be a boolean')

    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'color': list(self.color), 'visible': self.visible}

    @classmethod
    def from_dict(cls, value):
        _keys(value, ('id', 'name', 'color', 'visible'), 'class definition')
        return cls(**value)


DEFAULT_CLASSES = (
    ClassDefinition(0, 'Unclassified', (180, 190, 200)),
    ClassDefinition(1, 'Floaters', (255, 125, 85)),
    ClassDefinition(2, 'Surface haze', (235, 195, 65)),
)


def _class_array(values):
    """Accept numeric PLY columns, checking before uint16 conversion can wrap."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in 'iuf':
        raise ValueError('class IDs must be a one-dimensional numeric array')
    if raw.size and (not np.isfinite(raw).all() or np.any(raw < 0)
                     or np.any(raw > MAX_CLASS_ID) or np.any(raw != np.floor(raw))):
        raise ValueError('class IDs must be finite integers between 0 and 65535')
    return raw.astype(np.uint16, copy=True)


def _frozen_copy(array):
    # An immutable bytes backing prevents callers re-enabling write access.
    return np.frombuffer(array.tobytes(), dtype=array.dtype)


@dataclass(frozen=True)
class AssignmentDelta:
    """Sparse reversible edit; geometry and alive flags are not part of it."""
    indices: np.ndarray
    before: np.ndarray
    after: int

    def __post_init__(self):
        indices = np.asarray(self.indices)
        if indices.ndim != 1 or indices.dtype.kind not in 'iu':
            raise ValueError('assignment indices must be a one-dimensional integer array')
        if np.any(indices > np.iinfo(np.int64).max):
            raise ValueError('assignment index exceeds int64')
        indices = indices.astype(np.int64)
        if np.any(indices < 0) or np.any(indices[1:] <= indices[:-1]):
            raise ValueError('assignment indices must be nonnegative, sorted and unique')
        before = _class_array(self.before)
        if len(before) != len(indices):
            raise ValueError('assignment delta lengths differ')
        object.__setattr__(self, 'indices', _frozen_copy(indices))
        object.__setattr__(self, 'before', _frozen_copy(before))
        object.__setattr__(self, 'after', _integer(self.after, 'class ID'))

    @property
    def nbytes(self):
        return self.indices.nbytes + self.before.nbytes


class Classification:
    def __init__(self, ids, classes=DEFAULT_CLASSES, next_id=None):
        definitions = tuple(classes)
        if not definitions or not all(isinstance(c, ClassDefinition) for c in definitions):
            raise ValueError('classes must contain validated class definitions')
        self._classes = {c.id: c for c in definitions}
        if len(self._classes) != len(definitions) or 0 not in self._classes:
            raise ValueError('class IDs must be unique and include Unclassified (0)')
        self._ids = _class_array(ids)
        self._validate_registered(self._ids)
        minimum = max(self._classes) + 1
        self._next_id = minimum if next_id is None else _integer(next_id, 'next ID', minimum, MAX_CLASS_ID + 1)

    @classmethod
    def empty(cls, count):
        count = _integer(count, 'splat count', maximum=np.iinfo(np.intp).max)
        return cls(np.zeros(count, np.uint16))

    @property
    def count(self):
        return len(self._ids)

    @property
    def classes(self):
        return tuple(self._classes[k] for k in sorted(self._classes))

    def export_ids(self, indices=None):
        """Return an owned array; optional indices must match export point order."""
        return self._ids.copy() if indices is None else self._ids[self._indices(indices)].copy()

    def _validate_registered(self, ids):
        if any(int(i) not in self._classes for i in np.unique(ids)):
            raise ValueError('class IDs reference a class missing from the registry')

    def _indices(self, indices):
        raw = np.asarray(indices)
        if raw.ndim == 1 and not raw.size:
            return np.zeros(0, np.int64)
        if raw.ndim != 1 or raw.dtype.kind not in 'iu':
            raise ValueError('indices must be a one-dimensional integer array')
        if raw.size and (np.any(raw < 0) or np.any(raw >= self.count)):
            raise ValueError('splat index is outside the current document')
        return raw.astype(np.int64)

    def assign(self, indices, class_id):
        class_id = _integer(class_id, 'class ID')
        if class_id not in self._classes:
            raise ValueError('unknown class ID')
        idx = np.unique(self._indices(indices))
        idx = idx[self._ids[idx] != class_id]
        if not len(idx):
            return None
        delta = AssignmentDelta(idx, self._ids[idx], class_id)
        self._ids[idx] = class_id
        return delta

    def apply_delta(self, delta: AssignmentDelta, *, undo=False):
        """Replay only against its expected state; stale history fails atomically."""
        idx = self._indices(delta.indices)
        self._validate_registered(delta.before)
        if delta.after not in self._classes:
            raise ValueError('assignment target class is missing')
        expected = delta.after if undo else delta.before
        if not np.all(self._ids[idx] == expected):
            raise ValueError('classification history no longer matches the document')
        self._ids[idx] = delta.before if undo else delta.after

    def append(self, count, inherited=None):
        """New reconstruction defaults to 0; transforms pass exact source IDs."""
        count = _integer(count, 'appended count', maximum=np.iinfo(np.intp).max)
        added = np.zeros(count, np.uint16) if inherited is None else _class_array(inherited)
        if len(added) != count:
            raise ValueError('appended class count does not match appended geometry')
        self._validate_registered(added)
        self._ids = np.concatenate((self._ids, added))

    def truncate(self, count):
        """Undo geometry append, returning its exact class tail for redo."""
        count = _integer(count, 'truncated count', maximum=self.count)
        end = self.count - count
        tail = self._ids[end:].copy()
        self._ids = self._ids[:end].copy()
        return tail

    def create_class(self, name, color):
        if self._next_id > MAX_CLASS_ID:
            raise ValueError('all classification IDs have been allocated')
        definition = ClassDefinition(self._next_id, name, color)
        self._classes[definition.id] = definition
        self._next_id += 1
        return definition

    def replace_class(self, definition):
        """Return prior definition for document history. IDs cannot be removed."""
        if not isinstance(definition, ClassDefinition) or definition.id not in self._classes:
            raise ValueError('replacement must name an existing class')
        prior = self._classes[definition.id]
        self._classes[definition.id] = definition
        return prior

    def visibility_mask(self):
        table = np.zeros(MAX_CLASS_ID + 1, bool)
        for definition in self.classes:
            table[definition.id] = definition.visible
        return table[self._ids]

    def to_json(self, *, exported_count=None):
        count = self.count if exported_count is None else _integer(exported_count, 'exported count', maximum=self.count)
        data = {'version': SCHEMA_VERSION, 'property': PROPERTY_NAME, 'point_count': count,
                'next_id': self._next_id, 'classes': [c.to_dict() for c in self.classes]}
        text = json.dumps(data, ensure_ascii=True, separators=(',', ':'))
        if len(text.encode('utf-8')) > MAX_METADATA_BYTES:
            raise ValueError('classification metadata exceeds its size limit')
        return text

    @classmethod
    def from_json(cls, ids, text):
        if not isinstance(text, (str, bytes)):
            raise ValueError('classification metadata must be JSON text')
        if len(text.encode('utf-8') if isinstance(text, str) else text) > MAX_METADATA_BYTES:
            raise ValueError('classification metadata exceeds its size limit')

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate metadata field')
                result[key] = value
            return result

        data = json.loads(text, object_pairs_hook=unique_object)
        _keys(data, ('version', 'property', 'point_count', 'next_id', 'classes'), 'classification metadata')
        if _integer(data['version'], 'schema version') != SCHEMA_VERSION or data['property'] != PROPERTY_NAME:
            raise ValueError('unsupported classification metadata schema')
        array = _class_array(ids)
        if _integer(data['point_count'], 'point count', maximum=np.iinfo(np.intp).max) != len(array):
            raise ValueError('classification metadata point count does not match the scan')
        if not isinstance(data['classes'], list) or len(data['classes']) > MAX_CLASS_ID + 1:
            raise ValueError('invalid class registry')
        return cls(array, [ClassDefinition.from_dict(c) for c in data['classes']], data['next_id'])

    @classmethod
    def recover_without_metadata(cls, ids):
        """Keep IDs if the sidecar is missing; recovered classes are all visible."""
        array = _class_array(ids)
        definitions = {c.id: c for c in DEFAULT_CLASSES}
        for value in np.unique(array):
            value = int(value)
            if value not in definitions:
                definitions[value] = ClassDefinition(value, f'Class {value}', (150, 175, 205))
        return cls(array, definitions.values())
