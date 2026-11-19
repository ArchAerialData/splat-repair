"""Standalone metadata/state tests; no renderer or production Document wiring."""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'splatcore'))
from classification import Classification, ClassDefinition, AssignmentDelta, MAX_METADATA_BYTES


class ClassificationTests(unittest.TestCase):
    def test_default_classes_and_no_geometry_ownership(self):
        state = Classification.empty(6)
        self.assertEqual([c.name for c in state.classes], ['Unclassified', 'Floaters', 'Surface haze'])
        self.assertEqual(state.export_ids().dtype, np.uint16)
        self.assertTrue(state.visibility_mask().all())
        ids = state.export_ids(); ids[0] = 1
        self.assertEqual(state.export_ids()[0], 0, 'Export cannot alias live state')

    def test_sparse_assignment_undo_redo(self):
        state = Classification(np.array([0, 1, 0, 2, 0], np.uint16))
        delta = state.assign([3, 0, 0, 1], 2)
        self.assertEqual(delta.indices.tolist(), [0, 1])
        self.assertEqual(delta.before.tolist(), [0, 1])
        self.assertEqual(state.export_ids().tolist(), [2, 2, 0, 2, 0])
        state.apply_delta(delta, undo=True)
        self.assertEqual(state.export_ids().tolist(), [0, 1, 0, 2, 0])
        state.apply_delta(delta)
        self.assertEqual(state.export_ids().tolist(), [2, 2, 0, 2, 0])
        with self.assertRaises(ValueError):
            delta.before.setflags(write=True)

    def test_no_op_does_not_create_history(self):
        state = Classification.empty(3)
        self.assertIsNone(state.assign([1], 0))
        self.assertIsNone(state.assign([], 1))

    def test_replay_refuses_stale_history_atomically(self):
        state = Classification.empty(3)
        delta = state.assign([0, 1], 1)
        state.assign([1], 2)
        before = state.export_ids()
        with self.assertRaises(ValueError):
            state.apply_delta(delta, undo=True)
        np.testing.assert_array_equal(before, state.export_ids())

    def test_invalid_assignment_is_atomic(self):
        state = Classification.empty(4)
        for ids, target in [([0, 5], 1), ([-1], 1), ([0.5], 1), ([True], 1), ([0], 8), ([0], True)]:
            with self.assertRaises(ValueError):
                state.assign(ids, target)
            self.assertFalse(state.export_ids().any())

    def test_append_transform_and_tail_replay(self):
        state = Classification(np.array([0, 1, 2], np.uint16))
        state.append(2)  # Newly reconstructed splats.
        state.append(2, state.export_ids([2, 1]))  # Transformed copies inherit.
        self.assertEqual(state.export_ids().tolist(), [0, 1, 2, 0, 0, 2, 1])
        tail = state.truncate(2)
        self.assertEqual(state.count, 5)
        state.append(2, tail)
        self.assertEqual(state.export_ids().tolist(), [0, 1, 2, 0, 0, 2, 1])
        self.assertEqual(state.truncate(0).tolist(), [])
        self.assertEqual(state.count, 7)

    def test_append_and_truncate_validate_before_mutation(self):
        state = Classification.empty(3)
        for call in [lambda: state.append(1, [9]), lambda: state.append(2, [1]),
                     lambda: state.truncate(4), lambda: state.append(-1)]:
            with self.assertRaises(ValueError):
                call()
            self.assertEqual(state.count, 3)

    def test_visibility_does_not_remove_or_reassign(self):
        state = Classification([0, 1, 2, 1])
        prior = state.replace_class(replace(state.classes[1], visible=False))
        self.assertEqual(state.visibility_mask().tolist(), [True, False, True, False])
        self.assertEqual(state.export_ids().tolist(), [0, 1, 2, 1])
        state.replace_class(prior)
        self.assertTrue(state.visibility_mask().all())

    def test_registry_edit_and_metadata_round_trip(self):
        state = Classification.empty(3)
        custom = state.create_class('Stockpile A', [40, 110, 240])
        self.assertEqual(custom.id, 3)
        state.assign([2], custom.id)
        state.replace_class(replace(custom, visible=False))
        restored = Classification.from_json(state.export_ids(), state.to_json())
        self.assertEqual(restored.classes, state.classes)
        np.testing.assert_array_equal(restored.export_ids(), state.export_ids())
        self.assertEqual(restored.create_class('Next', [1, 2, 3]).id, 4)

    def test_export_subset_order_is_explicit(self):
        state = Classification([0, 1, 2, 1])
        subset = state.export_ids([3, 2])
        restored = Classification.from_json(subset, state.to_json(exported_count=2))
        self.assertEqual(restored.export_ids().tolist(), [1, 2])
        self.assertEqual(state.visibility_mask().size, 4)

    def test_numeric_ply_column_validation(self):
        self.assertEqual(Classification(np.array([0., 1., 2.])).export_ids().tolist(), [0, 1, 2])
        for values in [[-1], [65536], [np.nan], [np.inf], [1.2], ['1'], [True], [[1]]]:
            with self.assertRaises(ValueError):
                Classification(values)

    def test_missing_metadata_preserves_unknown_ids(self):
        state = Classification.recover_without_metadata([0, 17, 65535])
        self.assertEqual(state.export_ids().tolist(), [0, 17, 65535])
        self.assertTrue(state.visibility_mask().all())
        self.assertEqual(state.classes[-1].name, 'Class 65535')
        with self.assertRaises(ValueError):
            state.create_class('Exhausted', [1, 2, 3])

    def test_metadata_strict_validation(self):
        state = Classification.empty(1)
        original = json.loads(state.to_json())
        bad = [dict(original, version=True), dict(original, version=2),
               dict(original, point_count=2), dict(original, next_id=2),
               dict(original, property='other'), dict(original, unexpected=True),
               dict(original, classes=original['classes'] * 2)]
        for data in bad:
            with self.assertRaises(ValueError):
                Classification.from_json([0], json.dumps(data))
        with self.assertRaises(ValueError):
            Classification.from_json([0], '{"version":1,"version":1}')
        with self.assertRaises(ValueError):
            Classification.from_json([0], ' ' * (MAX_METADATA_BYTES + 1))

    def test_registry_input_validation(self):
        for args in [(True, 'A', [1, 2, 3]), (3, ' A', [1, 2, 3]),
                     (3, 'A\nB', [1, 2, 3]), (3, 'A', [1, 2, 256]),
                     (3, 'A', [True, 2, 3]), (3, 'A', [1, 2, 3], 1)]:
            with self.assertRaises(ValueError):
                ClassDefinition(*args)
        with self.assertRaises(ValueError):
            Classification([0], classes=[ClassDefinition(1, 'A', [1, 2, 3])])

    def test_delta_validation(self):
        for indices, before, after in [([1, 1], [0, 0], 1), ([2, 1], [0, 0], 1),
                                       ([-1], [0], 1), ([1], [0, 0], 1), ([1], [0], 65536)]:
            with self.assertRaises(ValueError):
                AssignmentDelta(np.asarray(indices), np.asarray(before), after)


if __name__ == '__main__':
    unittest.main(verbosity=2)
