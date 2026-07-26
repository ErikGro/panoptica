"""Tests for One-to-Many instance matching (deg_M(g) <= 1, deg_M(p) unconstrained)."""

import numpy as np
import pytest

from panoptica import (
    InputType,
    MaxBipartiteMatching,
    NaiveThresholdMatching,
    OneToManyMatching,
    Panoptica_Evaluator,
)
from panoptica.instance_matcher import map_instance_labels
from panoptica.utils.instancelabelmap import InstanceLabelMap
from panoptica.utils.processing_pair import UnmatchedInstancePair

# Two adjacent 6x6 references (36 voxels each) covered by a single 6x14 prediction
# (84 voxels), so IoU(p, g) = 36 / 84 for both references.
IOU = 36 / 84


def _ref_two_instances() -> np.ndarray:
    ref = np.zeros((20, 20), dtype=np.uint8)
    ref[2:8, 2:8] = 1
    ref[2:8, 10:16] = 2
    return ref


def _merged_prediction() -> np.ndarray:
    pred = np.zeros((20, 20), dtype=np.uint8)
    pred[2:8, 2:16] = 1
    return pred


def _evaluate(prediction_arr, reference_arr, matcher):
    evaluator = Panoptica_Evaluator(
        expected_input=InputType.UNMATCHED_INSTANCE,
        instance_matcher=matcher,
        verbose=False,
        log_times=False,
    )
    result = evaluator.evaluate(prediction_arr, reference_arr)["ungrouped"]
    return result


def _assert_counts(result, tp, fp, fn):
    assert (result.tp, result.fp, result.fn) == (tp, fp, fn)


class TestOneToManyMatching:
    def test_merged_prediction_credits_both_references(self):
        """One prediction covering two references yields two TPs and no FP."""
        result = _evaluate(
            _merged_prediction(),
            _ref_two_instances(),
            OneToManyMatching(matching_threshold=0.25),
        )
        _assert_counts(result, tp=2, fp=0, fn=0)
        # SQ is the mean IoU against the whole undivided prediction, not against
        # per-reference fragments of it.
        assert result.sq == pytest.approx(IOU)
        assert result.rq == pytest.approx(1.0)
        assert result.pq == pytest.approx(IOU)

    def test_threshold_above_achievable_iou_rejects_all(self):
        result = _evaluate(
            _merged_prediction(),
            _ref_two_instances(),
            OneToManyMatching(matching_threshold=0.5),
        )
        _assert_counts(result, tp=0, fp=1, fn=2)

    def test_unmatched_prediction_stays_false_positive(self):
        prediction_arr = _merged_prediction()
        prediction_arr[14:18, 14:18] = 2
        result = _evaluate(
            prediction_arr,
            _ref_two_instances(),
            OneToManyMatching(matching_threshold=0.25),
        )
        _assert_counts(result, tp=2, fp=1, fn=0)
        assert result.sq == pytest.approx(IOU)
        assert result.rq == pytest.approx(2 / 2.5)

    def test_perfect_prediction_is_unaffected(self):
        reference_arr = _ref_two_instances()
        result = _evaluate(
            reference_arr.copy(),
            reference_arr,
            OneToManyMatching(matching_threshold=0.5),
        )
        _assert_counts(result, tp=2, fp=0, fn=0)
        assert result.pq == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "strict_threshold, expected",
        [(True, (0, 1, 2)), (False, (2, 0, 0))],
    )
    def test_strict_threshold_at_exactly_tau(self, strict_threshold, expected):
        """The candidate edge set uses a strict inequality by default."""
        result = _evaluate(
            _merged_prediction(),
            _ref_two_instances(),
            OneToManyMatching(
                matching_threshold=IOU, strict_threshold=strict_threshold
            ),
        )
        _assert_counts(result, *expected)

    def test_labelmap_holds_one_prediction_for_two_references(self):
        prediction_arr = _merged_prediction()
        prediction_arr[14:18, 14:18] = 2
        pair = UnmatchedInstancePair(
            prediction_arr=prediction_arr, reference_arr=_ref_two_instances()
        )
        matcher = OneToManyMatching(matching_threshold=0.25)
        labelmap = matcher._match_instances(pair, None, matching_threshold=0.25)

        assert labelmap.has_multi_ref_predictions()
        assert labelmap.get_multi_ref_dictionary() == {1: [1, 2]}
        assert labelmap.get_one_to_one_dictionary() == {1: 1}
        assert labelmap.get_ref_to_pred_dictionary() == {1: [1], 2: [1]}
        assert labelmap.get_ref_labels_matched_to_pred(1) == [1, 2]
        assert labelmap.get_pred_labels_matched_to_ref(2) == [1]
        assert labelmap.get_primary_ref_label(1) == 1
        assert labelmap[1] == 1
        assert labelmap.contains_pred(1) and labelmap.contains_ref(2)
        assert not labelmap.contains_pred(2)
        assert labelmap.ref_labels() == [1, 2]

    def test_map_instance_labels_derives_counts_from_the_labelmap(self):
        prediction_arr = _merged_prediction()
        prediction_arr[14:18, 14:18] = 2
        pair = UnmatchedInstancePair(
            prediction_arr=prediction_arr, reference_arr=_ref_two_instances()
        )
        matcher = OneToManyMatching(matching_threshold=0.25)
        labelmap = matcher._match_instances(pair, None, matching_threshold=0.25)
        matched = map_instance_labels(pair.copy(), labelmap)

        assert matched.matched_instances == [1, 2]
        assert matched.missed_reference_labels == []
        assert matched.missed_prediction_labels == [3]
        # 2 matched references + 1 unmatched prediction, so fp = 3 - tp = 1.
        assert matched.n_pred_instances == 3
        assert matched.prediction_labels_per_ref == {1: [1], 2: [1]}
        # Reference 2 is evaluated against the prediction relabeled to reference 1.
        assert matched.prediction_labels_for(2) == [1]
        assert matched.has_shared_predictions
        assert matched.copy().prediction_labels_per_ref == (
            matched.prediction_labels_per_ref
        )

    def test_matched_pair_without_shared_predictions_is_transparent(self):
        reference_arr = _ref_two_instances()
        pair = UnmatchedInstancePair(
            prediction_arr=reference_arr.copy(), reference_arr=reference_arr
        )
        matcher = OneToManyMatching(matching_threshold=0.5)
        labelmap = matcher._match_instances(pair, None, matching_threshold=0.5)
        matched = map_instance_labels(pair.copy(), labelmap)

        assert not labelmap.has_multi_ref_predictions()
        assert matched.prediction_labels_per_ref is None
        assert not matched.has_shared_predictions
        assert matched.prediction_labels_for(2) == [2]

    def test_empty_inputs_produce_an_empty_labelmap(self):
        empty = np.zeros((20, 20), dtype=np.uint8)
        pair = UnmatchedInstancePair(
            prediction_arr=_merged_prediction(), reference_arr=empty
        )
        matcher = OneToManyMatching(matching_threshold=0.25)
        assert len(matcher._match_instances(pair, None, matching_threshold=0.25)) == 0


class TestExistingMatchersUnchanged:
    """The one-to-many extension must not alter any many-to-one or one-to-one result."""

    @pytest.mark.parametrize(
        "matcher_cls", [NaiveThresholdMatching, MaxBipartiteMatching]
    )
    def test_merged_prediction_still_costs_one_false_negative(self, matcher_cls):
        result = _evaluate(
            _merged_prediction(),
            _ref_two_instances(),
            matcher_cls(matching_threshold=0.25),
        )
        _assert_counts(result, tp=1, fp=0, fn=1)
        assert result.sq == pytest.approx(IOU)
        assert result.rq == pytest.approx(1 / 1.5)

    def test_many_to_one_absorbs_the_extra_prediction(self):
        reference_arr = np.zeros((20, 20), dtype=np.uint8)
        reference_arr[2:8, 2:16] = 1
        split = np.zeros((20, 20), dtype=np.uint8)
        split[2:8, 2:8] = 1
        split[2:8, 10:16] = 2

        result = _evaluate(
            split,
            reference_arr,
            NaiveThresholdMatching(matching_threshold=0.25, allow_many_to_one=True),
        )
        _assert_counts(result, tp=1, fp=0, fn=0)
        assert result.sq == pytest.approx(72 / 84)

        result = _evaluate(
            split, reference_arr, NaiveThresholdMatching(matching_threshold=0.25)
        )
        _assert_counts(result, tp=1, fp=1, fn=0)

    def test_add_labelmap_entry_still_rejects_remapping_by_default(self):
        labelmap = InstanceLabelMap()
        labelmap.add_labelmap_entry(1, 1)
        labelmap.add_labelmap_entry(1, 1)  # idempotent
        with pytest.raises(Exception, match="already assigned differently"):
            labelmap.add_labelmap_entry(1, 2)

        labelmap.add_labelmap_entry(1, 2, allow_multiple_refs=True)
        assert labelmap.get_ref_labels_matched_to_pred(1) == [1, 2]

    def test_many_to_one_labelmap_is_still_many_to_one(self):
        labelmap = InstanceLabelMap()
        labelmap.add_labelmap_entry([1, 2], 1)
        assert not labelmap.has_multi_ref_predictions()
        assert labelmap.get_pred_labels_matched_to_ref(1) == [1, 2]
        assert labelmap.get_one_to_one_dictionary() == {1: 1, 2: 1}
        assert dict(labelmap.items()) == {1: 1, 2: 1}
        assert labelmap.values() == [1, 1]
        assert labelmap.keys() == [1, 2]
        assert len(labelmap) == 2
