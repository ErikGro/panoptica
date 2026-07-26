import numpy as np


# Many-to-One Mapping, optionally One-to-Many as well
class InstanceLabelMap:
    """Creates a mapping between prediction labels and reference labels.

    By default the mapping is many-to-one: several prediction labels may point to the
    same reference label, but each prediction label points to exactly one reference
    label. Passing ``allow_multiple_refs=True`` to :meth:`add_labelmap_entry` lifts the
    second restriction, so a single prediction label may also be matched to several
    reference labels, as required by One-to-Many matching.

    The reference labels of a prediction are stored in insertion order. The first one is
    the *primary* reference: it is the label the prediction's voxels are relabeled to
    when the prediction array is rewritten by ``map_instance_labels``. Matchers that
    allow multiple references per prediction should therefore add the best-scoring
    reference first.

    Attributes:
        labelmap (dict[int, list[int]]): Dictionary storing the prediction-to-reference
            label mappings.

    Methods:
        add_labelmap_entry(pred_labels, ref_label, allow_multiple_refs): Adds a new entry mapping prediction labels to a reference label.
        get_pred_labels_matched_to_ref(ref_label): Retrieves prediction labels mapped to a given reference label.
        get_ref_labels_matched_to_pred(pred_label): Retrieves reference labels mapped to a given prediction label.
        get_primary_ref_label(pred_label): Retrieves the primary reference label of a prediction label.
        contains_pred(pred_label): Checks if a prediction label exists in the map.
        contains_ref(ref_label): Checks if a reference label exists in the map.
        contains_and(pred_label, ref_label): Checks if both a prediction and a reference label are in the map.
        contains_or(pred_label, ref_label): Checks if either a prediction or reference label is in the map.
        get_one_to_one_dictionary(): Returns the prediction-to-primary-reference mapping.
        get_multi_ref_dictionary(): Returns the full prediction-to-references mapping.
        get_ref_to_pred_dictionary(): Returns the inverted reference-to-predictions mapping.
        ref_labels(): Returns the sorted reference labels contained in the map.
        has_multi_ref_predictions(): Checks whether any prediction is matched to more than one reference.
    """

    __labelmap: dict[int, list[int]]

    def __init__(self) -> None:
        self.__labelmap = {}

    def add_labelmap_entry(
        self,
        pred_labels: list[int] | int,
        ref_label: int,
        allow_multiple_refs: bool = False,
    ):
        """Adds an entry that maps prediction labels to a reference label.

        Args:
            pred_labels (list[int] | int): List of prediction labels or a single prediction label.
            ref_label (int): Reference label to map to.
            allow_multiple_refs (bool): If True, a prediction label that is already
                mapped to a different reference label gains an additional reference
                instead of raising. Required for One-to-Many matching. Defaults to False,
                which preserves the strict many-to-one contract.

        Raises:
            TypeError: If `ref_label` is not an integer.
            ValueError: If `ref_label` is not positive.
            TypeError: If any `pred_labels` are not integers.
            ValueError: If any `pred_labels` are negative.
            Exception: If a prediction label is already mapped to a different reference label and `allow_multiple_refs` is False.
        """
        if not isinstance(pred_labels, list):
            pred_labels = [pred_labels]
        if not isinstance(ref_label, int):
            raise TypeError("add_labelmap_entry: got no int as ref_label")
        if ref_label <= 0:
            raise ValueError("add_labelmap_entry: got no positive int as ref_label")
        if not np.all([isinstance(r, int) for r in pred_labels]):
            raise TypeError("add_labelmap_entry: got no int as pred_label")
        if not np.all([r >= 0 for r in pred_labels]):
            raise ValueError("add_labelmap_entry: got a negative int as pred_label")
        for p in pred_labels:
            if p not in self.__labelmap:
                self.__labelmap[p] = [ref_label]
                continue
            if ref_label in self.__labelmap[p]:
                continue
            if not allow_multiple_refs:
                raise Exception(
                    f"You are mapping a prediction label to a reference label that was already assigned differently, got {str(self)} and you tried {pred_labels}, {ref_label}. The prediction label {p} is already mapped to {self[p]}. Pass allow_multiple_refs=True if this prediction is meant to match several references."
                )
            self.__labelmap[p].append(ref_label)

    def get_pred_labels_matched_to_ref(self, ref_label: int) -> list[int]:
        """Retrieves all prediction labels that map to a specified reference label.

        Args:
            ref_label (int): The reference label to search.

        Returns:
            list[int]: List of prediction labels mapped to `ref_label`.
        """
        return [k for k, v in self.__labelmap.items() if ref_label in v]

    def get_ref_labels_matched_to_pred(self, pred_label: int) -> list[int]:
        """Retrieves all reference labels that a specified prediction label maps to.

        Args:
            pred_label (int): The prediction label to search.

        Returns:
            list[int]: Reference labels mapped to `pred_label`, in insertion order, so
                the first entry is the primary reference. Empty if `pred_label` is
                unmatched.
        """
        return list(self.__labelmap.get(pred_label, []))

    def get_primary_ref_label(self, pred_label: int) -> int:
        """Retrieves the primary (first-assigned) reference label of a prediction label.

        Args:
            pred_label (int): The prediction label to search.

        Returns:
            int: The primary reference label.

        Raises:
            KeyError: If `pred_label` is not in the map.
        """
        return self.__labelmap[pred_label][0]

    def contains_pred(self, pred_label: int):
        """Checks if a prediction label exists in the map.

        Args:
            pred_label (int): The prediction label to search.

        Returns:
            bool: True if `pred_label` is in `labelmap`, otherwise False.
        """
        return pred_label in self

    def contains_ref(self, ref_label: int):
        """Checks if a reference label exists in the map.

        Args:
            ref_label (int): The reference label to search.

        Returns:
            bool: True if `ref_label` is mapped to by any prediction, otherwise False.
        """
        return any(ref_label in v for v in self.__labelmap.values())

    def contains_and(
        self, pred_label: int | None = None, ref_label: int | None = None
    ) -> bool:
        """Checks if both a prediction and a reference label are in the map.

        Args:
            pred_label (int | None): The prediction label to check.
            ref_label (int | None): The reference label to check.

        Returns:
            bool: True if both `pred_label` and `ref_label` are in the map; otherwise, False.
        """
        pred_in = True if pred_label is None else pred_label in self
        ref_in = True if ref_label is None else self.contains_ref(ref_label)
        return pred_in and ref_in

    def contains_or(
        self, pred_label: int | None = None, ref_label: int | None = None
    ) -> bool:
        """Checks if either a prediction or reference label is in the map.

        Args:
            pred_label (int | None): The prediction label to check.
            ref_label (int | None): The reference label to check.

        Returns:
            bool: True if either `pred_label` or `ref_label` are in the map; otherwise, False.
        """
        pred_in = True if pred_label is None else pred_label in self
        ref_in = True if ref_label is None else self.contains_ref(ref_label)
        return pred_in or ref_in

    def get_one_to_one_dictionary(self) -> dict[int, int]:
        """Returns the prediction-to-primary-reference mapping.

        Every prediction label appears exactly once, mapped to its primary reference, so
        the result is directly usable for relabeling an integer label array. Additional
        references of a prediction are not represented; use
        :meth:`get_multi_ref_dictionary` for those.

        Returns:
            dict[int, int]: The prediction-to-primary-reference label mapping.
        """
        return {p: refs[0] for p, refs in self.__labelmap.items()}

    def get_multi_ref_dictionary(self) -> dict[int, list[int]]:
        """Returns a copy of the full prediction-to-references mapping.

        Returns:
            dict[int, list[int]]: Each prediction label mapped to all of its reference
                labels, primary reference first.
        """
        return {p: list(refs) for p, refs in self.__labelmap.items()}

    def get_ref_to_pred_dictionary(self) -> dict[int, list[int]]:
        """Returns the inverted mapping from reference labels to prediction labels.

        Returns:
            dict[int, list[int]]: Each matched reference label mapped to the sorted list
                of prediction labels matched to it.
        """
        inverted: dict[int, list[int]] = {}
        for pred_label, ref_labels in self.__labelmap.items():
            for ref_label in ref_labels:
                inverted.setdefault(ref_label, []).append(pred_label)
        return {ref: sorted(preds) for ref, preds in inverted.items()}

    def ref_labels(self) -> list[int]:
        """Returns the sorted, unique reference labels contained in the map.

        Returns:
            list[int]: The matched reference labels.
        """
        return sorted({ref for refs in self.__labelmap.values() for ref in refs})

    def has_multi_ref_predictions(self) -> bool:
        """Checks whether any prediction label is matched to more than one reference.

        Returns:
            bool: True if the map is not many-to-one.
        """
        return any(len(refs) > 1 for refs in self.__labelmap.values())

    def __str__(self) -> str:
        return str(
            [
                str(tuple(self.get_pred_labels_matched_to_ref(v))) + " -> " + str(v)
                for v in self.ref_labels()
            ]
        )

    def __repr__(self) -> str:
        return str(self)

    def __contains__(self, key: int) -> bool:
        return key in self.__labelmap

    def __getitem__(self, key: int) -> int:
        """Returns the primary reference label of a prediction label.

        Use :meth:`get_ref_labels_matched_to_pred` to obtain every reference label of a
        prediction under One-to-Many matching.
        """
        return self.__labelmap[key][0]

    def __iter__(self):
        return iter(self.__labelmap.keys())

    def __setitem__(self, key: int, value: int):
        raise Exception("Attempting to alter read-only value")

    def __len__(self) -> int:
        return self.__labelmap.__len__()

    def items(self):
        """Prediction-to-primary-reference items, i.e. the one-to-one view."""
        return self.get_one_to_one_dictionary().items()

    def keys(self) -> list[int]:
        return list(self.__labelmap.keys())

    def values(self) -> list[int]:
        """Every reference label appearing in the map, flattened, with duplicates."""
        return [ref for refs in self.__labelmap.values() for ref in refs]

    # Make all variables read-only!
    def __setattr__(self, attr, value):
        """Overrides attribute setting to make attributes read-only after initialization.

        Args:
            attr (str): Attribute name.
            value (Any): Attribute value.

        Raises:
            Exception: If trying to alter an existing attribute.
        """
        if hasattr(self, attr):
            raise Exception("Attempting to alter read-only value")

        self.__dict__[attr] = value