import pytest

from reqlab.reconcile import partition


def test_partition_preserves_every_candidate_once():
    assert partition({"groups": [[2, 1]], "rejected": [3]}, 3) == ([[2, 1]], [3])
    for reply in ({"groups": [[1]], "rejected": []}, {"groups": [[1, 2]], "rejected": [2]},
                  {"groups": [[True, 2]], "rejected": []}, {"groups": [[1, 9]], "rejected": []}):
        with pytest.raises(ValueError):
            partition(reply, 2)
