from qwen.benchmark import _evenly_spaced_spans


def test_benchmark_batches_cover_full_split():
    spans = _evenly_spaced_spans(dataset_size=100, batch_size=10, requested_batches=3)

    assert spans == [(0, 10), (40, 50), (90, 100)]


def test_benchmark_batches_do_not_repeat_when_request_exceeds_population():
    spans = _evenly_spaced_spans(dataset_size=5, batch_size=2, requested_batches=10)

    assert spans == [(0, 2), (2, 4), (4, 5)]


def test_benchmark_batch_selection_rejects_nonpositive_sizes():
    assert _evenly_spaced_spans(0, 1, 1) == []
    assert _evenly_spaced_spans(10, 0, 1) == []
    assert _evenly_spaced_spans(10, 1, 0) == []
