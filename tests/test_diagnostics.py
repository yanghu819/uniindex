import torch
import pytest

from uniindex.diagnostics import _image_condition_tokens, _image_dependence_margins, _trace_step_requests


def test_image_condition_tokens_builds_control_batches():
    image_tokens = torch.tensor(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
    )

    assert torch.equal(
        _image_condition_tokens(image_tokens, mode="true_image", codebook_size=10),
        image_tokens,
    )
    assert torch.equal(
        _image_condition_tokens(image_tokens, mode="shuffled_image", codebook_size=10),
        image_tokens[[2, 0, 1]],
    )

    torch.manual_seed(0)
    random_tokens = _image_condition_tokens(image_tokens, mode="random_image_tokens", codebook_size=10)
    assert random_tokens.shape == image_tokens.shape
    assert int(random_tokens.min()) >= 0
    assert int(random_tokens.max()) < 10


def test_image_dependence_margins_compare_true_to_controls():
    results = [
        {
            "progress": 0.95,
            "mode": "true_image",
            "image_to_text_exact_match": 0.5,
            "image_to_text_token_accuracy": 0.6,
            "image_to_text_label_accuracy_constrained": 0.7,
        },
        {
            "progress": 0.95,
            "mode": "shuffled_image",
            "image_to_text_exact_match": 0.25,
            "image_to_text_token_accuracy": 0.2,
            "image_to_text_label_accuracy_constrained": 0.3,
        },
        {
            "progress": 0.95,
            "mode": "random_image_tokens",
            "image_to_text_exact_match": 0.1,
            "image_to_text_token_accuracy": 0.4,
            "image_to_text_label_accuracy_constrained": 0.2,
        },
    ]

    assert _image_dependence_margins(results) == [
        {
            "progress": 0.95,
            "true_minus_shuffled_label": 0.39999999999999997,
            "true_minus_shuffled_token": 0.39999999999999997,
            "true_minus_shuffled_exact": 0.25,
            "true_minus_random_label": 0.49999999999999994,
            "true_minus_random_token": 0.19999999999999996,
            "true_minus_random_exact": 0.4,
        }
    ]


def test_trace_step_requests_map_progress_to_nearest_sampler_step():
    assert _trace_step_requests(32, (0.5, 0.75, 0.9, 0.95)) == {
        16: [0.5],
        24: [0.75],
        29: [0.9],
        30: [0.95],
    }


def test_trace_step_requests_validate_bounds():
    with pytest.raises(ValueError, match="trace progress"):
        _trace_step_requests(32, (1.2,))
