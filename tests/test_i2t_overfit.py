import torch

from uniindex.i2t_overfit import _image_condition, _record_steps, _summarize_prediction_records


def test_record_steps_includes_start_regular_intervals_and_end():
    assert _record_steps(total_steps=10, eval_every=4) == [0, 4, 8, 10]
    assert _record_steps(total_steps=12, eval_every=4) == [0, 4, 8, 12]


def test_image_condition_shuffles_and_randomizes_without_shape_change():
    image_tokens = torch.tensor([[1, 2], [3, 4], [5, 6]])

    shuffled = _image_condition(image_tokens, mode="shuffled_image", codebook_size=10)
    random_tokens = _image_condition(image_tokens, mode="random_image_tokens", codebook_size=10)

    assert torch.equal(shuffled, torch.tensor([[5, 6], [1, 2], [3, 4]]))
    assert random_tokens.shape == image_tokens.shape
    assert random_tokens.ge(0).all()
    assert random_tokens.lt(10).all()


def test_summarize_prediction_records_averages_sample_metrics():
    records = [
        {
            "free_exact": True,
            "token_accuracy": 1.0,
            "label_correct": True,
            "true_label_margin": 2.0,
        },
        {
            "free_exact": False,
            "token_accuracy": 0.25,
            "label_correct": False,
            "true_label_margin": -1.0,
        },
    ]

    summary = _summarize_prediction_records(records)

    assert summary["image_to_text_exact_match"] == 0.5
    assert summary["image_to_text_token_accuracy"] == 0.625
    assert summary["image_to_text_label_accuracy_constrained"] == 0.5
    assert summary["mean_true_label_margin"] == 0.5
