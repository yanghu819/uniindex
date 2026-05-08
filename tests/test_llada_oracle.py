import torch

from uniindex.llada_oracle import digit_prompts, infer_grid_shape, normalize_image_token_rows


def test_digit_prompts_formats_labels_and_digits():
    prompts = digit_prompts("digit {digit}: {label}")

    assert prompts[0] == "digit 0: zero"
    assert prompts[-1] == "digit 9: nine"
    assert len(prompts) == 10


def test_normalize_image_token_rows_accepts_tensor_dict_and_tuple():
    assert normalize_image_token_rows(torch.tensor([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    assert normalize_image_token_rows({"image_tokens": torch.tensor([5, 6])}) == [[5, 6]]
    assert normalize_image_token_rows((None, [7, 8, 9])) == [[7, 8, 9]]


def test_infer_grid_shape_prefers_requested_or_square():
    assert infer_grid_shape(256, 16, 16) == (16, 16)
    assert infer_grid_shape(256, None, None) == (16, 16)
    assert infer_grid_shape(192, 12, None) == (12, 16)
