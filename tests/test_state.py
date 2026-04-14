import torch

from uniindex.layout import position_valid_token_mask
from uniindex.state import apply_time_schedule, build_flm_clean_state, restore_image_tokens, sample_masked_noise


def test_build_flm_clean_state_returns_one_hot():
    targets = torch.tensor([[1, 3], [0, 2]])
    state = build_flm_clean_state(targets, vocab_size=5)
    assert state.shape == (2, 2, 5)
    assert torch.equal(state.argmax(dim=-1), targets)


def test_apply_time_schedule_uses_different_powers_per_modality():
    progress = torch.tensor([0.25, 0.81])
    modality_ids = torch.tensor([0, 0, 1])
    t_pos = apply_time_schedule(progress, modality_ids, image_time_power=1.0, label_time_power=0.5)
    expected = torch.tensor(
        [
            [0.25, 0.25, 0.5],
            [0.81, 0.81, 0.9],
        ]
    )
    assert torch.allclose(t_pos, expected, atol=1e-6)


def test_restore_image_tokens_maps_compact_ids_back_to_original_ids():
    tokens = torch.tensor([[0, 2, 1]])
    tokenizer_state = {"original_token_ids": torch.tensor([7, 11, 19])}
    restored = restore_image_tokens(tokens, tokenizer_state)
    assert torch.equal(restored, torch.tensor([[7, 19, 11]]))


def test_position_valid_token_mask_separates_image_and_label_subspaces():
    mask = position_valid_token_mask(image_seq_len=2, codebook_size=4, num_labels=3)
    assert mask.shape == (3, 7)
    assert torch.equal(mask[0], torch.tensor([True, True, True, True, False, False, False]))
    assert torch.equal(mask[1], torch.tensor([True, True, True, True, False, False, False]))
    assert torch.equal(mask[2], torch.tensor([False, False, False, False, True, True, True]))


def test_sample_masked_noise_zeroes_invalid_dimensions():
    x1 = torch.zeros(2, 3, 7)
    mask = position_valid_token_mask(image_seq_len=2, codebook_size=4, num_labels=3)
    noise = sample_masked_noise(x1, mask)
    assert torch.all(noise[:, :2, 4:] == 0)
    assert torch.all(noise[:, 2:, :4] == 0)
