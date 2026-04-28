import torch

from uniindex.layout import TaskLayout
from uniindex.schedule import apply_schedule
from uniindex.state import (
    build_flm_clean_state,
    condition_clean_timesteps,
    mix_flm_simplex,
    mix_flm_state,
    restore_image_tokens,
    sample_masked_noise,
)


def test_build_flm_clean_state_returns_one_hot():
    targets = torch.tensor([[1, 3], [0, 2]])
    state = build_flm_clean_state(targets, vocab_size=5)
    assert state.shape == (2, 2, 5)
    assert torch.equal(state.argmax(dim=-1), targets)


def test_apply_schedule_uses_different_powers_per_modality():
    progress = torch.tensor([0.25, 0.81])
    modality_ids = torch.tensor([0, 0, 1])
    t_pos = apply_schedule(
        progress,
        modality_ids,
        {"kind": "power"},
        image_time_power=1.0,
        text_time_power=0.5,
    )
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


def test_position_valid_token_mask_separates_image_and_text_subspaces():
    layout = TaskLayout(image_seq_len=2, text_seq_len=3, codebook_size=4, text_vocab_size=3)
    mask = layout.position_valid_token_mask()
    assert mask.shape == (5, 7)
    assert torch.equal(mask[0], torch.tensor([True, True, True, True, False, False, False]))
    assert torch.equal(mask[1], torch.tensor([True, True, True, True, False, False, False]))
    assert torch.equal(mask[2], torch.tensor([False, False, False, False, True, True, True]))


def test_sample_masked_noise_zeroes_invalid_dimensions():
    x1 = torch.zeros(2, 5, 7)
    layout = TaskLayout(image_seq_len=2, text_seq_len=3, codebook_size=4, text_vocab_size=3)
    mask = layout.position_valid_token_mask()
    noise = sample_masked_noise(x1, mask)
    assert torch.all(noise[:, :2, 4:] == 0)
    assert torch.all(noise[:, 2:, :4] == 0)


def test_sample_unmasked_noise_keeps_full_vocab_support():
    torch.manual_seed(0)
    x1 = torch.zeros(2, 5, 7)
    noise = sample_masked_noise(x1, None)
    assert torch.any(noise[:, :2, 4:] != 0)
    assert torch.any(noise[:, 2:, :4] != 0)


def test_mix_flm_simplex_returns_positionwise_probabilities():
    targets = torch.tensor([[1, 3, 0]])
    x1 = build_flm_clean_state(targets, vocab_size=5)
    t_pos = torch.tensor([[0.0, 0.5, 1.0]])
    z_t = mix_flm_simplex(x1, t_pos)
    assert torch.all(z_t >= 0)
    assert torch.allclose(z_t.sum(dim=-1), torch.ones(1, 3))
    assert torch.allclose(z_t[:, 0], torch.full((1, 5), 0.2))
    assert torch.equal(z_t[:, 2].argmax(dim=-1), torch.tensor([0]))


def test_full_vocab_simplex_keeps_cross_modal_support():
    layout = TaskLayout(image_seq_len=2, text_seq_len=2, codebook_size=4, text_vocab_size=3)
    image_tokens = torch.tensor([[0, 1]])
    text_tokens = torch.tensor([[4, 5]])
    x1 = build_flm_clean_state(torch.cat([image_tokens, text_tokens], dim=1), layout.vocab_size)
    z_t = mix_flm_state(x1, torch.zeros(1, layout.seq_len), path="simplex", valid_token_mask=None)
    assert torch.allclose(z_t.sum(dim=-1), torch.ones(1, layout.seq_len))
    assert torch.all(z_t[:, layout.image_slice, layout.codebook_size :] > 0)
    assert torch.all(z_t[:, layout.text_slice, : layout.codebook_size] > 0)


def test_modality_masked_simplex_can_still_restrict_support():
    layout = TaskLayout(image_seq_len=2, text_seq_len=2, codebook_size=4, text_vocab_size=3)
    x1 = torch.zeros(1, layout.seq_len, layout.vocab_size)
    z_t = mix_flm_state(
        x1,
        torch.zeros(1, layout.seq_len),
        path="simplex",
        valid_token_mask=layout.position_valid_token_mask(),
    )
    assert torch.allclose(z_t.sum(dim=-1), torch.ones(1, layout.seq_len))
    assert torch.all(z_t[:, layout.image_slice, layout.codebook_size :] == 0)
    assert torch.all(z_t[:, layout.text_slice, : layout.codebook_size] == 0)


def test_condition_clean_timesteps_sets_conditioned_positions_to_one():
    t_pos = torch.tensor([[0.2, 0.3, 0.7, 0.8], [0.4, 0.5, 0.8, 0.9]])
    adjusted = condition_clean_timesteps(
        t_pos,
        image_seq_len=2,
        condition_image=True,
        condition_text=False,
    )
    assert torch.equal(adjusted[:, :2], torch.ones(2, 2))
    assert torch.equal(adjusted[:, 2:], t_pos[:, 2:])

    adjusted = condition_clean_timesteps(
        t_pos,
        image_seq_len=2,
        condition_image=False,
        condition_text=True,
    )
    assert torch.equal(adjusted[:, :2], t_pos[:, :2])
    assert torch.equal(adjusted[:, 2:], torch.ones(2, 2))
