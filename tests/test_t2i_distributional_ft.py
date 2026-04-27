import torch

from uniindex.layout import TaskLayout
from uniindex.state import build_flm_clean_state
from uniindex.t2i_distributional_ft import (
    _record_distributional_eval_steps,
    build_image_simplex_state,
    build_label_candidate_bank_from_batches,
    set_ce_k_loss,
)


def test_image_simplex_state_only_uses_image_codebook_and_preserves_text():
    layout = TaskLayout(image_seq_len=2, text_seq_len=1, codebook_size=4, text_vocab_size=3)
    image_tokens = torch.tensor([[1, 3]])
    text_tokens = torch.tensor([[2]])
    targets = torch.cat([image_tokens, text_tokens + layout.text_offset], dim=1)
    x1 = build_flm_clean_state(targets, layout.vocab_size)
    t_pos = torch.tensor([[0.25, 0.75, 1.0]])

    z_t = build_image_simplex_state(x1=x1, image_tokens=image_tokens, layout=layout, t_pos=t_pos)

    image_state = z_t[:, layout.image_slice]
    assert torch.allclose(image_state.sum(dim=-1), torch.ones(1, 2))
    assert torch.count_nonzero(image_state[:, :, layout.codebook_size :]).item() == 0
    assert torch.allclose(z_t[:, layout.text_slice], x1[:, layout.text_slice])
    assert image_state[0, 0, 1].item() > image_state[0, 0, 0].item()
    assert image_state[0, 1, 3].item() > image_state[0, 1, 0].item()


def test_set_ce_k_loss_rewards_matching_any_same_label_candidate():
    layout = TaskLayout(image_seq_len=2, text_seq_len=1, codebook_size=4, text_vocab_size=3)
    label_values = torch.tensor([0, 1])
    labels = torch.tensor([1])
    candidate_bank = torch.tensor(
        [
            [[0, 0], [0, 1]],
            [[1, 2], [2, 3]],
        ]
    )
    matching_logits = torch.zeros(1, layout.seq_len, layout.vocab_size)
    matching_logits[0, 0, 1] = 8.0
    matching_logits[0, 1, 2] = 8.0
    random_logits = torch.zeros_like(matching_logits)

    matching = set_ce_k_loss(
        logits=matching_logits,
        labels=labels,
        label_values=label_values,
        candidate_bank=candidate_bank,
        layout=layout,
        temperature=0.25,
    )
    random = set_ce_k_loss(
        logits=random_logits,
        labels=labels,
        label_values=label_values,
        candidate_bank=candidate_bank,
        layout=layout,
        temperature=0.25,
    )

    assert matching.item() < random.item()


def test_candidate_bank_keeps_labels_separate_and_repeats_small_classes():
    label_values = torch.tensor([10, 20])
    batches = [
        {
            "image_tokens": torch.tensor([[1, 1], [2, 2], [3, 3]]),
            "label": torch.tensor([10, 20, 10]),
        }
    ]

    bank = build_label_candidate_bank_from_batches(batches=batches, label_values=label_values, set_size=3)

    assert bank.tolist() == [
        [[1, 1], [3, 3], [1, 1]],
        [[2, 2], [2, 2], [2, 2]],
    ]


def test_distributional_eval_steps_default_and_custom():
    assert _record_distributional_eval_steps(500) == [0, 100, 300, 500]
    assert _record_distributional_eval_steps(2) == [0, 2]
    assert _record_distributional_eval_steps(500, (150, 450)) == [0, 150, 450, 500]
