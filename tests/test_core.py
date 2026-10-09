import torch

from didi_merging import build_task_vectors, fit_didi


def test_fit_and_reconstruct():
    torch.manual_seed(0)
    base = {
        "attn.weight": torch.zeros(8, 8),
        "mlp.weight": torch.zeros(10, 8),
    }
    fine_tuned = [
        {name: value + 0.1 * torch.randn_like(value) for name, value in base.items()}
        for _ in range(3)
    ]
    task_vectors = build_task_vectors(base, fine_tuned)
    model = fit_didi(
        task_vectors,
        budget_ratio=0.60,
        allocation_steps=20,
        refinement_steps=5,
    )

    assert model.storage_ratio() <= 0.60
    for task_id in range(3):
        reconstructed = model.reconstruct_task(task_id)
        assert reconstructed.keys() == base.keys()
        for name in base:
            assert reconstructed[name].shape == base[name].shape
            assert torch.isfinite(reconstructed[name]).all()
