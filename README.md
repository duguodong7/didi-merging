# DiDi-Merging

Minimal algorithm release for **Dynamic Model Merging Made Slim
(DiDi-Merging)**.

The package implements the three method stages:

1. shared--expert decomposition of task vectors;
2. differentiable rank allocation under one coupled parameter budget; and
3. data-free refinement of the retained low-rank factors against the stored
   task vectors.

This repository contains algorithm code only. It does not include datasets,
fine-tuning pipelines, benchmark configurations, checkpoints, or evaluation
scripts.

## Install

```bash
pip install -e .
```

Only PyTorch is required.

## Minimal example

```python
import torch
from didi_merging import build_task_vectors, fit_didi

base = {
    "attn.weight": torch.zeros(12, 12),
    "mlp.weight": torch.zeros(16, 12),
}
fine_tuned = [
    {name: value + 0.10 * torch.randn_like(value) for name, value in base.items()},
    {name: value + 0.10 * torch.randn_like(value) for name, value in base.items()},
]

task_vectors = build_task_vectors(base, fine_tuned)
merged = fit_didi(
    task_vectors,
    budget_ratio=0.40,
    allocation_steps=200,
    refinement_steps=50,
)

# Reconstruct the dynamic update selected for task 0.
delta_for_task_0 = merged.reconstruct_task(0)
print(merged.rank_summary())
```

`budget_ratio` applies to the stored shared and expert matrix components.
The base model is frozen and is therefore not duplicated by this package.

## Scope

This initial release exposes the tensor-level algorithm. Model-family adapters,
released checkpoints, and broader evaluation utilities can be added without
changing the core API.

## Citation

```bibtex
@inproceedings{du2026didi,
  title={Dynamic Model Merging Made Slim},
  author={Du, Guodong and Lin, Wanyu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

MIT
