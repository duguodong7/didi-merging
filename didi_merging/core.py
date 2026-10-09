from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _check_matrix_dicts(states: Sequence[Mapping[str, Tensor]]) -> tuple[str, ...]:
    if not states:
        raise ValueError("at least one task vector is required")
    keys = tuple(states[0].keys())
    if not keys:
        raise ValueError("task vectors must not be empty")
    for state in states:
        if tuple(state.keys()) != keys:
            raise ValueError("all task vectors must have identical ordered keys")
        for name, tensor in state.items():
            if tensor.ndim != 2:
                raise ValueError(f"{name} must be a matrix; got shape {tuple(tensor.shape)}")
    return keys


def build_task_vectors(
    base: Mapping[str, Tensor],
    fine_tuned: Sequence[Mapping[str, Tensor]],
) -> list[dict[str, Tensor]]:
    """Compute fine-tuned minus base parameters for selected matrices."""

    if not fine_tuned:
        raise ValueError("fine_tuned must contain at least one model")
    result: list[dict[str, Tensor]] = []
    for state in fine_tuned:
        if state.keys() != base.keys():
            raise ValueError("base and fine-tuned states must have identical keys")
        result.append({name: state[name] - base[name] for name in base})
    _check_matrix_dicts(result)
    return result


@dataclass(frozen=True)
class FactorizedMatrix:
    left: Tensor
    right: Tensor

    @property
    def rank(self) -> int:
        return self.left.shape[1]

    @property
    def storage(self) -> int:
        return self.left.numel() + self.right.numel()

    def reconstruct(self) -> Tensor:
        return self.left @ self.right


@dataclass(frozen=True)
class DiDiModel:
    shared: dict[str, FactorizedMatrix]
    experts: tuple[dict[str, FactorizedMatrix], ...]
    original_shapes: dict[str, tuple[int, int]]

    def reconstruct_task(self, task_id: int) -> dict[str, Tensor]:
        if not 0 <= task_id < len(self.experts):
            raise IndexError("task_id is out of range")
        return {
            name: self.shared[name].reconstruct()
            + self.experts[task_id][name].reconstruct()
            for name in self.shared
        }

    def storage(self) -> int:
        return sum(component.storage for component in self.shared.values()) + sum(
            component.storage
            for task in self.experts
            for component in task.values()
        )

    def dense_component_storage(self) -> int:
        copies = 1 + len(self.experts)
        return copies * sum(rows * cols for rows, cols in self.original_shapes.values())

    def storage_ratio(self) -> float:
        return self.storage() / self.dense_component_storage()

    def rank_summary(self) -> dict[str, object]:
        return {
            "shared": {name: value.rank for name, value in self.shared.items()},
            "experts": [
                {name: value.rank for name, value in task.items()}
                for task in self.experts
            ],
            "storage_ratio": self.storage_ratio(),
        }


def _svd(matrices: Mapping[str, Tensor]) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    return {
        name: torch.linalg.svd(matrix.float(), full_matrices=False)
        for name, matrix in matrices.items()
    }


def allocate_ranks(
    matrices: Mapping[str, Tensor],
    budget_ratio: float,
    *,
    steps: int = 300,
    learning_rate: float = 0.05,
    temperature: float = 0.10,
    budget_weight: float = 25.0,
) -> dict[str, int]:
    """Jointly allocate low-rank capacity with differentiable soft cutoffs.

    A learnable cutoff controls a monotone gate over each matrix's singular
    values. The objective balances reconstruction error and one global storage
    budget. Soft gates are converted to prefix ranks without exceeding the
    requested budget.
    """

    if not 0 < budget_ratio <= 1:
        raise ValueError("budget_ratio must be in (0, 1]")
    if not matrices:
        raise ValueError("matrices must not be empty")

    decompositions = _svd(matrices)
    names = tuple(matrices)
    cutoffs = nn.ParameterDict()
    for index, name in enumerate(names):
        rank = decompositions[name][1].numel()
        cutoffs[str(index)] = nn.Parameter(
            torch.tensor(float(max(0, rank * budget_ratio - 0.5)))
        )

    optimizer = torch.optim.Adam(cutoffs.parameters(), lr=learning_rate)
    dense_storage = sum(matrix.numel() for matrix in matrices.values())
    energy = sum(s.square().sum() for _, s, _ in decompositions.values()).clamp_min(1e-12)

    for _ in range(steps):
        optimizer.zero_grad()
        reconstruction = torch.zeros((), dtype=torch.float32)
        soft_storage = torch.zeros((), dtype=torch.float32)
        for index, name in enumerate(names):
            matrix = matrices[name]
            singular_values = decompositions[name][1]
            positions = torch.arange(
                singular_values.numel(), device=singular_values.device, dtype=torch.float32
            )
            gate = torch.sigmoid((cutoffs[str(index)] - positions) / temperature)
            reconstruction = reconstruction + ((1.0 - gate) * singular_values).square().sum()
            soft_storage = soft_storage + gate.sum() * (matrix.shape[0] + matrix.shape[1])
        storage_ratio = soft_storage / dense_storage
        loss = reconstruction / energy + budget_weight * (storage_ratio - budget_ratio).square()
        loss.backward()
        optimizer.step()

    candidates: list[tuple[float, int, str]] = []
    for index, name in enumerate(names):
        matrix = matrices[name]
        singular_values = decompositions[name][1]
        positions = torch.arange(
            singular_values.numel(), device=singular_values.device, dtype=torch.float32
        )
        gate = torch.sigmoid((cutoffs[str(index)].detach() - positions) / temperature)
        cost = matrix.shape[0] + matrix.shape[1]
        for component, (singular, soft_gate) in enumerate(zip(singular_values, gate)):
            score = float(singular.square() * soft_gate / cost)
            candidates.append((score, component, name))

    budget = int(dense_storage * budget_ratio)
    ranks = {name: 0 for name in names}
    used = 0
    for _, component, name in sorted(candidates, reverse=True):
        cost = matrices[name].shape[0] + matrices[name].shape[1]
        if component != ranks[name]:
            continue
        if used + cost > budget:
            continue
        ranks[name] += 1
        used += cost
    return ranks


def _factorize(matrix: Tensor, rank: int) -> FactorizedMatrix:
    if rank == 0:
        return FactorizedMatrix(
            matrix.new_zeros(matrix.shape[0], 0),
            matrix.new_zeros(0, matrix.shape[1]),
        )
    u, s, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    rank = min(rank, s.numel())
    root = s[:rank].sqrt()
    left = u[:, :rank] * root
    right = root[:, None] * vh[:rank]
    return FactorizedMatrix(left.to(matrix.dtype), right.to(matrix.dtype))


def _refine(
    targets: Mapping[str, Tensor],
    factors: Mapping[str, FactorizedMatrix],
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, FactorizedMatrix]:
    if steps <= 0:
        return dict(factors)

    left = nn.ParameterDict()
    right = nn.ParameterDict()
    name_to_id = {name: str(index) for index, name in enumerate(targets)}
    for name, identifier in name_to_id.items():
        left[identifier] = nn.Parameter(factors[name].left.float().clone())
        right[identifier] = nn.Parameter(factors[name].right.float().clone())

    parameters = list(left.parameters()) + list(right.parameters())
    if not parameters:
        return dict(factors)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = torch.zeros((), dtype=torch.float32)
        for name, target in targets.items():
            identifier = name_to_id[name]
            loss = loss + F.mse_loss(left[identifier] @ right[identifier], target.float())
        loss.backward()
        optimizer.step()

    return {
        name: FactorizedMatrix(
            left[name_to_id[name]].detach().to(target.dtype),
            right[name_to_id[name]].detach().to(target.dtype),
        )
        for name, target in targets.items()
    }


def fit_didi(
    task_vectors: Sequence[Mapping[str, Tensor]],
    budget_ratio: float,
    *,
    allocation_steps: int = 300,
    refinement_steps: int = 100,
    learning_rate: float = 0.03,
) -> DiDiModel:
    """Fit a compact dynamic shared--expert representation."""

    keys = _check_matrix_dicts(task_vectors)
    shared = {
        name: torch.stack([task[name] for task in task_vectors]).mean(dim=0)
        for name in keys
    }
    experts = [
        {name: task[name] - shared[name] for name in keys}
        for task in task_vectors
    ]

    components: dict[str, Tensor] = {}
    for name, matrix in shared.items():
        components[f"shared::{name}"] = matrix
    for task_id, expert in enumerate(experts):
        for name, matrix in expert.items():
            components[f"expert::{task_id}::{name}"] = matrix

    ranks = allocate_ranks(
        components,
        budget_ratio,
        steps=allocation_steps,
        learning_rate=learning_rate,
    )
    factors = {
        name: _factorize(matrix, ranks[name])
        for name, matrix in components.items()
    }
    factors = _refine(
        components,
        factors,
        steps=refinement_steps,
        learning_rate=learning_rate,
    )

    shared_factors = {
        name: factors[f"shared::{name}"]
        for name in keys
    }
    expert_factors = tuple(
        {
            name: factors[f"expert::{task_id}::{name}"]
            for name in keys
        }
        for task_id in range(len(task_vectors))
    )
    return DiDiModel(
        shared=shared_factors,
        experts=expert_factors,
        original_shapes={name: tuple(task_vectors[0][name].shape) for name in keys},
    )
