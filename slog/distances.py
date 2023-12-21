import torch


def l2(target: torch.Tensor, source: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.linalg.norm(source - target, ord=2, dim=dim)


def l1(target: torch.Tensor, source: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.linalg.norm(source - target, ord=1, dim=dim)


def guassian_distance_kernel(target: torch.Tensor,
                             source: torch.Tensor,
                             temperature: torch.Tensor,
                             dim: int = -1) -> torch.Tensor:
    distances = l2(target, source, dim)
    weights = torch.exp(-1.0 * distances / temperature)
    return weights


def exp_guassian_distance_kernel(target: torch.Tensor,
                                 source: torch.Tensor,
                                 temperature: torch.Tensor,
                                 dim: int = -1) -> torch.Tensor:
    distances = l2(target, source, dim)
    weights = torch.exp((-1.0 * distances) / torch.exp(temperature))
    return weights


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    product = torch.sum(a * b, dim=dim)
    normalizer = torch.linalg.norm(a, dim=dim) * torch.linalg.norm(b, dim=dim)
    return product / (normalizer + 1e-12)
