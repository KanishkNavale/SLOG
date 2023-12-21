from typing import List, Tuple
import torch

from slog.distances import cosine_similarity


@torch.jit.script
def _similarity(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    cs = cosine_similarity(a, b)
    return torch.exp(cs / temperature)


@torch.jit.script
def _pick_descriptors(batch_image_a: torch.Tensor,
                      batch_image_b: torch.Tensor,
                      batch_matches_a: torch.Tensor,
                      batch_matches_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    descriptors_a: List[torch.Tensor] = []
    descriptors_b: List[torch.Tensor] = []
    for image_a, image_b, matches_a, matches_b in zip(batch_image_a, batch_image_b, batch_matches_a, batch_matches_b):
        descriptors_a.append(image_a[:, matches_a[:, 0].long(), matches_a[:, 1].long()])
        descriptors_b.append(image_b[:, matches_b[:, 0].long(), matches_b[:, 1].long()])

    batch_descriptors_a = torch.stack(descriptors_a).permute(0, 2, 1)
    batch_descriptors_b = torch.stack(descriptors_b).permute(0, 2, 1)

    return batch_descriptors_a, batch_descriptors_b


@torch.jit.script
def _create_mask(batch_neg_pairs_in_a: torch.Tensor, batch_matches_a: torch.Tensor) -> torch.Tensor:
    mask: List[torch.Tensor] = []
    for matches_a in batch_matches_a:
        paddings = torch.ones_like(batch_neg_pairs_in_a[0])
        paddings[:, matches_a[:, 0].long(), matches_a[:, 1].long()] = torch.zeros_like(paddings[:, matches_a[:, 0].long(), matches_a[:, 1].long()])
        mask.append(paddings)
    mask = torch.stack(mask)
    return mask


class PixelwiseNTXentLoss:
    def __init__(self, temperature: float, reduction: str = 'mean') -> None:
        self.name = 'Pixelwise NXTent Loss'
        self.reduction = reduction
        self.temperature = temperature

    def _compute_batch_loss(self,
                            batch_image_a: torch.Tensor,
                            batch_image_b: torch.Tensor,
                            batch_matches_a: torch.Tensor,
                            batch_matches_b: torch.Tensor,
                            temperature: torch.Tensor) -> torch.Tensor:

        batch_descriptors_a, batch_descriptors_b = _pick_descriptors(batch_image_a, batch_image_b, batch_matches_a, batch_matches_b)

        batch_positive_pairs = _similarity(batch_descriptors_a, batch_descriptors_b, temperature)

        tiled_batch_descriptor_a = batch_descriptors_a.view(batch_descriptors_a.shape[0],
                                                            batch_descriptors_a.shape[1],
                                                            1,
                                                            1,
                                                            batch_descriptors_a.shape[2]).tile(1,
                                                                                               1,
                                                                                               batch_image_a.shape[-2],
                                                                                               batch_image_a.shape[-1],
                                                                                               1)

        batch_neg_pairs_in_b = _similarity(tiled_batch_descriptor_a, batch_image_b.permute(0, 2, 3, 1).unsqueeze(1), temperature)
        batch_neg_pairs_in_a = _similarity(tiled_batch_descriptor_a, batch_image_a.permute(0, 2, 3, 1).unsqueeze(1), temperature)

        mask = _create_mask(batch_neg_pairs_in_a, batch_matches_a)
        masked_neg_pairs_in_a = batch_neg_pairs_in_a * mask

        denominator = torch.sum(masked_neg_pairs_in_a, dim=(-2, -1)) + torch.sum(batch_neg_pairs_in_b, dim=(-2, -1)) + 1e-12
        loss = torch.mean(-1.0 * torch.log(batch_positive_pairs / (batch_positive_pairs + denominator)), dim=-1)

        if self.reduction == "sum":
            return torch.sum(loss)

        elif self.reduction == "mean":
            return torch.mean(loss)

    def __call__(self,
                 batch_image_a: torch.Tensor,
                 batch_image_b: torch.Tensor,
                 batch_matches_a: torch.Tensor,
                 batch_matches_b: torch.Tensor) -> torch.Tensor:

        backward_loss = self._compute_batch_loss(batch_image_a,
                                                 batch_image_b,
                                                 batch_matches_a,
                                                 batch_matches_b,
                                                 self.temperature)

        forward_loss = self._compute_batch_loss(batch_image_b,
                                                batch_image_a,
                                                batch_matches_b,
                                                batch_matches_a,
                                                self.temperature)

        return 0.5 * (backward_loss + forward_loss)
