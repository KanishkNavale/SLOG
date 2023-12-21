
import unittest
import torch

from slog.don import PixelwiseNTXentLoss


class TestPixelwisePixelwiseNXTentLoss(unittest.TestCase):

    def test_reductions(self):
        sum_loss_function = PixelwiseNTXentLoss(temperature=0.1, reduction='sum')
        mean_loss_function = PixelwiseNTXentLoss(temperature=0.1, reduction='mean')

        descriptor_image_a = torch.zeros((2, 3, 200, 200))
        descriptor_image_b = torch.zeros((2, 3, 200, 200))

        descriptor_image_a[0, :, 100, 10] = torch.ones(3) * 1e3
        descriptor_image_b[0, :, 10, 100] = torch.ones(3) * 1e3
        descriptor_image_a[1, :, 0, 1] = torch.ones(3) * 1e3
        descriptor_image_b[1, :, 10, 10] = torch.ones(3) * 1e3

        # Injecting errors for correspondence
        matches_image_a = torch.stack([torch.as_tensor([[100, 10]]), torch.as_tensor([[0, 2]])])
        matches_image_b = torch.stack([torch.as_tensor([[10, 100]]), torch.as_tensor([[10, 21]])])

        sum_loss = sum_loss_function(descriptor_image_a,
                                     descriptor_image_b,
                                     matches_image_a,
                                     matches_image_b)

        mean_loss = mean_loss_function(descriptor_image_a,
                                       descriptor_image_b,
                                       matches_image_a,
                                       matches_image_b)

        self.assertGreater(sum_loss, mean_loss)


if __name__ == '__main__':
    unittest.main()
