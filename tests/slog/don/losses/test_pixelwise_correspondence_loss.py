
import unittest
import torch

from slog.don import PixelwiseCorrespondenceLoss


class TestPixelwiseCorrespondenceLoss(unittest.TestCase):

    def build_descriptors_and_matches(self):
        descriptor_image_a = torch.zeros((1, 3, 200, 200))
        descriptor_image_b = torch.zeros((1, 3, 200, 200))

        descriptor_image_a[0, :, 100, 10] = torch.ones(3) * 1e3
        descriptor_image_b[0, :, 10, 100] = torch.ones(3) * 1e3

        matches_image_a = [torch.as_tensor([[100, 10]])]
        matches_image_b = [torch.as_tensor([[10, 100]])]

        return descriptor_image_a, descriptor_image_b, matches_image_a, matches_image_b

    def test_zero_loss(self):
        loss_function = PixelwiseCorrespondenceLoss(reduction="mean")
        loss = loss_function(*self.build_descriptors_and_matches())

        self.assertEqual(loss, torch.zeros_like(loss))

    def test_reductions(self):
        sum_loss_function = PixelwiseCorrespondenceLoss(reduction='sum')
        mean_loss_function = PixelwiseCorrespondenceLoss(reduction='mean')

        descriptor_image_a = torch.zeros((2, 3, 200, 200))
        descriptor_image_b = torch.zeros((2, 3, 200, 200))

        descriptor_image_a[0, :, 100, 10] = torch.ones(3) * 1e3
        descriptor_image_b[0, :, 10, 100] = torch.ones(3) * 1e3
        descriptor_image_a[1, :, 0, 1] = torch.ones(3) * 1e3
        descriptor_image_b[1, :, 10, 10] = torch.ones(3) * 1e3

        # Injecting errors for correspondence
        matches_image_a = [torch.as_tensor([[100, 10]]), torch.as_tensor([[0, 2]])]
        matches_image_b = [torch.as_tensor([[10, 100]]), torch.as_tensor([[10, 21]])]

        sum_loss = sum_loss_function(descriptor_image_a, descriptor_image_b, matches_image_a, matches_image_b)
        mean_loss = mean_loss_function(descriptor_image_a, descriptor_image_b, matches_image_a, matches_image_b)

        self.assertGreater(sum_loss, mean_loss)


if __name__ == '__main__':
    unittest.main()
