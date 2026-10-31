import unittest
from types import SimpleNamespace

import torch

from binary_factors_train import initialize_decoder_intercept_from_embedding_mean


class DecoderInterceptInitializationTest(unittest.TestCase):
    def test_expected_initial_reconstruction_matches_embedding_mean(self) -> None:
        linear = torch.nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            linear.weight.copy_(
                torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0]])
            )
            linear.bias.zero_()
        model = SimpleNamespace(
            decoder=SimpleNamespace(linear=linear),
            k_factors=3,
            target_rho=0.4,
        )
        embedding_mean = torch.tensor([7.0, -3.0])

        initialize_decoder_intercept_from_embedding_mean(model, embedding_mean)

        expected_code = torch.full((3,), 0.4)
        reconstructed_mean = linear(expected_code)
        torch.testing.assert_close(reconstructed_mean, embedding_mean)

    def test_rejects_incompatible_embedding_dimension(self) -> None:
        model = SimpleNamespace(
            decoder=SimpleNamespace(linear=torch.nn.Linear(3, 2)),
            k_factors=3,
            target_rho=0.4,
        )
        with self.assertRaisesRegex(ValueError, "embedding mean shape"):
            initialize_decoder_intercept_from_embedding_mean(
                model, torch.zeros(4)
            )


if __name__ == "__main__":
    unittest.main()
