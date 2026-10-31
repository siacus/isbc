from argparse import Namespace

from binary_factors_train import adaptive_sparsity_is_enabled


def test_zero_activity_weight_disables_adaptive_controller() -> None:
    assert not adaptive_sparsity_is_enabled(Namespace(lambda_tweet_sparse=0.0))


def test_positive_activity_weight_enables_adaptive_controller() -> None:
    assert adaptive_sparsity_is_enabled(Namespace(lambda_tweet_sparse=1.0))
