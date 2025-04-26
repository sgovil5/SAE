import pytest
import torch
from sae_lens import SAE
from torch.nn.functional import normalize
from transformer_lens import HookedTransformer

from sae_bench.evals.absorption.feature_absorption_calculator import (
    FeatureAbsorptionCalculator,
)


def test_FeatureAbsorptionCalculator_validate_prompts_are_same_length_errors_if_prompts_are_variable_lengths(
    gpt2_model: HookedTransformer,
):
    words = [" cat", " antelope", " fish"]
    calculator = FeatureAbsorptionCalculator(
        gpt2_model,
        icl_word_list=["dog"],
    )
    prompts = calculator._build_prompts(words)
    prompts[1].base += "EXTRA TEXT"

    with pytest.raises(ValueError):
        calculator._validate_prompts_are_same_length(prompts)


def test_FeatureAbsorptionCalculator_calculate_absorption_results_look_reasonable(
    gpt2_model: HookedTransformer, gpt2_l4_sae: SAE
):
    words = [" cat", " chair", " car"]
    calculator = FeatureAbsorptionCalculator(
        gpt2_model, icl_word_list=["dog"], topk_feats=10, batch_size=2
    )
    probe_dir = normalize(torch.randn(768), dim=-1)

    sampled_results = calculator.calculate_absorption(
        gpt2_l4_sae,
        words,
        probe_direction=probe_dir,
        layer=4,
        main_feature_ids=[1, 2, 3],
    )
    with torch.no_grad():
        assert sampled_results.main_feature_ids == [1, 2, 3]
        assert len(sampled_results.word_results) == 3
        for sample in sampled_results.word_results:
            assert sample.word in words
            assert len(sample.main_feature_scores) == 3
            assert len(sample.top_projection_feature_scores) == 10
            for feat_score in sample.main_feature_scores:
                assert feat_score.feature_id in [1, 2, 3]
                sae_dir = normalize(gpt2_l4_sae.W_dec[feat_score.feature_id], dim=-1)
                assert feat_score.probe_cos_sim == pytest.approx(
                    (probe_dir @ sae_dir).item(), abs=1e-5
                )
            for feat_score in sample.top_projection_feature_scores:
                sae_dir = normalize(gpt2_l4_sae.W_dec[feat_score.feature_id], dim=-1)
                assert feat_score.probe_cos_sim == pytest.approx(
                    (probe_dir @ sae_dir).item(), abs=1e-5
                )
