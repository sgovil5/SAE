from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from sae_bench.evals.core.eval_config import CoreEvalConfig

EVAL_TYPE_ID_CORE = "core"


# Define metrics for model behavior preservation
@dataclass
class ModelBehaviorPreservationMetrics(BaseMetrics):
    kl_div_score: float = Field(
        title="KL Divergence Score",
        description="Normalized KL divergence score comparing model behavior with and without SAE",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    kl_div_with_ablation: float = Field(
        title="KL Divergence with Ablation",
        description="KL divergence when the activation is ablated",
    )
    kl_div_with_sae: float = Field(
        title="KL Divergence with SAE",
        description="KL divergence when using the SAE reconstruction",
    )


# Define metrics for model performance preservation
@dataclass
class ModelPerformancePreservationMetrics(BaseMetrics):
    ce_loss_score: float = Field(
        title="Cross Entropy Loss Score",
        description="Normalized cross entropy loss score comparing model performance with and without SAE",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    ce_loss_with_ablation: float = Field(
        title="CE Loss with Ablation",
        description="Cross entropy loss when the activation is ablated",
    )
    ce_loss_with_sae: float = Field(
        title="CE Loss with SAE",
        description="Cross entropy loss when using the SAE reconstruction",
    )
    ce_loss_without_sae: float = Field(
        title="CE Loss without SAE",
        description="Base cross entropy loss without any intervention",
    )


# Define metrics for reconstruction quality
@dataclass
class ReconstructionQualityMetrics(BaseMetrics):
    explained_variance: float = Field(
        title="Explained Variance",
        description="Proportion of variance in the original activation explained by the SAE reconstruction",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    explained_variance_legacy: float = Field(
        title="Explained Variance (Legacy)",
        description="Previously used, incorrect, formula for explained variance",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    mse: float = Field(
        title="Mean Squared Error",
        description="Mean squared error between original activation and SAE reconstruction",
    )
    cossim: float = Field(
        title="Cosine Similarity",
        description="Cosine similarity between original activation and SAE reconstruction",
    )


# Define metrics for shrinkage
@dataclass
class ShrinkageMetrics(BaseMetrics):
    l2_norm_in: float = Field(
        title="Input L2 Norm",
        description="Average L2 norm of input activations",
    )
    l2_norm_out: float = Field(
        title="Output L2 Norm",
        description="Average L2 norm of reconstructed activations",
    )
    l2_ratio: float = Field(
        title="L2 Ratio",
        description="Ratio of output to input L2 norms",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    relative_reconstruction_bias: float = Field(
        title="Relative Reconstruction Bias",
        description="Measure of systematic bias in the reconstruction",
    )


# Define metrics for sparsity
@dataclass
class SparsityMetrics(BaseMetrics):
    l0: float = Field(
        title="L0 Sparsity",
        description="Average number of non-zero feature activations",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    l1: float = Field(
        title="L1 Sparsity",
        description="Average sum of absolute feature activations",
    )


@dataclass
class MiscMetrics(BaseMetrics):
    freq_over_1_percent: float = Field(
        title="Activation Frequency Over 1%",
        description="Proportion of tokens that activate each feature more than 1% of the time",
    )
    freq_over_10_percent: float = Field(
        title="Activation Frequency Over 10%",
        description="Proportion of tokens that activate each feature more than 10% of the time",
    )
    normalized_freq_over_1_percent: float = Field(
        title="Normalized Activation Frequency Over 1%",
        description="Sum of > 1% activation frequency probabilities, normalized by the sum of all feature probabilities",
    )
    normalized_freq_over_10_percent: float = Field(
        title="Normalized Activation Frequency Over 10%",
        description="Sum of > 10% activation frequency probabilities, normalized by the sum of all feature probabilities",
    )
    average_max_encoder_cosine_sim: float = Field(
        title="Average Max Encoder Cosine Similarity",
        description="Average of the maximum cosine similarity with any other feature's encoder weights",
    )
    average_max_decoder_cosine_sim: float = Field(
        title="Average Max Decoder Cosine Similarity",
        description="Average of the maximum cosine similarity with any other feature's decoder weights",
    )
    frac_alive: float = Field(
        title="Fraction of Alive Features",
        description="Fraction of features that fired at least once during evaluation. This will likely be an underestimation due to a limited amount of tokens",
    )


# Define metrics for token stats
@dataclass
class TokenStatsMetrics(BaseMetrics):
    total_tokens_eval_reconstruction: int = Field(
        title="Total Tokens (Reconstruction)",
        description="Total number of tokens used in reconstruction evaluation",
    )
    total_tokens_eval_sparsity_variance: int = Field(
        title="Total Tokens (Sparsity/Variance)",
        description="Total number of tokens used in sparsity and variance evaluation",
    )


# Define the categories themselves
@dataclass
class CoreMetricCategories(BaseMetricCategories):
    model_behavior_preservation: ModelBehaviorPreservationMetrics = Field(
        title="Model Behavior Preservation",
        description="Metrics related to how well the SAE preserves model behavior",
    )
    model_performance_preservation: ModelPerformancePreservationMetrics = Field(
        title="Model Performance Preservation",
        description="Metrics related to how well the SAE preserves model performance",
    )
    reconstruction_quality: ReconstructionQualityMetrics = Field(
        title="Reconstruction Quality",
        description="Metrics related to how well the SAE reconstructs the original activation",
    )
    shrinkage: ShrinkageMetrics = Field(
        title="Shrinkage",
        description="Metrics related to how the SAE changes activation magnitudes",
    )
    sparsity: SparsityMetrics = Field(
        title="Sparsity",
        description="Metrics related to feature activation sparsity",
    )
    token_stats: TokenStatsMetrics = Field(
        title="Token Statistics",
        description="Statistics about the number of tokens used in evaluation",
    )
    misc_metrics: MiscMetrics = Field(
        title="Miscellaneous Metrics",
        description="Miscellaneous metrics",
    )


# Define the feature-wise metrics
@dataclass
class CoreFeatureMetric(BaseResultDetail):
    index: int = Field(
        title="Feature Index",
        description="Index of the feature in the SAE",
    )
    feature_density: float = Field(
        title="Feature Density",
        description="Proportion of tokens that activate each feature",
    )
    consistent_activation_heuristic: float = Field(
        title="Consistent Activation Heuristic",
        description="Average number of tokens per prompt that activate each feature",
    )
    encoder_bias: float = Field(
        title="Encoder Bias",
        description="Bias terms in the encoder for each feature",
    )
    encoder_norm: float = Field(
        title="Encoder Norm",
        description="L2 norm of encoder weights for each feature",
    )
    encoder_decoder_cosine_sim: float = Field(
        title="Encoder-Decoder Cosine Similarity",
        description="Cosine similarity between encoder and decoder weights for each feature",
    )
    max_decoder_cosine_sim: float = Field(
        title="Max Decoder Cosine Similarity",
        description="Maximum cosine similarity with any other feature's decoder weights",
    )
    max_encoder_cosine_sim: float = Field(
        title="Max Encoder Cosine Similarity",
        description="Maximum cosine similarity with any other feature's encoder weights",
    )


# Define the eval output
@dataclass(config=ConfigDict(title="Core"))
class CoreEvalOutput(
    BaseEvalOutput[CoreEvalConfig, CoreMetricCategories, CoreFeatureMetric]
):
    """
    Core SAE evaluations measuring reconstruction quality, sparsity, and model preservation. From SAELens.
    """

    eval_config: CoreEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: CoreMetricCategories
    eval_result_details: list[CoreFeatureMetric] = Field(
        default_factory=list,
        title="Feature-wise Metrics",
        description="Detailed metrics for each feature in the SAE",
    )
    eval_type_id: str = Field(
        default=EVAL_TYPE_ID_CORE,
        title="Eval Type ID",
        description="The type of the evaluation",
    )
