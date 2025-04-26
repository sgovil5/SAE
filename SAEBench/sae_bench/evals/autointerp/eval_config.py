from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class AutoInterpEvalConfig:
    """
    Controls all parameters for how autointerp will work.

    Arguments:
        model_name:                     The name of the model to use
        device:                         The device to use
        n_latents:                      The number of latents to use
        override_latents:               The latents to use (overrides n_latents if supplied)
        dead_latent_threshold:          The log sparsity value below which we consider a latent to be dead
        seed:                           The seed to use for all randomness

        buffer:                         The size of the buffer to use for scoring
        no_overlap:                     Whether to allow overlapping sequences for scoring
        act_threshold_frac:             The fraction of the maximum activation to use as the activation threshold
        total_tokens:                   The total number of tokens we'll gather data for.
        batch_size:                     The batch size to use for the scoring phase
        scoring:                        Whether to perform the scoring phase, or just return explanation
        max_tokens_in_explanation:      The maximum number of tokens to allow in an explanation
        use_demos_in_explanation:       Whether to use demonstrations in the explanation prompt

        n_top_ex_for_generation:        The number of top activating sequences to use for the generation phase
        n_iw_sampled_ex_for_generation: The number of importance-sampled sequences to use for the generation phase (this
                                        is a replacement for quantile sampling)

        n_top_ex_for_scoring:           The number of top sequences to use for scoring
        n_random_ex_for_scoring:        The number of random sequences to use for scoring
        n_iw_sampled_ex_for_scoring:    The number of importance-sampled sequences to use for scoring
    """

    # High-level params (not specific to autointerp)
    model_name: str = Field(
        default="",
        title="Model Name",
        description="Model name. Must be set with a command line argument.",
    )
    n_latents: int | None = Field(
        default=1000,
        title="Number of Latents",
        description="The number of latents for the LLM judge to interpret",
    )
    override_latents: list[int] | None = Field(
        default=None,
        title="Override Latents",
        description="The latents to use (overrides n_latents if supplied)",
    )
    dead_latent_threshold: float = Field(
        default=15,
        title="Dead Latent Threshold",
        description="Minimum number of required activations",
    )
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="The seed to use for all randomness",
    )
    dataset_name: str = Field(
        default="monology/pile-uncopyrighted",
        title="Dataset Name",
        description="The name of the dataset to use",
    )
    llm_context_size: int = Field(
        default=128,
        title="LLM Context Size",
        description="The context size to use for the LLM",
    )
    llm_batch_size: int = Field(
        default=None,
        title="LLM Batch Size",
        description="LLM batch size. This is set by default in the main script, or it can be set with a command line argument.",
    )  # type: ignore
    llm_dtype: str = Field(
        default="",
        title="LLM Data Type",
        description="LLM data type. This is set by default in the main script, or it can be set with a command line argument.",
    )

    # Main autointerp params
    buffer: int = Field(
        default=10,
        title="Buffer Size",
        description="The size of the buffer to use for scoring",
    )
    no_overlap: bool = Field(
        default=True,
        title="No Overlap",
        description="Whether to allow overlapping sequences for scoring",
    )
    act_threshold_frac: float = Field(
        default=0.01,
        title="Activation Threshold Fraction",
        description="The fraction of the maximum activation to use as the activation threshold",
    )
    total_tokens: int = Field(
        default=2_000_000,
        title="Total Tokens",
        description="The total number of tokens we'll gather data for",
    )
    scoring: bool = Field(
        default=True,
        title="Scoring",
        description="Whether to perform the scoring phase, or just return explanation",
    )
    max_tokens_in_explanation: int = Field(
        default=30,
        title="Max Tokens in Explanation",
        description="The maximum number of tokens to allow in an explanation",
    )
    use_demos_in_explanation: bool = Field(
        default=True,
        title="Use Demos in Explanation",
        description="Whether to use demonstrations in the explanation prompt",
    )

    # Sequences included in scoring phase
    n_top_ex_for_generation: int = Field(
        default=10,
        title="Number of Top Examples for Generation",
        description="The number of top activating sequences to use for the generation phase",
    )
    n_iw_sampled_ex_for_generation: int = Field(
        default=5,
        title="Number of IW Sampled Examples for Generation",
        description="The number of importance-sampled sequences to use for the generation phase",
    )
    n_top_ex_for_scoring: int = Field(
        default=2,
        title="Number of Top Examples for Scoring",
        description="The number of top sequences to use for scoring",
    )
    n_random_ex_for_scoring: int = Field(
        default=10,
        title="Number of Random Examples for Scoring",
        description="The number of random sequences to use for scoring",
    )
    n_iw_sampled_ex_for_scoring: int = Field(
        default=2,
        title="Number of IW Sampled Examples for Scoring",
        description="The number of importance-sampled sequences to use for scoring",
    )

    def __post_init__(self):
        if self.n_latents is None:
            assert self.override_latents is not None
            self.latents = self.override_latents
            self.n_latents = len(self.latents)
        else:
            assert self.override_latents is None
            self.latents = None

    @property
    def n_top_ex(self):
        """When fetching data, we get the top examples for generation & scoring simultaneously."""
        return self.n_top_ex_for_generation + self.n_top_ex_for_scoring

    @property
    def max_tokens_in_prediction(self) -> int:
        """Predictions take the form of comma-separated numbers, which should all be single tokens."""
        return 2 * self.n_ex_for_scoring + 5

    @property
    def n_ex_for_generation(self) -> int:
        return self.n_top_ex_for_generation + self.n_iw_sampled_ex_for_generation

    @property
    def n_ex_for_scoring(self) -> int:
        """For scoring phase, we use a randomly shuffled mix of top-k activations and random sequences."""
        return (
            self.n_top_ex_for_scoring
            + self.n_random_ex_for_scoring
            + self.n_iw_sampled_ex_for_scoring
        )

    @property
    def n_iw_sampled_ex(self) -> int:
        return self.n_iw_sampled_ex_for_generation + self.n_iw_sampled_ex_for_scoring

    @property
    def n_correct_for_scoring(self) -> int:
        return self.n_top_ex_for_scoring + self.n_iw_sampled_ex_for_scoring
