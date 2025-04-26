from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from sae_bench.evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from sae_bench.evals.unlearning.eval_config import UnlearningEvalConfig

EVAL_TYPE_ID_UNLEARNING = "unlearning"


@dataclass
class UnlearningMetrics(BaseMetrics):
    unlearning_score: float = Field(
        title="Unlearning Score",
        description="Unlearning score, using methodology from APPLYING SPARSE AUTOENCODERS TO UNLEARN KNOWLEDGE IN LANGUAGE MODELS",
        json_schema_extra=DEFAULT_DISPLAY,
    )


# Define the categories themselves
@dataclass
class UnlearningMetricCategories(BaseMetricCategories):
    unlearning: UnlearningMetrics = Field(
        title="Unlearning",
        description="Metrics related to unlearning",
    )


# Define the eval output
@dataclass(config=ConfigDict(title="Unlearning"))
class UnlearningEvalOutput(
    BaseEvalOutput[UnlearningEvalConfig, UnlearningMetricCategories, BaseResultDetail]
):
    """
    An evaluation of the ability of SAEs to unlearn biology knowledge from LLMs, using methodology from `Applying Sparse Autoencoders to Unlearn Knowledge in Language Models`
    """

    eval_config: UnlearningEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: UnlearningMetricCategories

    eval_type_id: str = Field(
        default=EVAL_TYPE_ID_UNLEARNING,
        title="Eval Type ID",
        description="The type of the evaluation",
    )
