"""Retrieval modules.

Three-stage pipeline:

1. :mod:`src.retrieval.retrieval_planner` — single LLM call that produces
   :class:`CurrentQueryFrame` + :class:`MemoryRetrievalPlan`.
2. :mod:`src.retrieval.layered_retriever` — deterministic memory pack assembly
   driven by the (frame, plan) pair.
3. :mod:`src.retrieval.response_generator` — final reply LLM call from the
   resulting :class:`MemoryPack`.
"""

from src.retrieval.layered_retriever import (
    LayeredRetriever,
    build_memory_pack,
    reset_default_layered_retriever,
)
from src.retrieval.response_generator import (
    RESPONSE_GENERATION_SYSTEM,
    ResponseGenerator,
    build_response_generation_prompt,
    generate_response,
    reset_default_response_generator,
)
from src.retrieval.retrieval_planner import (
    RETRIEVAL_PLAN_SYSTEM,
    LLMRetrievalPlanner,
    RuleRetrievalPlanner,
    build_retrieval_plan,
    build_retrieval_plan_prompt,
    reset_default_retrieval_planner,
)
from src.retrieval.schemas import (
    CurrentQueryFrame,
    MemoryPack,
    MemoryRetrievalPlan,
    SelectedBasicInfoClaim,
)

__all__ = [
    # schemas
    "CurrentQueryFrame",
    "MemoryRetrievalPlan",
    "SelectedBasicInfoClaim",
    "MemoryPack",
    # planner
    "RETRIEVAL_PLAN_SYSTEM",
    "LLMRetrievalPlanner",
    "RuleRetrievalPlanner",
    "build_retrieval_plan",
    "build_retrieval_plan_prompt",
    "reset_default_retrieval_planner",
    # retriever
    "LayeredRetriever",
    "build_memory_pack",
    "reset_default_layered_retriever",
    # response generator
    "RESPONSE_GENERATION_SYSTEM",
    "ResponseGenerator",
    "build_response_generation_prompt",
    "generate_response",
    "reset_default_response_generator",
]
