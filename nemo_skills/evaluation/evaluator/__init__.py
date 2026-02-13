# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import importlib
from typing import Any, Callable, Dict

from nemo_skills.evaluation.evaluator.base import BaseEvaluator

# Lazy evaluator registry — stores dotted paths instead of eagerly importing
# every evaluator (which would pull in benchmark-specific deps like func_timeout,
# faiss, etc.). Actual imports happen on first use.

# Function-based evaluators (batch-only): eval_type -> "module_path:function_name"
_EVALUATOR_MAP_PATHS = {
    "evalplus": "nemo_skills.evaluation.evaluator.code:eval_evalplus",
    "if": "nemo_skills.evaluation.evaluator.ifeval:eval_if",
    "ifbench": "nemo_skills.evaluation.evaluator.ifbench:eval_ifbench",
    "bfcl": "nemo_skills.evaluation.evaluator.bfcl:eval_bfcl",
    "multichoice": "nemo_skills.evaluation.evaluator.mcq:eval_mcq",
    "ruler": "nemo_skills.evaluation.evaluator.ruler:eval_ruler",
    "ruler2": "nemo_skills.evaluation.evaluator.ruler:eval_ruler2",
    "livecodebench": "nemo_skills.evaluation.evaluator.livecodebench:eval_livecodebench",
    "livebench_coding": "nemo_skills.evaluation.evaluator.code:eval_livebench_coding",
    "livecodebench_pro": "nemo_skills.evaluation.evaluator.code:eval_livecodebench_pro",
    "scicode": "nemo_skills.evaluation.evaluator.scicode:eval_scicode",
    "mrcr": "nemo_skills.evaluation.evaluator.mrcr:eval_mrcr",
    "bigcodebench": "nemo_skills.evaluation.evaluator.code:eval_bigcodebench",
    "human_eval_infilling": "nemo_skills.evaluation.evaluator.code:eval_human_eval_infilling",
    "mmau-pro": "nemo_skills.evaluation.evaluator.mmau_pro:eval_mmau_pro",
}

# Class-based evaluators: eval_type -> "module_path:ClassName"
_EVALUATOR_CLASS_MAP_PATHS = {
    "math": "nemo_skills.evaluation.evaluator.math:MathEvaluator",
    "lean4-proof": "nemo_skills.evaluation.evaluator.math:Lean4ProofEvaluator",
    "code_exec": "nemo_skills.evaluation.evaluator.code:CodeExecEvaluator",
    "ioi": "nemo_skills.evaluation.evaluator.ioi:IOIEvaluator",
    "icpc": "nemo_skills.evaluation.evaluator.icpc:ICPCEvaluator",
    "audio": "nemo_skills.evaluation.evaluator.audio:AudioEvaluator",
    "bird": "nemo_skills.evaluation.evaluator.bird:BirdEvaluator",
    "compute-eval": "nemo_skills.evaluation.evaluator.compute_eval:ComputeEvalEvaluator",
}

# Validation: Ensure no overlap between class and function maps
_overlap = set(_EVALUATOR_CLASS_MAP_PATHS.keys()).intersection(_EVALUATOR_MAP_PATHS.keys())
if _overlap:
    raise ValueError(
        f"Evaluator types cannot be in both EVALUATOR_CLASS_MAP and EVALUATOR_MAP: {_overlap}. "
        f"Each eval_type must be in exactly one map."
    )

# Caches for resolved imports
_resolved_evaluator_map: Dict[str, Callable] = {}
_resolved_class_map: Dict[str, type] = {}


def _resolve(dotted: str):
    """Import 'module.path:AttributeName' and return the attribute."""
    module_path, attr_name = dotted.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _get_evaluator_fn(eval_type: str) -> Callable:
    if eval_type not in _resolved_evaluator_map:
        _resolved_evaluator_map[eval_type] = _resolve(_EVALUATOR_MAP_PATHS[eval_type])
    return _resolved_evaluator_map[eval_type]


def _get_evaluator_cls(eval_type: str) -> type:
    if eval_type not in _resolved_class_map:
        _resolved_class_map[eval_type] = _resolve(_EVALUATOR_CLASS_MAP_PATHS[eval_type])
    return _resolved_class_map[eval_type]


# --- Public API (unchanged signatures) ---

# Keep EVALUATOR_MAP and EVALUATOR_CLASS_MAP as lazy-resolving dicts for
# any code that iterates them (e.g. listing available types).
# Direct key access goes through the public functions below.
EVALUATOR_MAP = _EVALUATOR_MAP_PATHS
EVALUATOR_CLASS_MAP = _EVALUATOR_CLASS_MAP_PATHS


def is_evaluator_registered(eval_type: str):
    """Check if evaluator is registered in either class or function map."""
    return eval_type in _EVALUATOR_CLASS_MAP_PATHS or eval_type in _EVALUATOR_MAP_PATHS


def register_evaluator(eval_type: str, eval_fn: Callable[[Dict[str, Any]], None]):
    if is_evaluator_registered(eval_type):
        raise ValueError(f"Evaluator for {eval_type} already registered")

    _EVALUATOR_MAP_PATHS[eval_type] = "<dynamically-registered>"
    _resolved_evaluator_map[eval_type] = eval_fn


def get_evaluator_class(eval_type: str, config: Dict[str, Any]) -> BaseEvaluator:
    """Get evaluator instance by type."""
    if eval_type not in _EVALUATOR_CLASS_MAP_PATHS:
        all_types = sorted(list(_EVALUATOR_CLASS_MAP_PATHS.keys()) + list(_EVALUATOR_MAP_PATHS.keys()))
        raise ValueError(
            f"Evaluator class not found for type: {eval_type}.\n"
            f"Available types with class support: {list(_EVALUATOR_CLASS_MAP_PATHS.keys())}\n"
            f"All supported types: {all_types}"
        )

    evaluator_class = _get_evaluator_cls(eval_type)
    return evaluator_class(config)


def supports_single_eval(eval_type: str, config: Dict[str, Any]) -> bool:
    """Check if evaluator supports single data point evaluation during generation."""
    if eval_type not in _EVALUATOR_CLASS_MAP_PATHS:
        return False  # Only class-based evaluators support single eval

    evaluator = get_evaluator_class(eval_type, config)
    return evaluator.supports_single_eval()


def evaluate(eval_type, eval_config):
    """Main evaluation function that handles both class-based and function-based evaluators."""
    # Check if it's a class-based evaluator first
    if eval_type in _EVALUATOR_CLASS_MAP_PATHS:
        evaluator = get_evaluator_class(eval_type, eval_config)
        return asyncio.run(evaluator.eval_full())

    # Fall back to function-based evaluator
    if eval_type in _EVALUATOR_MAP_PATHS:
        return _get_evaluator_fn(eval_type)(eval_config)

    # Not found in either map
    all_types = list(_EVALUATOR_CLASS_MAP_PATHS.keys()) + list(_EVALUATOR_MAP_PATHS.keys())
    raise ValueError(f"Evaluator not found for type: {eval_type}.\nSupported types: {sorted(all_types)}")
