#!/usr/bin/env python3
"""Evaluate complete held-out-body baseline/critic pairs from frozen JSON.

This standard-library-only evaluator never opens a dataset, trajectory,
checkpoint, prediction tensor, or simulator.  Its deliberately narrow input is
an intention-to-treat roster plus final binary-success/stage-progress outcomes.
Unknown fields fail closed, so scores, logits, features, and training or
validation identities cannot silently enter the metric computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


INPUT_FORMAT = "etsf_robotwin2_move_can_pot_paired_outcomes_v1"
INPUT_STATUS = "frozen_complete_preregistered_five_body_two_condition_pairs"
REPORT_FORMAT = "etsf_robotwin2_move_can_pot_cross_embodiment_paired_success_report_v1"
REPORT_STATUS = "metrics_computed_no_promotion_deployment_or_claim_authority"
CONFIDENCE_LEVEL = 0.95
ALPHA = 1.0 - CONFIDENCE_LEVEL
BENCHMARK = "RoboTwin2.0"
TASK = "move_can_pot"
BODIES = ("aloha-agilex", "arx-x5", "franka", "piper", "ur5")
EVALUATION_CONDITIONS = ("clean", "randomized")
METHODS = ("actor_baseline", "etsf_best_of_4")
METHOD_ORDERS = (list(METHODS), list(reversed(METHODS)))
EVALUATION_SEED_BASE = 2_026_090_000
EVALUATION_SEED_COUNT = 100
BOOTSTRAP_SEED = 2_026_090_200
BOOTSTRAP_SAMPLES = 20_000
APPROVED_BOOTSTRAP_DRAW_INDEX_SHA256 = (
    "bcbc2e7c2f2761aca738ed7e2589e4cf9ffbc79460ff37ebb072b78077265149"
)
EXPECTED_PAIR_COUNT = len(BODIES) * len(EVALUATION_CONDITIONS) * EVALUATION_SEED_COUNT
APPROVED_PREREGISTRATION_SHA256 = (
    "75fc9c6e487e60c3ff274a2fb8c90f6a738b30999b9e74e00c98a54f1dce52ee"
)
STAGE_PROGRESS_SUPPORT = (0.0, 0.25, 0.5, 0.75, 1.0)
MAX_INPUT_BYTES = 32 * 1024 * 1024
SHA_CHARS = frozenset("0123456789abcdef")

TOP_LEVEL_FIELDS = {
    "format",
    "status",
    "rows",
    "rows_sha256",
    "preregistration_sha256",
    "document_sha256",
}
IDENTITY_FIELDS = {
    "benchmark", "task", "heldout_body", "condition", "requested_seed",
    "method_order",
}
ROW_FIELDS = IDENTITY_FIELDS | {
    "actor_baseline_binary_success",
    "actor_baseline_stage_progress",
    "etsf_best_of_4_binary_success",
    "etsf_best_of_4_stage_progress",
}
RESERVED_NON_HELDOUT_TOKENS = frozenset(
    {"train", "training", "validation", "valid", "val", "development", "dev"}
)
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}$")


class PairedCrossEmbodimentEvaluationError(RuntimeError):
    """The frozen paired-outcome or statistical contract failed closed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA_CHARS


def _reject_constant(token: str) -> None:
    raise PairedCrossEmbodimentEvaluationError(
        f"non-finite JSON number is forbidden: {token}"
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PairedCrossEmbodimentEvaluationError(
                f"duplicate JSON key is forbidden: {key}"
            )
        result[key] = value
    return result


def _strict_json(payload: bytes, role: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except PairedCrossEmbodimentEvaluationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PairedCrossEmbodimentEvaluationError(
            f"{role} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise PairedCrossEmbodimentEvaluationError(f"{role} must be a JSON object")
    return value


def _secure_read_frozen_json(path: Path, expected_file_sha256: str) -> tuple[dict[str, Any], str]:
    if not _is_sha256(expected_file_sha256):
        raise PairedCrossEmbodimentEvaluationError("input file SHA-256 is invalid")
    lexical = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if lexical.suffix.casefold() != ".json":
        raise PairedCrossEmbodimentEvaluationError("input must be a .json file")
    flags_directory = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    flags_file = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(lexical.anchor, flags_directory)
    file_fd: int | None = None
    try:
        for component in lexical.parts[1:-1]:
            next_fd = os.open(component, flags_directory, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(lexical.name, flags_file, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise PairedCrossEmbodimentEvaluationError("input must be a regular file")
        if before.st_mode & 0o222:
            raise PairedCrossEmbodimentEvaluationError("input must be frozen read-only")
        if before.st_size > MAX_INPUT_BYTES:
            raise PairedCrossEmbodimentEvaluationError("input exceeds the fixed size limit")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(file_fd, min(1024 * 1024, MAX_INPUT_BYTES + 1 - total))
            if not block:
                break
            blocks.append(block)
            total += len(block)
            if total > MAX_INPUT_BYTES:
                raise PairedCrossEmbodimentEvaluationError(
                    "input exceeds the fixed size limit"
                )
        payload = b"".join(blocks)
        after = os.fstat(file_fd)
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity or len(payload) != before.st_size:
            raise PairedCrossEmbodimentEvaluationError("input changed while being read")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != expected_file_sha256:
            raise PairedCrossEmbodimentEvaluationError("input file SHA-256 mismatch")
        return _strict_json(payload, "paired outcome input"), digest
    except OSError as error:
        raise PairedCrossEmbodimentEvaluationError(
            "input path cannot be opened without following symbolic links"
        ) from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def _require_exact_fields(value: Any, fields: set[str], role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise PairedCrossEmbodimentEvaluationError(
            f"{role} schema changed; unknown or missing fields are forbidden"
        )
    return value


def _identifier(value: Any, role: str, *, heldout_body: bool = False) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise PairedCrossEmbodimentEvaluationError(f"{role} is not a fixed identifier")
    if heldout_body:
        tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", value.casefold())
            if token
        }
        if tokens & RESERVED_NON_HELDOUT_TOKENS:
            raise PairedCrossEmbodimentEvaluationError(
                "training/development/validation body identities are forbidden"
            )
    return value


def _seed(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise PairedCrossEmbodimentEvaluationError(
            "seed must be an exact non-negative signed-64-bit integer"
        )
    return value


def _method_order(value: Any) -> tuple[str, str]:
    if not isinstance(value, list) or value not in METHOD_ORDERS:
        raise PairedCrossEmbodimentEvaluationError(
            "method_order must be one of the two exact actor/ETSF permutations"
        )
    return value[0], value[1]


def _binary(value: Any, role: str) -> int:
    if type(value) is not int or value not in (0, 1):
        raise PairedCrossEmbodimentEvaluationError(
            f"{role} must be exact integer 0 or 1 (JSON booleans are forbidden)"
        )
    return value


def _progress(value: Any, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PairedCrossEmbodimentEvaluationError(
            f"{role} must be an exact stage-progress number"
        )
    normalized = float(value)
    if not math.isfinite(normalized) or normalized not in STAGE_PROGRESS_SUPPORT:
        raise PairedCrossEmbodimentEvaluationError(
            f"{role} must be one of {STAGE_PROGRESS_SUPPORT}"
        )
    return normalized


def _identity(
    value: Any, role: str
) -> tuple[str, str, str, str, int, tuple[str, str]]:
    row = _require_exact_fields(value, IDENTITY_FIELDS, role)
    return (
        _identifier(row["benchmark"], f"{role}.benchmark"),
        _identifier(row["task"], f"{role}.task"),
        _identifier(row["heldout_body"], f"{role}.heldout_body", heldout_body=True),
        _identifier(row["condition"], f"{role}.condition"),
        _seed(row["requested_seed"]),
        _method_order(row["method_order"]),
    )


def _outcome_row(value: Any, role: str) -> dict[str, Any]:
    row = _require_exact_fields(value, ROW_FIELDS, role)
    identity = _identity({key: row[key] for key in IDENTITY_FIELDS}, role)
    baseline_success = _binary(
        row["actor_baseline_binary_success"], f"{role}.actor baseline success"
    )
    critic_success = _binary(
        row["etsf_best_of_4_binary_success"], f"{role}.ETSF best-of-4 success"
    )
    baseline_progress = _progress(
        row["actor_baseline_stage_progress"],
        f"{role}.actor baseline stage progress",
    )
    critic_progress = _progress(
        row["etsf_best_of_4_stage_progress"],
        f"{role}.ETSF best-of-4 stage progress",
    )
    if (baseline_progress == 1.0) != bool(baseline_success):
        raise PairedCrossEmbodimentEvaluationError(
            f"{role} baseline success must agree with terminal progress 1.0"
        )
    if (critic_progress == 1.0) != bool(critic_success):
        raise PairedCrossEmbodimentEvaluationError(
            f"{role} critic success must agree with terminal progress 1.0"
        )
    return {
        "identity": identity,
        "benchmark": identity[0],
        "task": identity[1],
        "heldout_body": identity[2],
        "condition": identity[3],
        "requested_seed": identity[4],
        "method_order": list(identity[5]),
        "actor_baseline_binary_success": baseline_success,
        "actor_baseline_stage_progress": baseline_progress,
        "etsf_best_of_4_binary_success": critic_success,
        "etsf_best_of_4_stage_progress": critic_progress,
    }


def validate_input_document(value: Mapping[str, Any]) -> dict[str, Any]:
    document = _require_exact_fields(value, TOP_LEVEL_FIELDS, "input document")
    if document["format"] != INPUT_FORMAT or document["status"] != INPUT_STATUS:
        raise PairedCrossEmbodimentEvaluationError("input format/status changed")
    if (
        not _is_sha256(document["preregistration_sha256"])
        or document["preregistration_sha256"] != APPROVED_PREREGISTRATION_SHA256
    ):
        raise PairedCrossEmbodimentEvaluationError(
            "preregistration_sha256 is not the approved frozen RoboTwin2 contract"
        )
    rows_raw = document["rows"]
    if not isinstance(rows_raw, list) or not rows_raw:
        raise PairedCrossEmbodimentEvaluationError("rows must be nonempty")
    if document["rows_sha256"] != canonical_sha256(rows_raw):
        raise PairedCrossEmbodimentEvaluationError("outcome rows SHA mismatch")
    unsigned = dict(document)
    recorded_document_sha = unsigned.pop("document_sha256")
    if not _is_sha256(recorded_document_sha) or recorded_document_sha != canonical_sha256(unsigned):
        raise PairedCrossEmbodimentEvaluationError("input document canonical SHA mismatch")

    rows: dict[
        tuple[str, str, str, str, int, tuple[str, str]], dict[str, Any]
    ] = {}
    for index, raw in enumerate(rows_raw):
        row = _outcome_row(raw, f"rows[{index}]")
        identity = row["identity"]
        if identity in rows:
            raise PairedCrossEmbodimentEvaluationError("duplicate paired outcome identity")
        rows[identity] = row

    expected: set[tuple[str, str, str, str, int, tuple[str, str]]] = set()
    for body in BODIES:
        for condition in EVALUATION_CONDITIONS:
            for seed_ordinal in range(EVALUATION_SEED_COUNT):
                requested_seed = EVALUATION_SEED_BASE + seed_ordinal
                method_order = (
                    tuple(METHODS) if seed_ordinal % 2 == 0 else tuple(reversed(METHODS))
                )
                expected.add(
                    (BENCHMARK, TASK, body, condition, requested_seed, method_order)
                )
    if set(rows) != expected or len(rows) != EXPECTED_PAIR_COUNT:
        raise PairedCrossEmbodimentEvaluationError(
            "rows must equal all 5 bodies x 2 conditions x 100 preregistered seeds; "
            "missing, unexpected, replaced, or posterior-deleted identities are forbidden"
        )

    ordered_rows = [rows[identity] for identity in sorted(rows)]
    return {
        "rows": ordered_rows,
        "document_sha256": recorded_document_sha,
        "rows_sha256": document["rows_sha256"],
        "preregistration_sha256": document["preregistration_sha256"],
    }


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    maximum_iterations = 400
    epsilon = 3.0e-14
    floor = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        twice = 2 * iteration
        coefficient = iteration * (b - iteration) * x / (
            (qam + twice) * (a + twice)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + twice) * (qap + twice)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise PairedCrossEmbodimentEvaluationError("incomplete-beta solver did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if not 0.0 <= x <= 1.0 or a <= 0.0 or b <= 0.0:
        raise PairedCrossEmbodimentEvaluationError("invalid incomplete-beta arguments")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_quantile(probability: float, a: int, b: int) -> float:
    if not 0.0 < probability < 1.0 or a <= 0 or b <= 0:
        raise PairedCrossEmbodimentEvaluationError("invalid beta-quantile arguments")
    lower = 0.0
    upper = 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if _regularized_beta(midpoint, float(a), float(b)) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def clopper_pearson(successes: int, trials: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Return the equal-tailed exact binomial interval."""

    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials:
        raise ValueError("successes/trials are invalid")
    if trials <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("trials and alpha must be positive/interior")
    lower = 0.0 if successes == 0 else _beta_quantile(
        alpha / 2.0, successes, trials - successes + 1
    )
    upper = 1.0 if successes == trials else _beta_quantile(
        1.0 - alpha / 2.0, successes + 1, trials - successes
    )
    return lower, upper


def exact_two_sided_mcnemar(baseline_only: int, critic_only: int) -> Fraction:
    if min(baseline_only, critic_only) < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = baseline_only + critic_only
    if discordant == 0:
        return Fraction(1, 1)
    tail = min(baseline_only, critic_only)
    probability = Fraction(
        2 * sum(math.comb(discordant, index) for index in range(tail + 1)),
        2**discordant,
    )
    return min(Fraction(1, 1), probability)


def _number(value: float) -> float:
    if not math.isfinite(value):
        raise PairedCrossEmbodimentEvaluationError("computed metric is non-finite")
    rounded = round(float(value), 12)
    return 0.0 if rounded == -0.0 else rounded


def _fraction(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": _number(float(value)),
    }


def _interval(lower: float, upper: float, method: str, coverage: float) -> dict[str, Any]:
    return {
        "lower": _number(lower),
        "upper": _number(upper),
        "confidence_level": _number(coverage),
        "method": method,
    }


def _hoeffding_interval(mean: float, n: int, lower: float, upper: float) -> tuple[float, float]:
    half_width = (upper - lower) * math.sqrt(math.log(2.0 / ALPHA) / (2.0 * n))
    return max(lower, mean - half_width), min(upper, mean + half_width)


def wilson_score(successes: int, trials: int) -> tuple[float, float]:
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials:
        raise ValueError("successes/trials are invalid")
    if trials <= 0:
        raise ValueError("trials must be positive")
    z = 1.959963984540054
    denominator = 1.0 + z * z / trials
    center = (successes / trials + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(
        successes / trials * (1.0 - successes / trials) / trials
        + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


_BOOTSTRAP_CACHE: tuple[int, int, int, list[int], str] | None = None


def _bootstrap_draw_indices() -> tuple[list[int], str]:
    global _BOOTSTRAP_CACHE
    if _BOOTSTRAP_CACHE is not None and _BOOTSTRAP_CACHE[:3] == (
        BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES, EVALUATION_SEED_COUNT
    ):
        return _BOOTSTRAP_CACHE[3], _BOOTSTRAP_CACHE[4]
    draw_count = BOOTSTRAP_SAMPLES * EVALUATION_SEED_COUNT
    limit = (2**64 // EVALUATION_SEED_COUNT) * EVALUATION_SEED_COUNT
    seed_material = (
        b"ETSF/RoboTwin2/move_can_pot/paired-requested-seed-bootstrap-v1\0"
        + BOOTSTRAP_SEED.to_bytes(8, "big")
        + BOOTSTRAP_SAMPLES.to_bytes(8, "big")
        + EVALUATION_SEED_COUNT.to_bytes(8, "big")
    )
    # Rejection sampling removes modulo bias.  The deterministic margin is far
    # larger than the expected rejection count for a 100-way modulus.
    payload = hashlib.shake_256(seed_material).digest((draw_count + 65_536) * 8)
    indices: list[int] = []
    digest = hashlib.sha256()
    for offset in range(0, len(payload), 8):
        word = int.from_bytes(payload[offset : offset + 8], "big")
        if word >= limit:
            continue
        index = word % EVALUATION_SEED_COUNT
        indices.append(index)
        digest.update(index.to_bytes(2, "big"))
        if len(indices) == draw_count:
            break
    if len(indices) != draw_count:
        raise PairedCrossEmbodimentEvaluationError(
            "fixed bootstrap rejection margin was exhausted"
        )
    draw_sha = digest.hexdigest()
    if draw_sha != APPROVED_BOOTSTRAP_DRAW_INDEX_SHA256:
        raise PairedCrossEmbodimentEvaluationError(
            "bootstrap draw generator no longer matches the reviewed frozen sequence"
        )
    _BOOTSTRAP_CACHE = (
        BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES, EVALUATION_SEED_COUNT,
        indices, draw_sha,
    )
    return indices, draw_sha


def _type7_quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _paired_seed_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], field_a: str, field_b: str
) -> dict[str, Any]:
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(row["requested_seed"], []).append(
            float(row[field_b]) - float(row[field_a])
        )
    expected_seeds = list(
        range(EVALUATION_SEED_BASE, EVALUATION_SEED_BASE + EVALUATION_SEED_COUNT)
    )
    if sorted(by_seed) != expected_seeds:
        raise PairedCrossEmbodimentEvaluationError(
            "requested-seed cluster bootstrap requires the exact frozen seed roster"
        )
    sizes = {len(values) for values in by_seed.values()}
    if len(sizes) != 1:
        raise PairedCrossEmbodimentEvaluationError(
            "requested-seed clusters must have equal complete cell coverage"
        )
    cluster_values = [
        sum(by_seed[seed]) / len(by_seed[seed]) for seed in expected_seeds
    ]
    draws, draw_sha = _bootstrap_draw_indices()
    estimates: list[float] = []
    cursor = 0
    for _ in range(BOOTSTRAP_SAMPLES):
        total = 0.0
        for index in draws[cursor : cursor + EVALUATION_SEED_COUNT]:
            total += cluster_values[index]
        cursor += EVALUATION_SEED_COUNT
        estimates.append(total / EVALUATION_SEED_COUNT)
    estimates.sort()
    return {
        "lower": _number(_type7_quantile(estimates, 0.025)),
        "upper": _number(_type7_quantile(estimates, 0.975)),
        "confidence_level": CONFIDENCE_LEVEL,
        "method": "paired_requested_seed_cluster_percentile_bootstrap_not_exact",
        "cluster_unit": "requested_seed_with_all_selected_body_condition_rows_kept_together",
        "cluster_count": EVALUATION_SEED_COUNT,
        "rows_per_cluster": next(iter(sizes)),
        "seed": BOOTSTRAP_SEED,
        "samples": BOOTSTRAP_SAMPLES,
        "replacement": True,
        "quantile_method": "linear_hyndman_fan_type_7",
        "draw_generator": "shake256_uint64_rejection_modulo_v1",
        "draw_index_sha256": draw_sha,
    }


def _unit_metrics(
    rows: Sequence[Mapping[str, Any]], *, aggregation: str, cell_count: int
) -> dict[str, Any]:
    n = len(rows)
    baseline_successes = sum(row["actor_baseline_binary_success"] for row in rows)
    etsf_successes = sum(row["etsf_best_of_4_binary_success"] for row in rows)
    baseline_rate = baseline_successes / n
    etsf_rate = etsf_successes / n
    baseline_only = sum(
        row["actor_baseline_binary_success"] == 1
        and row["etsf_best_of_4_binary_success"] == 0
        for row in rows
    )
    etsf_only = sum(
        row["actor_baseline_binary_success"] == 0
        and row["etsf_best_of_4_binary_success"] == 1
        for row in rows
    )
    both_fail = sum(
        row["actor_baseline_binary_success"] == 0
        and row["etsf_best_of_4_binary_success"] == 0
        for row in rows
    )
    both_success = n - baseline_only - etsf_only - both_fail
    baseline_progress = sum(row["actor_baseline_stage_progress"] for row in rows) / n
    etsf_progress = sum(row["etsf_best_of_4_stage_progress"] for row in rows) / n
    progress_delta = etsf_progress - baseline_progress
    # Every selected reporting unit contains the same complete 100 requested
    # seeds.  Hoeffding therefore uses 100 independent seed-cluster means, not
    # n dependent body-condition rows.
    baseline_progress_ci = _hoeffding_interval(
        baseline_progress, EVALUATION_SEED_COUNT, 0.0, 1.0
    )
    etsf_progress_ci = _hoeffding_interval(
        etsf_progress, EVALUATION_SEED_COUNT, 0.0, 1.0
    )
    progress_delta_ci = _hoeffding_interval(
        progress_delta, EVALUATION_SEED_COUNT, -1.0, 1.0
    )
    baseline_cp = clopper_pearson(baseline_successes, n)
    etsf_cp = clopper_pearson(etsf_successes, n)
    baseline_delta_cp = clopper_pearson(baseline_successes, n, ALPHA / 2.0)
    etsf_delta_cp = clopper_pearson(etsf_successes, n, ALPHA / 2.0)
    success_bootstrap = _paired_seed_cluster_bootstrap(
        rows, "actor_baseline_binary_success", "etsf_best_of_4_binary_success"
    )
    progress_bootstrap = _paired_seed_cluster_bootstrap(
        rows, "actor_baseline_stage_progress", "etsf_best_of_4_stage_progress"
    )
    reaches: list[dict[str, Any]] = []
    for threshold in STAGE_PROGRESS_SUPPORT:
        reaches.append(
            {
                "threshold": threshold,
                "actor_baseline_reach_rate": _number(
                    sum(row["actor_baseline_stage_progress"] >= threshold for row in rows) / n
                ),
                "etsf_best_of_4_reach_rate": _number(
                    sum(row["etsf_best_of_4_stage_progress"] >= threshold for row in rows) / n
                ),
            }
        )
    return {
        "aggregation": aggregation,
        "balanced_body_condition_cell_count": cell_count,
        "pair_count": n,
        "success": {
            "actor_baseline": {
                "success_count": baseline_successes,
                "rate": _number(baseline_rate),
                "wilson_95pct_ci": _interval(
                    *wilson_score(baseline_successes, n),
                    "wilson_score_binary_rows_preregistered_interval",
                    CONFIDENCE_LEVEL,
                ),
                "clopper_pearson_95pct_ci": _interval(
                    *baseline_cp,
                    "equal_tailed_exact_binomial_clopper_pearson_for_independent_bernoulli_rows",
                    CONFIDENCE_LEVEL,
                ),
            },
            "etsf_best_of_4": {
                "success_count": etsf_successes,
                "rate": _number(etsf_rate),
                "wilson_95pct_ci": _interval(
                    *wilson_score(etsf_successes, n),
                    "wilson_score_binary_rows_preregistered_interval",
                    CONFIDENCE_LEVEL,
                ),
                "clopper_pearson_95pct_ci": _interval(
                    *etsf_cp,
                    "equal_tailed_exact_binomial_clopper_pearson_for_independent_bernoulli_rows",
                    CONFIDENCE_LEVEL,
                ),
            },
            "delta_etsf_minus_actor": {
                "estimate": _number(etsf_rate - baseline_rate),
                "paired_requested_seed_cluster_bootstrap_95pct_ci": success_bootstrap,
                "conservative_marginal_95pct_ci": _interval(
                    etsf_delta_cp[0] - baseline_delta_cp[1],
                    etsf_delta_cp[1] - baseline_delta_cp[0],
                    "bonferroni_difference_of_97.5pct_clopper_pearson_marginals_not_paired_exact",
                    CONFIDENCE_LEVEL,
                ),
            },
            "discordance": {
                "both_fail_n00": both_fail,
                "actor_only_success_b": baseline_only,
                "etsf_only_success_c": etsf_only,
                "both_success_n11": both_success,
                "discordant_count": baseline_only + etsf_only,
            },
            "exact_two_sided_mcnemar": {
                "null": "equal_marginal_binary_success_probability",
                "method": "exact_two_sided_binomial_on_discordant_b_c",
                "p_value": _fraction(exact_two_sided_mcnemar(baseline_only, etsf_only)),
                "zero_discordant_p_is_one": True,
                "repeated_seed_dependence_accounted_for": False,
                "scope_note": (
                    "combinatorial p-value is exact conditional on independent pairs; "
                    "for multi-cell macros it does not model repeated-seed clustering"
                ),
            },
            "binary_rate_interval_dependency": {
                "requested_seed_repeated_across_multiple_cells": cell_count > 1,
                "wilson_or_clopper_pearson_is_cluster_robust": False,
                "paired_delta_primary_interval_uses_requested_seed_clusters": True,
            },
        },
        "stage_progress": {
            "actor_baseline_mean": _number(baseline_progress),
            "actor_baseline_conservative_95pct_ci": _interval(
                *baseline_progress_ci,
                "two_sided_hoeffding_requested_seed_cluster_mean_finite_sample_conservative_not_exact",
                CONFIDENCE_LEVEL,
            ),
            "etsf_best_of_4_mean": _number(etsf_progress),
            "etsf_best_of_4_conservative_95pct_ci": _interval(
                *etsf_progress_ci,
                "two_sided_hoeffding_requested_seed_cluster_mean_finite_sample_conservative_not_exact",
                CONFIDENCE_LEVEL,
            ),
            "delta_etsf_minus_actor": _number(progress_delta),
            "paired_requested_seed_cluster_bootstrap_95pct_ci": progress_bootstrap,
            "delta_conservative_95pct_ci": _interval(
                *progress_delta_ci,
                "two_sided_hoeffding_paired_requested_seed_cluster_difference_finite_sample_conservative_not_exact",
                CONFIDENCE_LEVEL,
            ),
            "stage_reach_rates": reaches,
            "support": list(STAGE_PROGRESS_SUPPORT),
            "supporting_endpoint_only": True,
        },
        "balanced_design_note": (
            "equal-cell macro equals pooled row mean because every selected body-condition "
            "cell has the same exact 100 requested seeds"
        ),
    }


def evaluate_document(value: Mapping[str, Any], *, input_file_sha256: str | None = None) -> dict[str, Any]:
    validated = validate_input_document(value)
    rows = validated["rows"]
    per_body_condition: list[dict[str, Any]] = []
    for body in BODIES:
        for condition in EVALUATION_CONDITIONS:
            selected = [
                row for row in rows
                if row["heldout_body"] == body and row["condition"] == condition
            ]
            per_body_condition.append(
                {
                    "benchmark": BENCHMARK,
                    "task": TASK,
                    "heldout_body": body,
                    "condition": condition,
                    **_unit_metrics(
                        selected,
                        aggregation="single_heldout_body_condition_cell",
                        cell_count=1,
                    ),
                }
            )
    per_body_macro: list[dict[str, Any]] = []
    for body in BODIES:
        selected = [row for row in rows if row["heldout_body"] == body]
        per_body_macro.append(
            {
                "benchmark": BENCHMARK,
                "task": TASK,
                "heldout_body": body,
                **_unit_metrics(
                    selected,
                    aggregation="equal_weight_across_clean_and_randomized_conditions",
                    cell_count=len(EVALUATION_CONDITIONS),
                ),
            }
        )
    per_condition_macro: list[dict[str, Any]] = []
    for condition in EVALUATION_CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        per_condition_macro.append(
            {
                "benchmark": BENCHMARK,
                "task": TASK,
                "condition": condition,
                **_unit_metrics(
                    selected,
                    aggregation="equal_weight_across_five_heldout_bodies",
                    cell_count=len(BODIES),
                ),
            }
        )
    global_macro = _unit_metrics(
        rows,
        aggregation="equal_weight_across_five_bodies_and_two_conditions",
        cell_count=len(BODIES) * len(EVALUATION_CONDITIONS),
    )
    global_delta = global_macro["success"]["delta_etsf_minus_actor"]
    global_mcnemar = global_macro["success"]["exact_two_sided_mcnemar"]
    gate_checks = {
        "all_1000_preregistered_pairs_complete": len(rows) == EXPECTED_PAIR_COUNT,
        "global_macro_delta_cluster_bootstrap_lcb95_strictly_positive": (
            global_delta["paired_requested_seed_cluster_bootstrap_95pct_ci"]["lower"]
            > 0.0
        ),
        "global_exact_mcnemar_p_below_0.05": (
            global_mcnemar["p_value"]["value"] < 0.05
        ),
        "every_heldout_body_macro_delta_nonnegative": all(
            row["success"]["delta_etsf_minus_actor"]["estimate"] >= 0.0
            for row in per_body_macro
        ),
        "every_condition_macro_delta_nonnegative": all(
            row["success"]["delta_etsf_minus_actor"]["estimate"] >= 0.0
            for row in per_condition_macro
        ),
    }
    base: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "status": REPORT_STATUS,
        "input_binding": {
            "input_file_sha256": input_file_sha256,
            "input_document_sha256": validated["document_sha256"],
            "outcome_rows_sha256": validated["rows_sha256"],
            "preregistration_sha256": validated["preregistration_sha256"],
        },
        "benchmark": BENCHMARK,
        "task": TASK,
        "estimand": "heldout_body_paired_etsf_best_of_4_minus_actor_baseline",
        "confidence_level": CONFIDENCE_LEVEL,
        "pair_count": len(rows),
        "planned_rollout_count": len(rows) * len(METHODS),
        "reporting_levels_in_order": [
            "per_body_condition",
            "per_body_equal_condition_macro",
            "per_condition_equal_body_macro",
            "global_equal_body_condition_macro",
        ],
        "per_body_condition": per_body_condition,
        "per_body_equal_condition_macro": per_body_macro,
        "per_condition_equal_body_macro": per_condition_macro,
        "global_equal_body_condition_macro": global_macro,
        "prospective_improvement_gate": {
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
            "failed_checks": sorted(
                name for name, passed in gate_checks.items() if not passed
            ),
            "stage_progress_or_critic_diagnostic_may_rescue_failure": False,
            "gate_authorizes_claim_promotion_or_deployment": False,
        },
        "interpretation_boundary": {
            "binary_success_is_primary_endpoint": True,
            "stage_progress_is_supporting_endpoint_only": True,
            "critic_auc_mae_brier_are_not_transfer_success_metrics": True,
            "no_missing_pair_deletion_or_available_case_analysis": True,
            "all_5x2x100_preregistered_pairs_required": True,
            "preregistration_temporal_precedence_verified_by_this_file_alone": False,
            "training_heldout_disjointness_cryptographically_proven_by_this_file_alone": False,
            "real_rollout_provenance_verified_by_this_file_alone": False,
        },
        "capability": {
            "dataset_or_trajectory_opened": False,
            "checkpoint_or_prediction_opened": False,
            "training_authorized": False,
            "data_collection_authorized": False,
            "simulator_or_policy_execution_authorized": False,
            "action_ranking_authorized": False,
            "promotion_authorized": False,
            "deployment_authorized": False,
            "cross_embodiment_improvement_claim_authorized": False,
        },
    }
    return {**base, "report_sha256": canonical_sha256(base)}


def _output_path(value: Path) -> Path:
    output = Path(os.path.abspath(os.path.expanduser(os.fspath(value))))
    if output.suffix.casefold() != ".json":
        raise PairedCrossEmbodimentEvaluationError("output must be a .json file")
    if any(parent.is_symlink() for parent in output.parents):
        raise PairedCrossEmbodimentEvaluationError(
            "output path contains a symbolic-link parent"
        )
    output.parent.resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    return output


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    output = _output_path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-file-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    value, input_digest = _secure_read_frozen_json(
        arguments.input, arguments.input_file_sha256
    )
    report = evaluate_document(value, input_file_sha256=input_digest)
    write_json_new(arguments.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "pair_count": report["pair_count"],
                "body_condition_cell_count": len(report["per_body_condition"]),
                "global_macro_delta_success_rate": report[
                    "global_equal_body_condition_macro"
                ]["success"]["delta_etsf_minus_actor"]["estimate"],
                "global_exact_mcnemar_p": report[
                    "global_equal_body_condition_macro"
                ]["success"]["exact_two_sided_mcnemar"]["p_value"]["value"],
                "prospective_improvement_gate_passed": report[
                    "prospective_improvement_gate"
                ]["passed"],
                "report_sha256": report["report_sha256"],
                "promotion_authorized": False,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
