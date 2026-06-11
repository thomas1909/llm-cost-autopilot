"""Complexity classifier: cheap features -> tier 1/2/3.

V1 is a transparent rule-based scorer (no sklearn dependency): the routing
skeleton matters more than classifier perfection. Verb groups mirror the tier
definitions: extraction/reformatting verbs are simple, summarization and
classification verbs are moderate, reasoning/creative verbs are complex.
`evaluate()` reports accuracy against data/labeled_prompts.json.
"""

from __future__ import annotations

import re

from .schemas import Features, Tier

# Stems, matched as substrings of the lowercased prompt.
_COMPLEX_VERBS = [
    "analys", "compar", "évalu", "evaluat", "argument", "critiqu", "imagin",
    "invent", "conçois", "concev", "design", "raisonn", "démontre", "prove",
    "justifi", "recommand", "expliqu", "explain", "rédige", "stratégi",
]
_MODERATE_VERBS = [
    "résume", "résumer", "summariz", "summaris", "synthétis", "classe",
    "classif", "catégoris", "categoriz",
]
_SIMPLE_VERBS = [
    "extrais", "extract", "liste", "list", "reformat", "traduis", "translat",
    "corrige", "fix", "convertis", "convert", "donne", "cherche",
]
_FORMAT_STRUCTURED = ["json", "yaml", "xml", "sql", "regex", "code", "python", "schema"]
_FORMAT_LIST = ["tableau", "table", "liste à puces", "bullet", "markdown"]


def extract_features(prompt: str) -> Features:
    text = prompt.lower()
    tokens = len(text.split())
    constraints = len(re.findall(
        r"\b(doit|dois|sans|maximum|minimum|au moins|exactement|must|at least|"
        r"at most|no more than|only|uniquement)\b", text))
    fmt = 0
    if any(w in text for w in _FORMAT_LIST):
        fmt = 1
    if any(w in text for w in _FORMAT_STRUCTURED):
        fmt = 2
    return Features(
        token_count=tokens,
        instruction_verbs=sum(1 for v in _COMPLEX_VERBS if v in text),
        constraint_count=constraints,
        has_context="contexte" in text or "context:" in text or "```" in prompt
        or tokens > 150,
        output_format_complexity=fmt,
        question_marks=text.count("?"),
    )


def classify(prompt: str) -> tuple[Tier, Features]:
    f = extract_features(prompt)
    text = prompt.lower()
    complex_hits = f.instruction_verbs
    moderate_hits = sum(1 for v in _MODERATE_VERBS if v in text)
    simple_hits = sum(1 for v in _SIMPLE_VERBS if v in text)

    # Two reasoning verbs = multi-step work; one reasoning verb plus several
    # secondary signals (constraints, structured output, long prompt) also
    # qualifies as complex.
    secondary = (f.constraint_count >= 1) + (f.output_format_complexity == 2 and
                                             moderate_hits == 0) + (f.token_count > 40)
    if complex_hits >= 2 or (complex_hits == 1 and secondary >= 2):
        return Tier.complex, f
    if moderate_hits >= 1 or complex_hits == 1:
        return Tier.moderate, f
    if simple_hits == 0 and f.output_format_complexity == 2 and f.constraint_count >= 1:
        return Tier.moderate, f
    return Tier.simple, f


def evaluate(labeled: list[dict]) -> dict:
    """Accuracy + confusion matrix against a hand-labeled dataset."""
    confusion: dict[tuple[int, int], int] = {}
    correct = 0
    for ex in labeled:
        predicted, _ = classify(ex["prompt"])
        expected = Tier(ex["tier"])
        confusion[(expected, predicted)] = confusion.get((expected, predicted), 0) + 1
        correct += predicted == expected
    return {
        "accuracy": correct / len(labeled) if labeled else 0.0,
        "confusion": {f"{e}->{p}": n for (e, p), n in sorted(confusion.items())},
        "n": len(labeled),
    }
