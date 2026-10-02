"""
convergence.py — Checks whether a set of neuron outputs has converged.

Supports two similarity metrics:
  1. Jaccard: word-set based (simple, no external calls)
  2. LLM: semantic similarity via LLM judgment (accurate, requires API call)
"""

import json
from itertools import combinations
from typing import Optional

from openai import OpenAI


# =============================================================================
# Jaccard Similarity (baseline, no API calls)
# =============================================================================


def jaccard(a: str, b: str) -> float:
    """
    Compute the Jaccard similarity between two strings based on their
    word-level sets (case-insensitive).

    Returns a float in [0, 1].  Returns 1.0 if both strings are empty.
    """
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())

    if not set_a and not set_b:
        return 1.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# =============================================================================
# LLM-based Similarity (semantic, requires API call)
# =============================================================================


def llm_similarity(
    a: str,
    b: str,
    client: OpenAI,
    model: str = "deepseek-chat",
    system_prompt: str = "",
) -> dict:
    """
    Use an LLM to judge the semantic similarity between two strings.

    Returns a dict with:
      - score: float in [0, 1]
      - similar: bool (score >= 0.5)
      - reasoning: str

    If the LLM response is malformed, falls back to Jaccard.
    """
    if not system_prompt:
        system_prompt = (
            "You are a similarity judge. Compare two passages and rate similarity 0-1. "
            "Return ONLY JSON: {\"score\": 0.X, \"similar\": bool, \"reasoning\": \"...\"}"
        )

    user_message = f"Passage A:\n{a}\n\nPassage B:\n{b}\n\nHow similar are these?"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,  # Lower temperature for consistent judgments
            stream=False,
        )
        raw_text = response.choices[0].message.content
        result = json.loads(raw_text)

        # Validate and normalize
        score = float(result.get("score", 0.0))
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        similar = bool(result.get("similar", score >= 0.5))
        reasoning = str(result.get("reasoning", ""))

        return {
            "score": score,
            "similar": similar,
            "reasoning": reasoning,
        }

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        # Fallback to Jaccard if LLM fails
        print(f"[Warning] LLM similarity failed ({e}), falling back to Jaccard")
        score = jaccard(a, b)
        return {
            "score": score,
            "similar": score >= 0.5,
            "reasoning": f"Fallback to Jaccard: {score:.3f}",
        }


# =============================================================================
# Pairwise Scoring (supports both Jaccard and LLM)
# =============================================================================


def pairwise_scores(
    outputs: list[str],
    use_llm: bool = False,
    client: Optional[OpenAI] = None,
    llm_model: str = "deepseek-chat",
    llm_system_prompt: str = "",
) -> list[float]:
    """
    Return a list of similarity scores for every pair of outputs.

    With N outputs there are N*(N-1)/2 pairs.

    Parameters
    ----------
    outputs : list[str]
        List of neuron outputs to compare.
    use_llm : bool
        If True, use LLM-based similarity. If False, use Jaccard.
    client : OpenAI, optional
        OpenAI client (required if use_llm=True).
    llm_model : str
        Which LLM model to use for scoring.
    llm_system_prompt : str
        System prompt for the LLM judge.

    Returns
    -------
    list[float]
        Similarity scores in [0, 1].
    """
    if use_llm and client is None:
        raise ValueError("client is required when use_llm=True")

    scores = []
    for a, b in combinations(outputs, 2):
        if use_llm:
            result = llm_similarity(
                a,
                b,
                client,
                model=llm_model,
                system_prompt=llm_system_prompt,
            )
            scores.append(result["score"])
        else:
            scores.append(jaccard(a, b))

    return scores


# =============================================================================
# Convergence Checks
# =============================================================================


def has_converged(
    outputs: list[str],
    threshold: float = 0.75,
    use_llm: bool = False,
    client: Optional[OpenAI] = None,
    llm_model: str = "deepseek-chat",
    llm_system_prompt: str = "",
) -> bool:
    """
    Return True if the mean similarity score exceeds threshold.

    All neurons converge when their outputs are sufficiently similar.

    Parameters
    ----------
    outputs : list[str]
        List of neuron outputs.
    threshold : float
        Convergence threshold [0, 1].
    use_llm : bool
        If True, use LLM-based similarity. If False, use Jaccard.
    client : OpenAI, optional
        OpenAI client (required if use_llm=True).
    llm_model : str
        Which LLM model to use for scoring.
    llm_system_prompt : str
        System prompt for the LLM judge.

    Returns
    -------
    bool
        True if converged.
    """
    if len(outputs) < 2:
        return False

    mean = mean_similarity(
        outputs,
        use_llm=use_llm,
        client=client,
        llm_model=llm_model,
        llm_system_prompt=llm_system_prompt,
    )
    return mean >= threshold


# =============================================================================
# Convenience / Reporting
# =============================================================================


def mean_similarity(
    outputs: list[str],
    use_llm: bool = False,
    client: Optional[OpenAI] = None,
    llm_model: str = "deepseek-chat",
    llm_system_prompt: str = "",
) -> float:
    """
    Return the mean of all pairwise similarity scores.

    Useful for progress reporting.

    Parameters
    ----------
    outputs : list[str]
        List of neuron outputs.
    use_llm : bool
        If True, use LLM-based similarity. If False, use Jaccard.
    client : OpenAI, optional
        OpenAI client (required if use_llm=True).
    llm_model : str
        Which LLM model to use for scoring.
    llm_system_prompt : str
        System prompt for the LLM judge.

    Returns
    -------
    float
        Mean similarity in [0, 1].
    """
    scores = pairwise_scores(
        outputs,
        use_llm=use_llm,
        client=client,
        llm_model=llm_model,
        llm_system_prompt=llm_system_prompt,
    )
    return sum(scores) / len(scores) if scores else 0.0
