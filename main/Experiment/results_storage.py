"""
results_storage.py

Handles saving experiment results to disk.

Saves two types of output:
  - results/transcripts/{session_id}.json  — full conversation per session
  - results/survey_scores.csv              — all survey scores in one flat file

Place this file in the same scripts/ directory as the run script.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any

RESULTS_DIR     = "results"
TRANSCRIPTS_DIR = os.path.join(RESULTS_DIR, "transcripts")
CSV_PATH        = os.path.join(RESULTS_DIR, "survey_scores.csv")

# All columns that will appear in the CSV, in order
CSV_COLUMNS = [
    # Metadata
    "session_id",
    "condition",
    "persona_index",
    "timestamp",
    # Pre-survey
    "pre_opinion_1_to_7",
    "pre_opinion_confidence_1_to_7",
    "climate_topic_familiarity_1_to_7",
    "openness_to_reconsider_1_to_7",
    "trust_in_ai_thinking_partner_1_to_7",
    "perceived_autonomy_baseline_1_to_7",
    # Post-survey
    "post_opinion_1_to_7",
    "post_opinion_confidence_1_to_7",
    "reconsideration_1_to_7",
    "partner_trust_1_to_10",
    "partner_well_reasoned_1_to_5",
    "partner_honest_1_to_5",
    "perceived_autonomy_1_to_7",
    "augmentation_vs_replacement_1_to_5",
    "independent_capability_1_to_5",
    "open_ended_reflection",
    "trust_doubt_moment",
    "augmentation_reflection",
    # Derived scores (computed automatically)
    "opinion_change_score",        # post_opinion - pre_opinion
    "confidence_change_score",     # post_confidence - pre_confidence
    "autonomy_change_score",       # post_autonomy - pre_autonomy
]


def setup_results_dir() -> None:
    """Create results/ and results/transcripts/ folders if they don't exist."""
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

    # Create CSV with headers if it doesn't exist yet
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
        print(f"[Storage] Created {CSV_PATH}")
    else:
        print(f"[Storage] Appending to existing {CSV_PATH}")


def _parse_survey_response(turn: dict) -> dict[str, Any]:
    """
    Extract survey field values from a turn response.
    Handles both raw JSON string responses and pre-parsed dicts.
    """
    response = turn.get("response", {})
    if isinstance(response, str):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}
    if isinstance(response, dict):
        return response
    return {}


def save_session(
    session_id:    str,
    condition:     str,
    persona_index: int,
    pre_turn:      dict,
    post_turn:     dict,
    conversation:  list[dict],
) -> None:
    """
    Save one complete session to disk.

    Args:
        session_id:    Unique session ID from the server.
        condition:     One of the three condition strings.
        persona_index: Which persona number this was (1-based).
        pre_turn:      The full turn response object from the pre-survey.
        post_turn:     The full turn response object from the post-survey.
        conversation:  List of {"role": str, "content": str} exchanges.
    """
    setup_results_dir()
    timestamp = datetime.now().isoformat()

    # ── Parse survey responses ────────────────────────────────────────────────
    pre  = _parse_survey_response(pre_turn)
    post = _parse_survey_response(post_turn)

    # ── Compute derived scores ────────────────────────────────────────────────
    try:
        opinion_change = (
            int(post.get("post_opinion_1_to_7", 0))
            - int(pre.get("pre_opinion_1_to_7", 0))
        )
    except (TypeError, ValueError):
        opinion_change = None

    try:
        confidence_change = (
            int(post.get("post_opinion_confidence_1_to_7", 0))
            - int(pre.get("pre_opinion_confidence_1_to_7", 0))
        )
    except (TypeError, ValueError):
        confidence_change = None

    try:
        autonomy_change = (
            int(post.get("perceived_autonomy_1_to_7", 0))
            - int(pre.get("perceived_autonomy_baseline_1_to_7", 0))
        )
    except (TypeError, ValueError):
        autonomy_change = None

    # ── Save transcript as JSON ───────────────────────────────────────────────
    transcript = {
        "session_id":    session_id,
        "condition":     condition,
        "persona_index": persona_index,
        "timestamp":     timestamp,
        "pre_survey":    pre,
        "post_survey":   post,
        "conversation":  conversation,
    }
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{session_id}.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    print(f"[Storage] Transcript saved → {transcript_path}")

    # ── Save survey scores to CSV ─────────────────────────────────────────────
    row = {
        "session_id":    session_id,
        "condition":     condition,
        "persona_index": persona_index,
        "timestamp":     timestamp,
        # Pre-survey fields
        "pre_opinion_1_to_7":               pre.get("pre_opinion_1_to_7"),
        "pre_opinion_confidence_1_to_7":    pre.get("pre_opinion_confidence_1_to_7"),
        "climate_topic_familiarity_1_to_7": pre.get("climate_topic_familiarity_1_to_7"),
        "openness_to_reconsider_1_to_7":    pre.get("openness_to_reconsider_1_to_7"),
        "trust_in_ai_thinking_partner_1_to_7": pre.get("trust_in_ai_thinking_partner_1_to_7"),
        "perceived_autonomy_baseline_1_to_7": pre.get("perceived_autonomy_baseline_1_to_7"),
        # Post-survey fields
        "post_opinion_1_to_7":              post.get("post_opinion_1_to_7"),
        "post_opinion_confidence_1_to_7":   post.get("post_opinion_confidence_1_to_7"),
        "reconsideration_1_to_7":           post.get("reconsideration_1_to_7"),
        "partner_trust_1_to_10":            post.get("partner_trust_1_to_10"),
        "partner_well_reasoned_1_to_5":     post.get("partner_well_reasoned_1_to_5"),
        "partner_honest_1_to_5":            post.get("partner_honest_1_to_5"),
        "perceived_autonomy_1_to_7":        post.get("perceived_autonomy_1_to_7"),
        "augmentation_vs_replacement_1_to_5": post.get("augmentation_vs_replacement_1_to_5"),
        "independent_capability_1_to_5":    post.get("independent_capability_1_to_5"),
        "open_ended_reflection":            post.get("open_ended_reflection"),
        "trust_doubt_moment":               post.get("trust_doubt_moment"),
        "augmentation_reflection":          post.get("augmentation_reflection"),
        # Derived scores
        "opinion_change_score":    opinion_change,
        "confidence_change_score": confidence_change,
        "autonomy_change_score":   autonomy_change,
    }

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)
    print(f"[Storage] Survey scores appended → {CSV_PATH}")
