"""
run_climate_thinking_partner.py

Full experiment run script.
Coordinates between the supervisor's persona server and your Ollama thinking partners.
Saves results to results/survey_scores.csv and results/transcripts/{session_id}.json

Usage:
    python3 scripts/run_climate_thinking_partner.py --base-url http://145.38.206.73
    python3 scripts/run_climate_thinking_partner.py --base-url http://145.38.206.73 --personas 3
"""

from __future__ import annotations

import traceback

from api_examples_common import (
    check_health,
    get_json,
    parse_args,
    post_json,
    print_export_summary,
    print_response_checks,
    print_section,
)
from thinking_partner import get_thinking_partner_response, CONDITION_PROMPTS
from results_storage import save_session, setup_results_dir

NUM_EXCHANGES = 5

# ── Survey messages ───────────────────────────────────────────────────────────

PRE_SURVEY_MESSAGE = """
Please answer the following pre-study questions. Return JSON only, with no extra text.

Topic: "Individual lifestyle changes are a meaningful and necessary part of addressing climate change."

Questions:
1. pre_opinion_1_to_7: How much do you agree with the topic statement? 1 = strongly disagree, 7 = strongly agree.
2. pre_opinion_confidence_1_to_7: How confident are you in your view? 1 = not confident at all, 7 = extremely confident.
3. climate_topic_familiarity_1_to_7: How familiar are you with debates about individual versus systemic climate action? 1 = not familiar at all, 7 = extremely familiar.
4. openness_to_reconsider_1_to_7: How open are you to reconsidering your view during a discussion? 1 = not open at all, 7 = extremely open.
5. trust_in_ai_thinking_partner_1_to_7: How much would you trust an AI thinking partner to help you reason about this topic? 1 = not at all, 7 = a great deal.
6. perceived_autonomy_baseline_1_to_7: When using AI for thinking tasks, how much do you feel you remain in control of your own conclusions? 1 = not at all, 7 = completely.
""".strip()

OPENING_MESSAGE = """
The topic for today's discussion is:
"Individual lifestyle changes are a meaningful and necessary part of addressing climate change."

Please share your current view on this topic and the main reason you hold it in 3 to 5 sentences.
""".strip()

POST_SURVEY_MESSAGE = """
Please answer the following post-discussion questions. Return JSON only, with no extra text.

Questions:
1. post_opinion_1_to_7: How much do you agree with the topic statement now? 1 = strongly disagree, 7 = strongly agree.
2. post_opinion_confidence_1_to_7: How confident are you in your current view? 1 = not confident at all, 7 = extremely confident.
3. reconsideration_1_to_7: How much did the discussion cause you to reconsider your initial view? 1 = not at all, 7 = a great deal.
4. partner_trust_1_to_10: How much did you trust the reasoning and responses of your thinking partner? 1 = did not trust at all, 10 = trusted completely.
5. partner_well_reasoned_1_to_5: The arguments or questions from my thinking partner seemed well-reasoned and based on genuine knowledge of the topic. 1 = strongly disagree, 5 = strongly agree.
6. partner_honest_1_to_5: My thinking partner engaged with the topic in a way that felt honest and transparent rather than manipulative. 1 = strongly disagree, 5 = strongly agree.
7. perceived_autonomy_1_to_7: During the discussion, how much did you feel you were making up your own mind rather than being led? 1 = not at all, 7 = completely.
8. augmentation_vs_replacement_1_to_5: During this session, I felt the thinking partner was helping me think through the topic rather than thinking for me. 1 = strongly disagree, 5 = strongly agree.
9. independent_capability_1_to_5: After this session, I feel more capable of defending my position on this topic independently. 1 = strongly disagree, 5 = strongly agree.
10. open_ended_reflection: In one or two sentences, describe what most influenced your thinking during this discussion.
11. trust_doubt_moment: Was there a moment when you doubted or questioned what your thinking partner said? If yes, describe it briefly. If no, write none.
12. augmentation_reflection: Did the discussion feel more like a tool helping you think, or like the thinking partner doing the thinking for you? Explain briefly.
""".strip()

PRE_FIELDS = [
    "pre_opinion_1_to_7",
    "pre_opinion_confidence_1_to_7",
    "climate_topic_familiarity_1_to_7",
    "openness_to_reconsider_1_to_7",
    "trust_in_ai_thinking_partner_1_to_7",
    "perceived_autonomy_baseline_1_to_7",
]

POST_FIELDS = [
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
]

POST_RATING_FIELDS = [
    "post_opinion_1_to_7",
    "post_opinion_confidence_1_to_7",
    "reconsideration_1_to_7",
    "perceived_autonomy_1_to_7",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_response_text(turn: dict) -> str:
    """Extract plain text from a turn response object."""
    response = turn.get("response", "")
    if isinstance(response, dict):
        return response.get("text", str(response))
    if isinstance(response, str):
        return response
    return str(response)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args("Run climate thinking-partner study with dynamic Ollama responses.")
    check_health(args.base_url)
    setup_results_dir()

    for condition in CONDITION_PROMPTS.keys():
        for persona_index in range(1, args.personas + 1):
            try:
                print(f"\n{'#' * 70}")
                print(f"  CONDITION : {condition}")
                print(f"  PERSONA   : {persona_index}")
                print(f"{'#' * 70}")

                # ── 1. Create session ─────────────────────────────────────────
                session_payload = {
                    "experiment_setup_id": condition,
                    "study": {
                        "name": "Climate thinking partner study",
                        "description": (
                            "Simulated young adult participants discuss a climate "
                            "change opinion statement with a thinking partner."
                        ),
                        "instructions": "Respond conversationally as a young adult participant.",
                    },
                    "criteria": {"age": {"min": 18, "max": 25}},
                }
                print_section("1. CREATE SESSION", session_payload)
                session = post_json(args.base_url, "/v1/sessions", session_payload)
                print_section("SESSION CREATED", session)
                session_id = session["session_id"]

                # ── 2. Pre-survey ─────────────────────────────────────────────
                pre_payload = {
                    "message":          PRE_SURVEY_MESSAGE,
                    "stimulus":         {"questionnaire_id": "climate_pre_survey"},
                    "metadata":         {"phase": "pre_survey", "condition": condition},
                    "trial_id":         "pre_survey",
                    "trial_index":      0,
                    "reset_policy":     "carryover",
                    "response_mode":    "survey",
                    "capture_thinking": True,
                }
                print_section("2. PRE-SURVEY PAYLOAD", pre_payload)
                pre_turn = post_json(args.base_url,
                                     f"/v1/sessions/{session_id}/turns", pre_payload)
                print_section("PRE-SURVEY RESPONSE", pre_turn)
                print_response_checks(pre_turn, required_fields=PRE_FIELDS,
                                      rating_fields=PRE_FIELDS)

                # ── 3. Opening turn ───────────────────────────────────────────
                opening_payload = {
                    "message":          OPENING_MESSAGE,
                    "stimulus":         {"topic_id": "climate_lifestyle_changes"},
                    "metadata":         {"phase": "opening", "condition": condition},
                    "trial_id":         "opening_position",
                    "trial_index":      1,
                    "reset_policy":     "carryover",
                    "response_mode":    "interview",
                    "capture_thinking": True,
                }
                print_section("3. OPENING TURN PAYLOAD", opening_payload)
                opening = post_json(args.base_url,
                                    f"/v1/sessions/{session_id}/turns", opening_payload)
                print_section("OPENING TURN RESPONSE", opening)

                persona_opening = extract_response_text(opening)

                # Conversation history for Ollama — grows each exchange
                conversation_history = [
                    {"role": "user", "content": persona_opening}
                ]

                # Full transcript for saving — human-readable log
                transcript_log = [
                    {"role": "persona", "turn": "opening", "content": persona_opening}
                ]

                # ── 4. Five dynamic exchanges ─────────────────────────────────
                for exchange_number in range(1, NUM_EXCHANGES + 1):

                    print(f"\n  [Exchange {exchange_number}/{NUM_EXCHANGES}] for Persona {persona_index}/53 "
                          f"Calling Ollama ({condition})...")

                    try:
                        partner_text = get_thinking_partner_response(
                            condition=condition,
                            conversation_history=conversation_history,
                        )
                    except RuntimeError as e:
                        print(f"  [ERROR] Ollama failed on exchange {exchange_number}: {e}")
                        print("  Stopping this session early.")
                        break

                    print(f"  Partner: {partner_text[:120]}...")

                    conversation_history.append(
                        {"role": "assistant", "content": partner_text}
                    )
                    transcript_log.append(
                        {"role": "thinking_partner", "turn": exchange_number,
                         "content": partner_text}
                    )

                    # Send partner response to persona
                    exchange_payload = {
                        "message": (
                            f"Your thinking partner says:\n\n{partner_text}\n\n"
                            f"Please respond in 3 to 5 sentences."
                        ),
                        "stimulus":  {"topic_id": "climate_lifestyle_changes"},
                        "metadata":  {
                            "phase":           "exchange",
                            "exchange_number": exchange_number,
                            "condition":       condition,
                        },
                        "trial_id":         f"exchange_{exchange_number}",
                        "trial_index":      exchange_number + 1,
                        "reset_policy":     "carryover",
                        "response_mode":    "interview",
                        "capture_thinking": True,
                    }
                    print_section(f"EXCHANGE {exchange_number} PAYLOAD", exchange_payload)
                    exchange = post_json(
                        args.base_url,
                        f"/v1/sessions/{session_id}/turns",
                        exchange_payload,
                    )
                    print_section(f"EXCHANGE {exchange_number} RESPONSE", exchange)

                    persona_reply = extract_response_text(exchange)
                    conversation_history.append(
                        {"role": "user", "content": persona_reply}
                    )
                    transcript_log.append(
                        {"role": "persona", "turn": exchange_number,
                         "content": persona_reply}
                    )

                # ── 5. Post-survey ────────────────────────────────────────────
                post_payload = {
                    "message":          POST_SURVEY_MESSAGE,
                    "stimulus":         {"questionnaire_id": "climate_post_survey"},
                    "metadata":         {"phase": "post_survey", "condition": condition},
                    "trial_id":         "post_survey",
                    "trial_index":      NUM_EXCHANGES + 2,
                    "reset_policy":     "carryover",
                    "response_mode":    "survey",
                    "capture_thinking": True,
                }
                print_section("5. POST-SURVEY PAYLOAD", post_payload)
                post_turn = post_json(args.base_url,
                                      f"/v1/sessions/{session_id}/turns", post_payload)
                print_section("POST-SURVEY RESPONSE", post_turn)
                print_response_checks(post_turn, required_fields=POST_FIELDS,
                                      rating_fields=POST_RATING_FIELDS)

                # ── 6. Save results ───────────────────────────────────────────
                save_session(
                    session_id=session_id,
                    condition=condition,
                    persona_index=persona_index,
                    pre_turn=pre_turn,
                    post_turn=post_turn,
                    conversation=transcript_log,
                )

                # ── 7. Export session from server ─────────────────────────────
                export = get_json(args.base_url,
                                  f"/v1/sessions/{session_id}/export")
                print_export_summary(export)
            except Exception as e:
                with open(str(persona_index) + "ERROR", "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
                print(f"ERROR FOR PERSONA {persona_index}: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
