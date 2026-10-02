"""
main.py — CLI entry point for the LLM neuron network.

This reads all configuration from config.py. To customize behavior,
edit the values in config.py directly — no command-line arguments needed.

Example config.py values to modify:
  • SEED_PROMPT — the initial question/stimulus
  • NUM_NEURONS — how many neurons in the network
  • MAX_ROUNDS — when to stop (or when convergence hits)
  • CONVERGENCE_THRESHOLD — similarity threshold for convergence
  • System prompts, formatting, API keys — all in config.py
"""

import sys

from config import (
    DEEPSEEK_API_KEY,
    MAX_ROUNDS,
    NUM_NEURONS,
    CONVERGENCE_THRESHOLD,
    SEED_PROMPT,
)
from network import Network


def validate_config() -> None:
    """Validate that required config values are set correctly."""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-deepseek-api-key-here":
        print(
            "Error: DEEPSEEK_API_KEY in config.py is not set.\n"
            "Edit config.py and replace 'your-deepseek-api-key-here' with your actual key.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not SEED_PROMPT or not SEED_PROMPT.strip():
        print(
            "Error: SEED_PROMPT in config.py is empty.",
            file=sys.stderr,
        )
        sys.exit(1)

    if NUM_NEURONS < 2:
        print(
            "Error: NUM_NEURONS in config.py must be at least 2.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not (0.0 < CONVERGENCE_THRESHOLD <= 1.0):
        print(
            "Error: CONVERGENCE_THRESHOLD in config.py must be in (0, 1].",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    validate_config()

    network = Network(
        n=NUM_NEURONS,
        seed=SEED_PROMPT,
        max_rounds=MAX_ROUNDS,
        threshold=CONVERGENCE_THRESHOLD,
    )
    network.run()


if __name__ == "__main__":
    main()
