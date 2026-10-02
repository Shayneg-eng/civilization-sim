import os
"""
config.py — Centralized configuration for the neuron network.

Edit these values to customize the network's behaviour without
touching the core logic files.
"""

# ==============================================================================
# API Configuration
# ==============================================================================

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ==============================================================================
# Network Configuration (edit these to customize without CLI args)
# ==============================================================================

# The seed prompt / question that the network will explore
SEED_PROMPT = "What is the nature of consciousness?"

# Number of neurons in the network
NUM_NEURONS = 7

# Maximum number of rounds before terminating
MAX_ROUNDS = 20

# Convergence threshold (0-1): when all pairwise Jaccard scores exceed this, stop
CONVERGENCE_THRESHOLD = 0.85

# ==============================================================================
# System Prompt Factory
# ==============================================================================


def get_system_prompt(neuron_id: int) -> str:
    """
    Generate a system prompt for a neuron with the given id.

    Customize this function to change how neurons "think".
    You can use neuron_id to create distinct personas if desired.
    """
    # Example 1: All neurons share the same generic synthesis prompt
    return (
        f"You are neuron {neuron_id} in a fully-connected thinking network. "
        "Your role is to carefully read the seed prompt and any messages from "
        "your peer neurons, then synthesise a concise, thoughtful perspective "
        "of your own. Do not simply agree — integrate the ideas and add new "
        "insight where you can."
    )

    # Example 2: Uncomment below for neuron-specific personas
    # personas = [
    #     "You are a critical thinker. Challenge assumptions and dig deeper.",
    #     "You are a synthesizer. Combine ideas holistically.",
    #     "You are a pragmatist. Focus on practical implications.",
    #     "You are a visionary. Think boldly about possibilities.",
    #     "You are a skeptic. Question everything and ask for evidence.",
    # ]
    # return (
    #     f"You are neuron {neuron_id} in a thinking network. "
    #     f"Your archetype: {personas[neuron_id % len(personas)]} "
    #     "Read the seed and peer neurons' outputs, then respond with your own perspective."
    # )


# ==============================================================================
# Output Formatting
# ==============================================================================

# Whether to wrap neuron outputs when printing
WRAP_OUTPUT = True
OUTPUT_WIDTH = 68

# Separator characters for pretty printing
LINE_SEPARATOR = "─" * 72
ROUND_SEPARATOR = "━" * 72
FINAL_SEPARATOR = "═" * 72

# ==============================================================================
# Convergence Behavior
# ==============================================================================

# Similarity metric: "jaccard" (word-set Jaccard) or "llm" (LLM-based semantic)
SIMILARITY_METRIC = "llm"

# LLM similarity configuration (used when SIMILARITY_METRIC == "llm")
LLM_SIMILARITY_MODEL = "deepseek-chat"  # which model to use for scoring
LLM_SIMILARITY_SYSTEM_PROMPT = (
    "You are a similarity judge. Compare two text passages and rate how similar they are. "
    "Consider semantic meaning, not just word overlap. "
    "Return ONLY valid JSON with keys: 'score' (0.0-1.0), 'similar' (bool), 'reasoning' (str). "
    "Example: {\"score\": 0.85, \"similar\": true, \"reasoning\": \"Both discuss X\"}"
)

# Verbose output (more detailed round-by-round feedback)
VERBOSE = True
