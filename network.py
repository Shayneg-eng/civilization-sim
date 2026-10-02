"""
network.py — Orchestrates a fully-connected network of LLM neurons.

Every round each neuron receives:
  • the original seed prompt
  • the labelled last_output of every *other* neuron (from the previous round)

The run loop continues until either:
  • semantic similarity scores converge (using LLM or Jaccard)
  • `max_rounds` is reached
"""

import textwrap

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    FINAL_SEPARATOR,
    LINE_SEPARATOR,
    LLM_SIMILARITY_MODEL,
    LLM_SIMILARITY_SYSTEM_PROMPT,
    OUTPUT_WIDTH,
    ROUND_SEPARATOR,
    SIMILARITY_METRIC,
    WRAP_OUTPUT,
)
from convergence import has_converged, mean_similarity
from neuron import Neuron


class Network:
    def __init__(
        self,
        n: int,
        seed: str,
        max_rounds: int,
        threshold: float,
    ):
        if n < 2:
            raise ValueError("A network needs at least 2 neurons.")

        self.seed = seed
        self.max_rounds = max_rounds
        self.threshold = threshold
        self.neurons: list[Neuron] = [Neuron(neuron_id=i) for i in range(n)]
        
        # Set up client for convergence checking (used if SIMILARITY_METRIC == "llm")
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        self.use_llm = SIMILARITY_METRIC.lower() == "llm"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the network until convergence or the round cap is hit."""
        metric_name = "LLM Semantic" if self.use_llm else "Jaccard Word-Set"
        
        print(LINE_SEPARATOR)
        print(f"  SEED: {self.seed}")
        print(f"  Neurons: {len(self.neurons)}  |  Max rounds: {self.max_rounds}  |  Threshold: {self.threshold}")
        print(f"  Similarity metric: {metric_name}")
        print(LINE_SEPARATOR)

        for round_num in range(self.max_rounds):
            print(f"\n{ROUND_SEPARATOR}")
            print(f"  ROUND {round_num}")
            print(f"{ROUND_SEPARATOR}")

            self._run_round(round_num)
            self._print_outputs(round_num)

            outputs = [n.last_output for n in self.neurons]
            avg = mean_similarity(
                outputs,
                use_llm=self.use_llm,
                client=self.client if self.use_llm else None,
                llm_model=LLM_SIMILARITY_MODEL,
                llm_system_prompt=LLM_SIMILARITY_SYSTEM_PROMPT,
            )

            print(f"\n  [Convergence] mean {metric_name.lower()} similarity = {avg:.3f}  (threshold = {self.threshold})")

            if has_converged(
                outputs,
                threshold=self.threshold,
                use_llm=self.use_llm,
                client=self.client if self.use_llm else None,
                llm_model=LLM_SIMILARITY_MODEL,
                llm_system_prompt=LLM_SIMILARITY_SYSTEM_PROMPT,
            ):
                print("\n  ✓ Network converged — mean similarity exceeds threshold.")
                break
        else:
            print(f"\n  ✗ Reached max rounds ({self.max_rounds}) without convergence.")

        self._print_summary()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_round(self, round_num: int) -> None:
        """Fire every neuron once, passing the previous round's peer outputs."""
        for neuron in self.neurons:
            peer_outputs = {
                other.neuron_id: other.last_output
                for other in self.neurons
                if other.neuron_id != neuron.neuron_id
            }
            # On round 0 every neuron has empty last_output — pass along anyway;
            # the Neuron._build_user_message will show "no peer outputs yet".
            neuron.activate(
                seed=self.seed,
                peer_outputs={k: v for k, v in peer_outputs.items() if v},
                round_num=round_num,
            )

    def _print_outputs(self, round_num: int) -> None:
        for neuron in self.neurons:
            print(f"\n  [Neuron {neuron.neuron_id}]")
            if WRAP_OUTPUT:
                wrapped = textwrap.fill(
                    neuron.last_output,
                    width=OUTPUT_WIDTH,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
                print(wrapped)
            else:
                print(f"    {neuron.last_output}")

    def _print_summary(self) -> None:
        print(f"\n{FINAL_SEPARATOR}")
        print("  FINAL OUTPUTS")
        print(f"{FINAL_SEPARATOR}")
        for neuron in self.neurons:
            print(f"\n  [Neuron {neuron.neuron_id}]")
            if WRAP_OUTPUT:
                wrapped = textwrap.fill(
                    neuron.last_output,
                    width=OUTPUT_WIDTH,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
                print(wrapped)
            else:
                print(f"    {neuron.last_output}")
        print(f"\n{FINAL_SEPARATOR}\n")
