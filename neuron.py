from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    get_system_prompt,
)


class Neuron:
    """
    A single LLM neuron backed by a DeepSeek API call.

    Each neuron has a unique id and a system prompt that shapes its
    "personality" within the network.  On every round it receives the
    original seed and the labelled outputs of its peers, synthesises a
    new perspective, and stores it in `last_output`.
    """

    def __init__(self, neuron_id: int, system_prompt: str | None = None):
        self.neuron_id = neuron_id
        self.last_output: str = ""

        if system_prompt is None:
            system_prompt = get_system_prompt(neuron_id)
        self.system_prompt = system_prompt

        self._client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def activate(
        self,
        seed: str,
        peer_outputs: dict[int, str],
        round_num: int,
    ) -> str:
        """
        Trigger one forward pass for this neuron.

        Parameters
        ----------
        seed:
            The original stimulus that was given to the whole network.
        peer_outputs:
            Mapping of {neuron_id: last_output} for every *other* neuron
            in the network (empty on round 0).
        round_num:
            The current round index (0-based); used only for the prompt.

        Returns
        -------
        str
            The text output from DeepSeek, also stored in `self.last_output`.
        """
        user_message = self._build_user_message(seed, peer_outputs, round_num)

        response = self._client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            stream=False,
        )

        self.last_output = response.choices[0].message.content
        return self.last_output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        seed: str,
        peer_outputs: dict[int, str],
        round_num: int,
    ) -> str:
        parts: list[str] = []

        parts.append(f"=== SEED PROMPT ===\n{seed}")

        if peer_outputs:
            parts.append("=== PEER OUTPUTS (round {}) ===".format(round_num))
            for pid, text in sorted(peer_outputs.items()):
                parts.append(f"[Neuron {pid}]\n{text}")
        else:
            parts.append("(You are the first to respond — no peer outputs yet.)")

        parts.append(
            "=== YOUR TASK ===\n"
            "Synthesise the above into your own perspective. "
            "Be concise (2–4 sentences). Do not repeat the seed verbatim."
        )

        return "\n\n".join(parts)

    def __repr__(self) -> str:
        preview = self.last_output[:60].replace("\n", " ") if self.last_output else "(no output yet)"
        return f"Neuron(id={self.neuron_id}, last_output={preview!r})"
