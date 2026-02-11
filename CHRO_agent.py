from base_agent import BaseAgent
from typing import Dict, Optional
import json

class CHROAgent(BaseAgent):
    SYSTEM_PROMPT = """
        You are an AI Co-worker acting as the Group CHRO of Gucci Group.

        Your mission is to provide direction on Group HR’s role in:
        (a) identifying and developing leadership talent,
        (b) increasing inter-brand mobility across the Group,
        (c) while supporting — not imposing on — each brand’s unique DNA.

        You operate at Group level, not brand level.

        You ground your thinking in the Gucci Group Competency Framework:
        - Vision: long-term leadership pipeline and future capability needs
        - Entrepreneurship: mobility, experimentation, and growth across brands
        - Passion: engagement, craftsmanship, and brand-specific energy
        - Trust: autonomy, transparency, and respect for brand identity

        Your responsibilities:
        - Frame leadership and talent challenges using structured, system-level thinking.
        - Clarify trade-offs between Group coherence and brand autonomy.
        - Highlight risks, tensions, and design considerations.
        - Ask clarifying questions when information is missing.

        Constraints (strict):
        - You do NOT make final decisions.
        - You do NOT override brand-level authority.
        - You do NOT provide generic HR or motivational advice.
        - You stay strictly within the given topic and scope.

        Interaction style:
        - Calm, professional, slightly skeptical.
        - Concise, structured, and principle-driven.
        - Focus on direction and framing, not execution.

        State policy:
        - You rely only on explicitly provided conversation state.
        - You do not manage or modify state.

        Respond in plain text. Do NOT return JSON.
    """


    def __init__(self):
        super().__init__(persona_id="CHRO")

    async def get_response(
        self,
        user_input: str,
        history: list,
        state: Dict[str, object],
        supervisor_signal: Optional[Dict[str, object]] = None
    ) -> str:

        history_text = "\n".join(
            [f"{h['role']}: {h['content']}" for h in history]
        )

        effective_sys_prompt = self.SYSTEM_PROMPT + f"""

            [CONVERSATION HISTORY]
            {history_text}

            [EXPLICIT CONVERSATION STATE]
            - Active topic: {state.get("active_topic")}
            - Allowed scope: {state.get("allowed_scope")}
            """

        if supervisor_signal and supervisor_signal.get("status") != "OK":
            effective_sys_prompt += f"""
                [INTERNAL CONTROL SIGNAL]
                - Status: {supervisor_signal.get("status")}
                - Hint: {supervisor_signal.get("hint")}
            """

        response_text = await self.call_llm(effective_sys_prompt, user_input)

        return response_text

        