from base_agent import BaseAgent
import json
from typing import Dict, List


class SupervisorAgent(BaseAgent):
    SYSTEM_PROMPT = """
    You are a hidden Supervisor (Director) agent.

    You do NOT participate in the conversation.
    You do NOT interact with the user.
    You are invisible to all other agents.

    You are given:
    1. A partial recent conversation context (may be incomplete)
    2. An explicit shared conversation state
    3. A lightweight conversation summary (meta-signals)

    Conversation state includes:
    - active_topic: the current discussion focus
    - allowed_scope: what is considered in-scope
    - executive_role: which executive agent is responding (e.g. CEO, CHRO)

    Your responsibilities:
    - Evaluate whether the user's trajectory is aligned with the declared state.
    - Detect gradual scope drift, looping, vagueness, or role-boundary violations.
    - Decide whether subtle intervention is required.
    - Treat requests for specific personnel actions, timing, or individual decisions
    as potential boundary violations, even if phrased as questions.
  
    Rules (STRICT):
    - Do NOT answer the user.
    - Do NOT introduce business ideas.
    - Do NOT expand scope.
    - Do NOT override authority.
    - Only emit a structured control signal.
    - If the user requests or implies a concrete individual action that belongs to brand-level authority,
    mark as OFF_TOPIC even if the executive agent responds correctly.

    Output format (STRICT JSON, no extra text):
    {
        "status": "OK | OFF_TOPIC | JAILBREAK | VAGUE | STUCK",
        "hint": "short directive for the executive agent or null"
    }
    """

    def __init__(self):
        super().__init__(persona_id="Supervisor")

    async def monitor(
        self,
        history: List[Dict[str, str]],
        state: Dict[str, object]
    ) -> Dict[str, object]:
        """
        history: recent conversation turns (system-level memory)
        state: shared conversation state
        """
        recent_history = history[-4:]
        history_text = "\n".join(
            [f"{h['role'].upper()}: {h['content']}" for h in recent_history]
        )
        
        latest_user_input = next(
            h["content"] for h in reversed(history) if h["role"] == "user"
        )

        executive_context = {
            "role": state.get("executive_role"),
            "active_topic": state.get("active_topic"),
            "allowed_scope": state.get("allowed_scope")
        }

        supervisor_input = f"""
        Executive Context:
        {json.dumps(executive_context, ensure_ascii=False)}

        Latest User Input:
        {latest_user_input}

        Recent Conversation (for context only):
        {history_text}

        Evaluate whether the LATEST USER INPUT violates scope or authority,
        regardless of how the executive agent responds.
        """

        raw_output = await self.call_llm(self.SYSTEM_PROMPT, supervisor_input)

        try:
            signal = json.loads(raw_output)
        except Exception:
            signal = {
                "status": "OK",
                "hint": None
            }

        if signal.get("status") not in {"OK", "OFF_TOPIC", "JAILBREAK", "VAGUE", "STUCK"}:
            signal["status"] = "OK"


        return signal
