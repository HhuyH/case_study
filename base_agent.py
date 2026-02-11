from config import client, MODEL_ID, TEMPERATURE

class BaseAgent:
    def __init__(self, persona_id: str):
        self.persona_id = persona_id
        self.model_id = MODEL_ID
        self.memory = []

    async def call_llm(self, sys_prompt: str, user_input: str):
        response = client.models.generate_content(
            model=self.model_id,
            config={'system_instruction': sys_prompt, 'temperature': TEMPERATURE},
            contents=user_input
        )
        return response.text