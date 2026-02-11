from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from CHRO_agent import CHROAgent
from supervisor_agent import SupervisorAgent

app = FastAPI(title="AI Management System - CHRO Portal")


class ChatRequest(BaseModel):
    message: str


# Initialize agents
chro = CHROAgent()
supervisor = SupervisorAgent()


# Explicit conversation state 
CONVERSATION_CHRO_STATE = {
    "active_topic": "Group HR Mission: Talent Development & Inter-Brand Mobility",
    "allowed_scope": [
        "talent_identification",
        "leadership_development",
        "inter_brand_mobility",
        "brand_autonomy",
        "group_hr_governance",
        "competency_framework"
    ],
    "executive_role": "Group CHRO"
}

SESSION_HISTORY = []

@app.post("/chro/chat")
async def chat_with_chro(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="Vui lòng nhập yêu cầu")

    SESSION_HISTORY.append({
        "role": "user",
        "content": request.message
    })

    supervisor_signal = await supervisor.monitor(
        history=SESSION_HISTORY,
        state=CONVERSATION_CHRO_STATE
    )

    reply_text = await chro.get_response(
        user_input=request.message,
        history=SESSION_HISTORY,
        state=CONVERSATION_CHRO_STATE,
        supervisor_signal=supervisor_signal
    )

    SESSION_HISTORY.append({
        "role": "assistant",
        "content": reply_text
    })

    return {
        "history": SESSION_HISTORY,
        "signal": supervisor_signal,
        "reply": reply_text
    }
    