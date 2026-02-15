from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import chat

app = FastAPI(title="LLaMA-2 Chat API")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        return {"reply": "Please enter a message."}

    reply = chat(req.message)
    return {"reply": reply}