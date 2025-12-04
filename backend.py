import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# --- CONFIGURATION & SETUP ---
app = FastAPI(title="LangGraph Personality Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. MOCK DATA ---
MOCK_CHAT_HISTORY = [
    "Hey, I'm struggling with this Python script today.",
    "I honestly feel like I'm not good enough to be a senior dev.",
    "My cat Luna knocked over my water glass again.",
    "I need a coffee. Badly. Espresso, specifically.",
    "I hate tea, it's just hot leaf juice.",
    "Deadlines are piling up and I feel paralyzed.",
    "Do you think impostor syndrome ever goes away?",
    "I spent 3 hours debugging a typo. I'm so stupid.",
    "Actually, I solved it. Felt good for about 5 minutes.",
    "Now I'm worried about the deployment next Tuesday.",
    "I prefer working late at night when it's quiet.",
    "My manager, Sarah, didn't reply to my email. Is she mad?",
    "I need to buy more dark roast beans.",
    "Sometimes I just want to quit and become a woodworker.",
    "Coding is fun when it works, but hell when it doesn't.",
    "I'm feeling a bit better after that second coffee.",
    "Luna is sleeping on my keyboard now. It's cute but annoying.",
    "I wish I could just focus on backend. I hate CSS.",
    "React is okay, but I prefer Vue if I'm being honest.",
    "Why is documentation always so outdated?",
    "I feel anxious every time I hear a Slack notification.",
    "Maybe I should go for a run. I usually like running.",
    "No, too tired. Netflix and pizza tonight.",
    "Pepperoni pizza is the only valid pizza.",
    "I'm scared I'll break production.",
    "Do you have any tips for burnout?",
    "I feel like everyone else knows what they're doing except me.",
    "Thanks for listening, usually I just bottle this up.",
    "I'm going to try the Pomodoro technique tomorrow.",
    "Okay, one last bug fix before bed."
]

# --- 2. DATA MODELS ---

class MemoryProfile(BaseModel):
    user_name: Optional[str] = Field(description="The user's name if mentioned.")
    preferences: List[str] = Field(description="Likes, dislikes, dietary choices, tool preferences.")
    emotional_patterns: List[str] = Field(description="Recurring emotional states, triggers, or coping mechanisms.")
    facts: List[str] = Field(description="Concrete facts: pet names, job details, specific people mentioned.")

class AnalysisRequest(BaseModel):
    query: str
    persona: str
    api_key: str

class AgentState(TypedDict):
    chat_history: List[str]
    current_query: str
    selected_persona: str
    memory: MemoryProfile
    standard_response: str
    personalized_response: str

# --- 3. NODE LOGIC ---

def get_llm(api_key: str):
    if not api_key:
        raise ValueError("API Key is missing")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.7)

def extract_memory_node(state: AgentState, config):
    api_key = config.get("configurable", {}).get("api_key")
    llm = get_llm(api_key)
    parser = PydanticOutputParser(pydantic_object=MemoryProfile)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert user researcher. Analyze the following chat logs and extract a structured memory profile. {format_instructions}"),
        ("user", "Chat Logs:\n{chat_logs}")
    ])
    
    history_text = "\n".join([f"- {msg}" for msg in state["chat_history"]])
    chain = prompt | llm | parser
    
    try:
        memory = chain.invoke({
            "chat_logs": history_text,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        print(f"Extraction Error: {e}")
        memory = MemoryProfile(user_name="Unknown", preferences=[], emotional_patterns=[], facts=[])

    return {"memory": memory}

def generate_standard_response_node(state: AgentState, config):
    api_key = config.get("configurable", {}).get("api_key")
    llm = get_llm(api_key)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        ("user", "{query}")
    ])
    response = prompt | llm
    result = response.invoke({"query": state["current_query"]})
    return {"standard_response": result.content}

def generate_personalized_response_node(state: AgentState, config):
    api_key = config.get("configurable", {}).get("api_key")
    llm = get_llm(api_key)
    memory = state["memory"]
    persona = state["selected_persona"]
    
    personas = {
        "calm_mentor": "You are a wise, senior mentor. You speak slowly and metaphorically. You focus on long-term growth and stability. Use the user's past experiences to guide them gently.",
        "witty_friend": "You are a chaotic good best friend. You use internet slang, emojis, and roast the user lovingly. You keep things brief and high energy.",
        "therapist": "You are a compassionate cognitive behavioral therapist. You validate feelings, ask reflective questions, and focus on emotional regulation and facts."
    }
    
    system_instruction = personas.get(persona, personas["calm_mentor"])
    
    context_block = f"""
    USER MEMORY CONTEXT:
    - Name: {memory.user_name}
    - Likes/Dislikes: {', '.join(memory.preferences)}
    - Emotional State: {', '.join(memory.emotional_patterns)}
    - Key Facts: {', '.join(memory.facts)}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{system_instruction}\n\nUse the following context to personalize the response deeply:\n{context_block}"),
        ("user", "{query}")
    ])
    
    response = prompt | llm
    result = response.invoke({"query": state["current_query"]})
    return {"personalized_response": result.content}

# --- 4. GRAPH CONSTRUCTION ---

workflow = StateGraph(AgentState)
workflow.add_node("extract_memory", extract_memory_node)
workflow.add_node("standard_gen", generate_standard_response_node)
workflow.add_node("persona_gen", generate_personalized_response_node)

workflow.set_entry_point("extract_memory")
workflow.add_edge("extract_memory", "standard_gen")
workflow.add_edge("extract_memory", "persona_gen")
workflow.add_edge("standard_gen", END)
workflow.add_edge("persona_gen", END)

app_graph = workflow.compile()

# --- 5. API ENDPOINTS ---

@app.post("/analyze")
async def analyze(request: AnalysisRequest):
    inputs = {
        "chat_history": MOCK_CHAT_HISTORY,
        "current_query": request.query,
        "selected_persona": request.persona,
        "memory": None, 
        "standard_response": "",
        "personalized_response": ""
    }
    config = {"configurable": {"api_key": request.api_key}}
    
    try:
        result = app_graph.invoke(inputs, config=config)
        return {
            "memory": result["memory"].dict(),
            "standard_response": result["standard_response"],
            "personalized_response": result["personalized_response"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def serve_ui():
    # Looks for index.html in the same directory
    return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)