# ======================================================
# 🧠 DAY 4: TEACH-THE-TUTOR (BIOLOGY EDITION)
# ======================================================

import logging
import json
import os
import asyncio
from typing import Annotated, Literal, Optional
from dataclasses import dataclass

print("\n" + "🧬" * 50)
print("🚀 BIOLOGY TUTOR - DAY 4 TUTORIAL")
print("🧬" * 50 + "\n")

from dotenv import load_dotenv
from pydantic import Field

# 🔌 Correct LiveKit Imports (2025)
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ======================================================
# 📚 KNOWLEDGE BASE
# ======================================================

CONTENT_FILE = "biology_content.json"

DEFAULT_CONTENT = [
    {
        "id": "dna",
        "title": "DNA",
        "summary": "DNA carries genetic instructions and has a double helix structure.",
        "sample_question": "What is the full form of DNA and what is its structure called?"
    },
    {
        "id": "cell",
        "title": "The Cell",
        "summary": "The cell is the basic unit of life.",
        "sample_question": "What is the difference between Prokaryotic and Eukaryotic cells?"
    },
    {
        "id": "nucleus",
        "title": "Nucleus",
        "summary": "The nucleus controls the cell and stores DNA.",
        "sample_question": "Why is the nucleus called the control center?"
    },
    {
        "id": "cell_cycle",
        "title": "Cell Cycle",
        "summary": "The cell cycle includes interphase and mitosis.",
        "sample_question": "In which phase does the cell spend most of its time?"
    }
]


def load_content():
    try:
        path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)

        if not os.path.exists(path):
            print("⚠️ Content file missing — generating now...")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONTENT, f, indent=4)
            print("✅ Content generated.")
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print("⚠️ Failed to load content:", e)
        return DEFAULT_CONTENT


COURSE_CONTENT = load_content()

# ======================================================
# 🧠 STATE MANAGEMENT
# ======================================================

@dataclass
class TutorState:
    current_topic_id: str | None = None
    current_topic_data: dict | None = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"

    def set_topic(self, topic_id: str):
        topic = next((t for t in COURSE_CONTENT if t["id"] == topic_id), None)
        if topic:
            self.current_topic_id = topic_id
            self.current_topic_data = topic
            return True
        return False


@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None

# ======================================================
# 🛠️ TOOLS
# ======================================================

@function_tool
async def select_topic(ctx: RunContext[Userdata], topic_id: Annotated[str, Field(description="Topic ID")]):
    state = ctx.userdata.tutor_state

    if state.set_topic(topic_id.lower()):
        return f"Topic set to {state.current_topic_data['title']}. Ask user: Learn, Quiz, or Teach Back?"

    available = ", ".join([t["id"] for t in COURSE_CONTENT])
    return f"Topic not found. Available: {available}"


@function_tool
async def set_learning_mode(ctx: RunContext[Userdata], mode: Annotated[str, Field(description="learn, quiz, teach_back")]):
    state = ctx.userdata.tutor_state
    state.mode = mode.lower()
    session = ctx.userdata.agent_session

    if not session:
        return "Session not found."

    if state.mode == "learn":
        session.tts.update_options(voice="en-US-matthew", style="Promo")
        return f"Learn mode activated. Summary: {state.current_topic_data['summary']}"

    elif state.mode == "quiz":
        session.tts.update_options(voice="en-US-alicia", style="Conversational")
        return f"Quiz mode activated. Question: {state.current_topic_data['sample_question']}"

    elif state.mode == "teach_back":
        session.tts.update_options(voice="en-US-ken", style="Promo")
        return "Teach-back mode activated. Ask the user to explain the topic."

    return "Invalid mode."


@function_tool
async def evaluate_teaching(ctx: RunContext[Userdata], user_explanation: Annotated[str, Field(description="User explanation")]):
    return "Analyze user's explanation and rate it out of 10."

# ======================================================
# 🧠 AGENT
# ======================================================

class TutorAgent(Agent):
    def __init__(self):
        topics = ", ".join([f"{t['id']} ({t['title']})" for t in COURSE_CONTENT])

        super().__init__(
            instructions=f"""
            You are a Biology Tutor.

            Available topics: {topics}

            Modes:
            - Learn (explain)
            - Quiz (ask)
            - Teach Back (user explains)

            Always ask what topic the user wants first.
            """,
            tools=[select_topic, set_learning_mode, evaluate_teaching],
        )

# ======================================================
# 🎬 ENTRYPOINT
# ======================================================

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):

    userdata = Userdata(tutor_state=TutorState())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Promo", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    userdata.agent_session = session

    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
