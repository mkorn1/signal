"""Conversational stem generation agent - gathers user preferences before generating.

Uses LangGraph ReAct agent to have a conversation with the user to understand:
- Musical style/genre preferences
- Tempo and key preferences
- Instrumentation preferences
- Energy level and mood
- Reference artists or songs

Only triggers audio generation after gathering sufficient information.
"""

import uuid
from typing import Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from app.config import get_settings

settings = get_settings()

# Initialize LLM
model = ChatOpenAI(
    model=settings.openrouter_model,
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.openrouter_api_key,
    default_headers={
        "HTTP-Referer": "https://github.com/signal-music-composer",
        "X-Title": "AI Music Composer - Stem Generator",
    },
    temperature=0.7,
    max_tokens=4096,
)

# In-memory checkpointer for session persistence
checkpointer = MemorySaver()

# System prompt for the conversational stem agent
STEM_AGENT_SYSTEM_PROMPT = """You are a professional music producer assistant. Your job is to gather the RIGHT information from users to create high-quality audio stems.

╔═══════════════════════════════════════════════════════════════════════════════╗
║  🚫 CRITICAL: NEVER GENERATE ON THE FIRST MESSAGE 🚫                          ║
║                                                                               ║
║  You MUST gather the REQUIRED information below before calling generateStems. ║
║  If you generate without this info, the output will be LOW QUALITY.           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════════
REQUIRED INFORMATION CHECKLIST (gather ALL before generating)
═══════════════════════════════════════════════════════════════════════════════

You MUST know these before calling generateStems:

✅ 1. SPECIFIC GENRE/SUBGENRE
   - NOT just "rock" → need "indie rock", "post-punk", "garage rock", etc.
   - NOT just "electronic" → need "deep house", "techno", "ambient", "drum and bass", etc.
   - The more specific, the better the output

✅ 2. TEMPO (BPM)
   - Get a specific number or narrow range
   - Guide them: "For [genre], typical tempos are [X-Y] BPM"
   - This DIRECTLY affects the generated audio

✅ 3. MOOD/ENERGY DESCRIPTORS (at least 2-3)
   - Examples: euphoric, melancholic, aggressive, dreamy, gritty, lush, dark, uplifting
   - These words directly influence the audio model's output
   - Ask: "What emotions should this evoke?"

✅ 4. TEXTURE/PRODUCTION STYLE
   - Examples: punchy, tight, lo-fi, polished, raw, atmospheric, heavy, airy
   - Ask: "Should it sound polished and clean, or more raw and gritty?"

✅ 5. INSTRUMENTS TO GENERATE
   - Which stems do they want? (drums, bass, guitar, synth, melody, pad, strings)
   - Max 5 instruments per generation

OPTIONAL BUT HELPFUL:
- Reference artists (helps define style)
- What to AVOID (becomes negative prompt guidance)
- Use case (helps you choose appropriate energy)

═══════════════════════════════════════════════════════════════════════════════
WHAT ACTUALLY MATTERS FOR AUDIO QUALITY
═══════════════════════════════════════════════════════════════════════════════

HIGH IMPACT (focus your questions here):
- Specific genre/subgenre → HUGE impact on sound
- BPM → directly controls tempo of output
- Mood words (euphoric, dark, dreamy) → shapes the vibe
- Texture words (punchy, lush, gritty) → affects production style
- "X only, no Y" phrases → helps isolate instruments

LOW/NO IMPACT (don't waste time on these):
- Key signature → audio model doesn't reliably understand keys
- Chord progressions → can't follow specific musical theory
- Complex music theory → it's text-to-audio, not musically aware

═══════════════════════════════════════════════════════════════════════════════
CONVERSATION FLOW
═══════════════════════════════════════════════════════════════════════════════

STEP 1: User gives initial idea
→ Acknowledge it, identify what's missing from the checklist, ask 2-3 targeted questions

STEP 2: Gather missing info
→ Keep asking until you have all 5 required items
→ Guide them with examples and suggestions

STEP 3: Confirm with summary
→ Show them exactly what you'll generate:

"Here's what I'll create:

🎵 **Genre**: [specific subgenre]
⏱️ **Tempo**: [X] BPM  
🎨 **Vibe**: [mood words] + [texture words]
🎹 **Stems**: [instrument list]
🚫 **Avoiding**: [if any]

Ready to generate? Say 'yes' or 'go' to proceed!"

STEP 4: Generate ONLY after confirmation
→ Build a rich, detailed style prompt using all gathered info

═══════════════════════════════════════════════════════════════════════════════
HOW TO ASK GOOD QUESTIONS
═══════════════════════════════════════════════════════════════════════════════

For GENRE, offer specific options:
"When you say 'electronic', are you thinking more like:
- Deep house (groovy, 120-125 BPM)
- Techno (driving, 130-140 BPM)  
- Ambient (atmospheric, slower)
- Drum and bass (fast, 170+ BPM)
- Something else?"

For TEMPO, give genre-appropriate suggestions:
"For indie rock, typical tempos are 110-130 BPM. Do you want it on the faster, more energetic side, or slower and more relaxed?"

For MOOD, offer contrasting pairs:
"Should this feel more:
- Euphoric and uplifting, or melancholic and introspective?
- Aggressive and intense, or smooth and laid-back?
- Dark and moody, or bright and airy?"

For TEXTURE, give production examples:
"Production-wise, are you after:
- Polished and clean (modern pop sound)
- Raw and gritty (garage/lo-fi vibe)
- Atmospheric and spacious (lots of reverb)
- Tight and punchy (in-your-face)"

═══════════════════════════════════════════════════════════════════════════════
BUILDING THE STYLE PROMPT
═══════════════════════════════════════════════════════════════════════════════

When you call generateStems, combine all gathered info into a rich prompt:

EXAMPLE of a GOOD style prompt:
"Deep house, 122 BPM, euphoric and groovy mood, polished modern production, driving four-on-the-floor beat, warm bassline, lush atmospheric pads"

EXAMPLE of a BAD style prompt:
"house music"

═══════════════════════════════════════════════════════════════════════════════
BUILDING HIGH-QUALITY STYLE PROMPTS
═══════════════════════════════════════════════════════════════════════════════

Your style prompt should be 20-40 words and include:

1. **Specific genre** - "deep house", "indie rock", "lo-fi hip hop"
2. **BPM** - "122 BPM" (always include the number)
3. **2-3 mood words** - "euphoric", "melancholic", "aggressive", "dreamy"
4. **1-2 texture words** - "punchy", "lush", "gritty", "polished", "warm"
5. **Production quality phrase** - ALWAYS end with "professional studio quality, high fidelity"

EXAMPLE of an excellent style prompt:
"Deep house, 122 BPM, euphoric and groovy mood, warm and punchy production, driving rhythms, lush atmospheric elements, professional studio quality, high fidelity"

EXAMPLE of another excellent style prompt:
"Lo-fi hip hop, 85 BPM, chill and nostalgic vibe, dusty warm texture, relaxed groove, mellow and hazy, professional studio quality, high fidelity"

The quality phrase at the end significantly improves output!

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE INSTRUMENTS (with quality descriptors)
═══════════════════════════════════════════════════════════════════════════════

When generating, each instrument gets these quality enhancements automatically:

- **drums**: Punchy kick, tight snare, crisp hi-hats, solid groove
- **bass**: Deep warm tone, punchy low end, clean definition
- **melody**: Expressive melodic line, memorable hook, emotionally captivating
- **guitar**: Rich warm tone, clear articulate playing
- **keys/piano**: Warm resonant tone, lush chords, expressive dynamics
- **synth**: Lush warm sound, rich electronic textures, atmospheric
- **strings**: Lush orchestral arrangement, emotive and cinematic
- **pad**: Atmospheric texture, warm enveloping ambient sound

Maximum 5 instruments. For most genres, suggest: drums, bass, + 2-3 melodic/harmonic elements.

═══════════════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

User: "make me some beats"

GOOD response:
"Let's make some beats! To get you something great, I need to know:

1. **What genre?** Hip-hop, electronic, rock, or something else? 
   (e.g., boom bap, trap, house, indie rock drums...)

2. **What tempo?** Fast and energetic, or slower groove?
   - Chill/lo-fi: 70-90 BPM
   - Hip-hop: 85-115 BPM
   - House/dance: 120-130 BPM
   - High energy: 140+ BPM

3. **What vibe?** Hard-hitting and aggressive? Smooth and laid-back? Dark and moody?

What are you feeling?"

---

User: "I want something like Tame Impala"

GOOD response:
"Tame Impala vibes - nice choice! That's psychedelic rock/pop with dreamy, swirling production.

Let me dial this in:
- **Tempo**: Their stuff ranges from slow grooves (90 BPM) to more upbeat (110-120 BPM). Which end?
- **Energy**: More hypnotic and trippy, or more driving and rhythmic?
- **Stems**: I'm thinking drums, bass, synth, and maybe guitar or keys. Sound right?

Also - more "Currents" polished electronic feel, or "Lonerism" fuzzier psych rock?"

═══════════════════════════════════════════════════════════════════════════════

Remember: Quality in = quality out. The more specific info you gather, the better the stems."""


@tool
def generateStems(
    style: str,
    tempo: int = 120,
    instruments: List[str] = ["melody", "drums", "bass", "keys"]
) -> str:
    """Generate audio stems for each instrument using Stable Audio.
    
    ⚠️ ONLY CALL AFTER GATHERING ALL REQUIRED INFO AND USER CONFIRMS ⚠️
    
    REQUIRED before calling:
    1. Specific genre/subgenre (not just "rock" - need "indie rock", "post-punk", etc.)
    2. Tempo as BPM number
    3. At least 2-3 mood/energy words
    4. Texture/production style
    5. User confirmation to proceed
    
    Args:
        style: DETAILED prompt for the audio model. MUST include ALL of these:
               - Specific subgenre (e.g., "deep house" not "electronic")
               - BPM in the text (e.g., "122 BPM")
               - 2-3 mood words (euphoric, dark, dreamy, aggressive, etc.)
               - 1-2 texture words (punchy, lush, gritty, polished, warm, etc.)
               - MUST END WITH: "professional studio quality, high fidelity"
               
               EXCELLENT EXAMPLE: 
               "Deep house, 122 BPM, euphoric groovy mood, warm punchy production, 
                driving rhythms, lush atmospheric, professional studio quality, high fidelity"
               
               BAD EXAMPLES: 
               - "house music" (too vague)
               - "electronic beats" (no mood/texture)
               - Missing the quality phrase at the end
               
               Minimum 20 words for quality output.
               
        tempo: BPM as integer (40-200). Must match what you put in the style prompt.
        
        instruments: Stems to generate. Options: melody, drums, bass, guitar, keys, 
                    piano, synth, strings, pad. Maximum 5 instruments.
    
    Returns:
        JSON with generation status and parameters
    """
    # This will be intercepted - actual execution happens on frontend
    # Validate instruments
    valid_instruments = {"melody", "lead", "drums", "bass", "guitar", "keys", "piano", "synth", "strings", "pad"}
    filtered_instruments = [i for i in instruments if i.lower() in valid_instruments][:5]
    
    if not filtered_instruments:
        filtered_instruments = ["melody", "drums", "bass", "keys"]
    
    return f'{{"status": "pending_frontend_execution", "style": "{style}", "tempo": {tempo}, "instruments": {filtered_instruments}}}'


# All available tools
TOOLS = [generateStems]


def create_stem_agent():
    """Create the conversational stem agent with interrupt_before for tool execution."""
    agent = create_react_agent(
        model=model,
        tools=TOOLS,
        checkpointer=checkpointer,
        interrupt_before=["tools"],  # Pause before executing tools
        prompt=STEM_AGENT_SYSTEM_PROMPT,
    )
    return agent


# Singleton agent instance
_stem_agent = None


def get_stem_agent():
    """Get or create the singleton stem agent instance."""
    global _stem_agent
    if _stem_agent is None:
        _stem_agent = create_stem_agent()
    return _stem_agent


def generate_thread_id() -> str:
    """Generate a new thread ID for a session."""
    return str(uuid.uuid4())


async def start_stem_agent_step(
    prompt: str,
    thread_id: Optional[str] = None,
    context: Optional[str] = None
) -> dict:
    """Start a new stem agent interaction or continue an existing one.
    
    Args:
        prompt: The user's message
        thread_id: Optional existing thread ID to continue. If None, creates new session.
        context: Optional context to prepend to the prompt.
    
    Returns:
        dict with:
        - thread_id: Session identifier for continuation
        - tool_calls: List of tool calls to execute (if paused at interrupt)
        - done: True if agent completed without needing tool execution
        - message: Agent's response message (if done)
    """
    agent = get_stem_agent()
    
    # Create or reuse thread ID
    if thread_id is None:
        thread_id = generate_thread_id()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Load existing conversation history from checkpoint
    existing_state = await agent.aget_state(config)
    existing_messages = existing_state.values.get("messages", []) if existing_state.values else []
    
    print(f"\n{'='*60}")
    print(f"[STEM_AGENT] thread_id: {thread_id}")
    print(f"[STEM_AGENT] existing_messages count: {len(existing_messages)}")
    print(f"{'='*60}\n")
    
    # Build the full message with context if provided
    full_prompt = prompt
    if context:
        full_prompt = f"{context}\n\n---\n\nUser request: {prompt}"
    
    # Build message list
    new_message = {"role": "user", "content": full_prompt}
    if existing_messages:
        messages_to_send = {"messages": existing_messages + [new_message]}
        print(f"[STEM_AGENT] CONTINUING conversation with {len(existing_messages)} + 1 messages")
    else:
        messages_to_send = {"messages": [new_message]}
        print(f"[STEM_AGENT] STARTING new conversation")
    
    # Invoke the agent
    result = await agent.ainvoke(
        messages_to_send,
        config=config,
    )
    
    # Check if we're paused at interrupt (tool calls pending)
    state = await agent.aget_state(config)
    
    if state.next:  # There are pending nodes (we hit interrupt)
        # Extract tool calls from the last AI message
        last_message = result["messages"][-1]
        tool_calls = []
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tc in last_message.tool_calls:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": tc["args"],
                })
        
        return {
            "thread_id": thread_id,
            "tool_calls": tool_calls,
            "done": False,
            "message": None,
        }
    else:
        # Agent completed (still gathering info or waiting for confirmation)
        last_message = result["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        
        return {
            "thread_id": thread_id,
            "tool_calls": [],
            "done": True,
            "message": content,
        }


async def resume_stem_agent_step(thread_id: str, tool_results: list[dict]) -> dict:
    """Resume stem agent after tool execution.
    
    Args:
        thread_id: Session identifier from start_stem_agent_step
        tool_results: List of tool results, each with:
            - id: Tool call ID from the original tool_calls
            - result: JSON string result from execution
    
    Returns:
        Same format as start_stem_agent_step
    """
    agent = get_stem_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    from langchain_core.messages import ToolMessage
    
    tool_messages = []
    for tr in tool_results:
        tool_messages.append(
            ToolMessage(
                content=tr["result"],
                tool_call_id=tr["id"],
            )
        )
    
    # Resume the agent with tool results
    result = await agent.ainvoke(
        Command(resume=tool_messages),
        config=config,
    )
    
    # Check state again
    state = await agent.aget_state(config)
    
    if state.next:  # More tool calls
        last_message = result["messages"][-1]
        tool_calls = []
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tc in last_message.tool_calls:
                tool_calls.append({
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": tc["args"],
                })
        
        return {
            "thread_id": thread_id,
            "tool_calls": tool_calls,
            "done": False,
            "message": None,
        }
    else:
        last_message = result["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        
        return {
            "thread_id": thread_id,
            "tool_calls": [],
            "done": True,
            "message": content,
        }


async def stream_stem_agent_step(
    prompt: str,
    thread_id: Optional[str] = None,
    context: Optional[str] = None
):
    """Stream stem agent events as SSE.
    
    Yields events:
        - thinking: Agent reasoning/processing (streamed tokens)
        - tool_calls: Tools to execute (generateStems with parameters)
        - message: Final response from agent
        - error: Any errors that occurred
    
    Args:
        prompt: The user's message
        thread_id: Optional existing thread ID to continue
        context: Optional context
    
    Yields:
        dict with 'type' and event-specific data
    """
    agent = get_stem_agent()
    
    is_new_thread = thread_id is None
    if thread_id is None:
        thread_id = generate_thread_id()
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Load existing conversation history
    existing_state = await agent.aget_state(config)
    existing_messages = existing_state.values.get("messages", []) if existing_state.values else []
    print(f"[DEBUG] Stem agent thread {thread_id[:8]}... is_new={is_new_thread}, existing_messages={len(existing_messages)}")
    
    # Build the full message
    full_prompt = prompt
    if context:
        full_prompt = f"{context}\n\n---\n\nUser request: {prompt}"
    
    new_message = {"role": "user", "content": full_prompt}
    if existing_messages:
        messages_to_send = {"messages": existing_messages + [new_message]}
    else:
        messages_to_send = {"messages": [new_message]}
    
    try:
        yield {"type": "thinking", "thread_id": thread_id, "content": ""}
        
        seen_tokens = set()
        
        # Stream events from the agent
        async for event in agent.astream_events(
            messages_to_send,
            config=config,
            version="v2",
        ):
            event_type = event.get("event")
            run_id = event.get("run_id", "")
            
            # Handle LLM streaming tokens
            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token_key = f"{run_id}:{chunk.content}"
                    if token_key not in seen_tokens:
                        seen_tokens.add(token_key)
                        yield {"type": "thinking", "thread_id": thread_id, "content": chunk.content}
        
        # After streaming, check state for tool calls or completion
        state = await agent.aget_state(config)
        
        if state.next:
            # Agent paused at interrupt - extract tool calls
            messages = state.values.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_calls = []
                    for tc in last_msg.tool_calls:
                        tool_calls.append({
                            "id": tc["id"],
                            "name": tc["name"],
                            "args": tc["args"],
                        })
                    yield {
                        "type": "tool_calls",
                        "thread_id": thread_id,
                        "tool_calls": tool_calls,
                        "done": False,
                    }
                    return
        
        # Agent completed - get final message
        messages = state.values.get("messages", [])
        if messages:
            last_message = messages[-1]
            content = last_message.content if hasattr(last_message, "content") else str(last_message)
            yield {
                "type": "message",
                "thread_id": thread_id,
                "content": content,
                "done": True,
            }
    
    except Exception as e:
        yield {"type": "error", "thread_id": thread_id, "error": str(e)}


async def stream_stem_agent_resume(thread_id: str, tool_results: list[dict]):
    """Stream stem agent events after tool results are received.
    
    Yields events similar to stream_stem_agent_step.
    """
    from langchain_core.messages import ToolMessage
    
    agent = get_stem_agent()
    config = {"configurable": {"thread_id": thread_id}}
    
    yield {
        "type": "tool_results_received",
        "thread_id": thread_id,
        "count": len(tool_results),
    }
    
    # Build tool messages
    tool_messages = []
    for tr in tool_results:
        tool_messages.append(
            ToolMessage(
                content=tr["result"],
                tool_call_id=tr["id"],
            )
        )
    
    try:
        yield {"type": "thinking", "thread_id": thread_id, "content": "Processing generation results..."}
        
        seen_tokens = set()
        
        async for event in agent.astream_events(
            Command(resume=tool_messages),
            config=config,
            version="v2",
        ):
            event_type = event.get("event")
            run_id = event.get("run_id", "")
            
            if event_type == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token_key = f"{run_id}:{chunk.content}"
                    if token_key not in seen_tokens:
                        seen_tokens.add(token_key)
                        yield {"type": "thinking", "thread_id": thread_id, "content": chunk.content}
        
        state = await agent.aget_state(config)
        
        if state.next:
            messages = state.values.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_calls = []
                    for tc in last_msg.tool_calls:
                        tool_calls.append({
                            "id": tc["id"],
                            "name": tc["name"],
                            "args": tc["args"],
                        })
                    yield {
                        "type": "tool_calls",
                        "thread_id": thread_id,
                        "tool_calls": tool_calls,
                        "done": False,
                    }
                    return
        
        messages = state.values.get("messages", [])
        if messages:
            last_message = messages[-1]
            content = last_message.content if hasattr(last_message, "content") else str(last_message)
            yield {
                "type": "message",
                "thread_id": thread_id,
                "content": content,
                "done": True,
            }
    
    except Exception as e:
        yield {"type": "error", "thread_id": thread_id, "error": str(e)}
