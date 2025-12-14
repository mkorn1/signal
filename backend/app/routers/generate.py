import re
import json
import asyncio
import base64
from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from app.models.schemas import (
    GenerateRequest,
    GenerateResponse,
    RegenerateRequest,
    RegenerateResponse,
    TrackData,
    SongMetadata,
    AttemptLog,
    AgentStepRequest,
    AgentStepResponse,
    ToolCall,
    CohesionSpec,
    StrengthProfile,
    CohesionValidationResult,
)
from app.services.composition_agent import (
    generate_midi_code as generate_midi_code_composition,
    generate_single_track_code as generate_single_track_code_composition,
    stream_composition_process,
)
from app.services.llm import (
    generate_midi_code as generate_midi_code_llm,
    generate_single_track_code as generate_single_track_code_llm,
)
from app.services.midi_executor import (
    execute_midi_generation,
    midi_to_base64,
    MIDIExecutionError,
)
from app.services.generator import generate_song_deep, GenerationError
from app.services.hybrid_agent import (
    start_agent_step,
    resume_agent_step,
    stream_agent_step,
    stream_agent_resume,
)
from app.services.per_instrument_agent import generate_per_instrument
from app.services.stem_separation_agent import generate_with_stem_separation
from app.services.conversational_stem_agent import (
    start_stem_agent_step,
    resume_stem_agent_step,
    stream_stem_agent_step,
    stream_stem_agent_resume,
)

router = APIRouter()

# Default channel/program mappings for common instruments
# The LLM will set program numbers in the MIDI files, but we use these
# as fallbacks and for channel assignment
DEFAULT_TRACK_CONFIG = {
    "drums": {"channel": 9, "program": 0},
    "bass": {"channel": 0, "program": 33},
    "guitar": {"channel": 1, "program": 25},
    "keys": {"channel": 2, "program": 4},
    "piano": {"channel": 2, "program": 0},
    "melody": {"channel": 3, "program": 73},
    "synth": {"channel": 4, "program": 81},
    "strings": {"channel": 5, "program": 48},
    "pad": {"channel": 6, "program": 89},
    "lead": {"channel": 3, "program": 80},
    "arp": {"channel": 7, "program": 84},
    "organ": {"channel": 2, "program": 16},
    "brass": {"channel": 5, "program": 61},
    "flute": {"channel": 3, "program": 73},
    "sax": {"channel": 4, "program": 66},
}


def get_track_config(track_name: str, index: int) -> dict:
    """Get channel/program for a track, with fallback for unknown instruments."""
    name_lower = track_name.lower()

    # Check for exact or partial matches
    for key, config in DEFAULT_TRACK_CONFIG.items():
        if key in name_lower:
            return config

    # Fallback: assign channel based on index (avoid channel 9 for non-drums)
    channel = index if index < 9 else index + 1
    return {"channel": min(channel, 15), "program": 0}

MAX_RETRIES = 2
MAX_TRACKS = 8


def parse_prompt(prompt: str) -> dict:
    """Extract tempo, key, and style from prompt."""
    # Simple parsing - can be enhanced later
    tempo = 120
    key = "Am"

    prompt_lower = prompt.lower()

    # Try to extract BPM
    bpm_match = re.search(r"(\d+)\s*bpm", prompt_lower)
    if bpm_match:
        tempo = int(bpm_match.group(1))

    # Try to extract key
    key_match = re.search(r"\b([A-G][#b]?m?)\b", prompt)
    if key_match:
        key = key_match.group(1)

    return {"tempo": tempo, "key": key, "style": prompt}


@router.post("/generate", response_model=GenerateResponse)
async def generate_song(request: GenerateRequest):
    """Generate a multi-track song from a text prompt."""

    params = parse_prompt(request.prompt)
    agent_type = request.agent_type or "composition_agent"
    last_error = None

    # Handle audio-based agents
    if agent_type == "per_instrument":
        try:
            audio_files = await generate_per_instrument(
                prompt=params["style"], tempo=params["tempo"], key=params["key"]
            )
            
            if len(audio_files) > MAX_TRACKS:
                audio_files = dict(list(audio_files.items())[:MAX_TRACKS])
            
            if len(audio_files) == 0:
                raise HTTPException(status_code=422, detail="No tracks were generated")
            
            tracks = []
            for index, (name, audio_bytes) in enumerate(audio_files.items()):
                config = get_track_config(name, index)
                tracks.append(
                    TrackData(
                        name=name,
                        audio_data=base64.b64encode(audio_bytes).decode("utf-8"),
                        channel=config["channel"],
                        program_number=config["program"],
                        data_type="audio",
                    )
                )
            
            return GenerateResponse(
                tracks=tracks,
                metadata=SongMetadata(
                    tempo=params["tempo"], key=params["key"], time_signature="4/4"
                ),
                message=f"Generated {len(tracks)} track(s)",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    elif agent_type == "stem_separation":
        try:
            audio_files = await generate_with_stem_separation(
                prompt=params["style"], tempo=params["tempo"], key=params["key"]
            )
            
            if len(audio_files) > MAX_TRACKS:
                audio_files = dict(list(audio_files.items())[:MAX_TRACKS])
            
            if len(audio_files) == 0:
                raise HTTPException(status_code=422, detail="No tracks were generated")
            
            tracks = []
            for index, (name, audio_bytes) in enumerate(audio_files.items()):
                config = get_track_config(name, index)
                tracks.append(
                    TrackData(
                        name=name,
                        audio_data=base64.b64encode(audio_bytes).decode("utf-8"),
                        channel=config["channel"],
                        program_number=config["program"],
                        data_type="audio",
                    )
                )
            
            return GenerateResponse(
                tracks=tracks,
                metadata=SongMetadata(
                    tempo=params["tempo"], key=params["key"], time_signature="4/4"
                ),
                message=f"Generated {len(tracks)} track(s)",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # MIDI-based agents (existing logic)
    # Select the appropriate generation function based on agent_type
    if agent_type == "llm":
        generate_func = generate_midi_code_llm
    else:
        generate_func = generate_midi_code_composition

    # Retry loop for transient generation errors
    for attempt in range(MAX_RETRIES):
        try:
            # Generate code from Claude
            code = await generate_func(
                prompt=params["style"], tempo=params["tempo"], key=params["key"]
            )

            # Execute the code
            midi_files = execute_midi_generation(
                code=code, tempo=params["tempo"], key=params["key"]
            )

            # Validate track count
            if len(midi_files) > MAX_TRACKS:
                # Take only the first MAX_TRACKS files
                midi_files = dict(list(midi_files.items())[:MAX_TRACKS])

            if len(midi_files) == 0:
                raise HTTPException(status_code=422, detail="No tracks were generated")

            # Convert to response format
            tracks = []
            for index, (name, midi_bytes) in enumerate(midi_files.items()):
                config = get_track_config(name, index)
                tracks.append(
                    TrackData(
                        name=name,
                        midi_data=midi_to_base64(midi_bytes),
                        channel=config["channel"],
                        program_number=config["program"],
                        data_type="midi",
                    )
                )

            return GenerateResponse(
                tracks=tracks,
                metadata=SongMetadata(
                    tempo=params["tempo"], key=params["key"], time_signature="4/4"
                ),
                message=f"Generated {len(tracks)} track(s)",
            )

        except MIDIExecutionError as e:
            last_error = e
            error_msg = str(e).lower()
            # Retry on transient generation errors
            if "overlapping notes" in error_msg or "syntax error" in error_msg:
                continue
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # If we exhausted retries
    raise HTTPException(
        status_code=422,
        detail=f"Generation failed after {MAX_RETRIES} attempts: {str(last_error)}",
    )


async def stream_llm_process(prompt: str, tempo: int, key: str):
    """Generate code silently without streaming - LLM Direct mode doesn't stream."""
    code = await generate_midi_code_llm(prompt=prompt, tempo=tempo, key=key)
    # Send code_ready signal directly without streaming the code
    yield json.dumps({
        "type": "code_ready",
        "code": code
    }) + "\n"


@router.post("/generate/stream")
async def generate_song_stream(request: GenerateRequest):
    """Stream the composition process with real-time updates."""
    params = parse_prompt(request.prompt)
    agent_type = request.agent_type or "composition_agent"
    
    async def generate():
        try:
            code = None
            # Select streaming function based on agent_type
            if agent_type == "llm":
                stream_func = stream_llm_process
            else:
                stream_func = stream_composition_process
            
            async for chunk in stream_func(
                prompt=params["style"],
                tempo=params["tempo"],
                key=params["key"]
            ):
                # Parse the chunk
                try:
                    data = json.loads(chunk.strip())
                    if data.get("type") == "code_ready":
                        code = data.get("code")
                        # Send final message
                        yield f"data: {json.dumps({'type': 'code_ready'})}\n\n"
                    else:
                        # Stream message content
                        yield f"data: {chunk}"
                except json.JSONDecodeError:
                    # If it's not JSON, send as-is
                    yield f"data: {chunk}"
            
            # After streaming, execute the code if we have it
            if code:
                try:
                    midi_files = execute_midi_generation(
                        code=code,
                        tempo=params["tempo"],
                        key=params["key"]
                    )
                    
                    # Validate track count
                    if len(midi_files) > MAX_TRACKS:
                        midi_files = dict(list(midi_files.items())[:MAX_TRACKS])
                    
                    if len(midi_files) == 0:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'No tracks were generated'})}\n\n"
                        return
                    
                    # Convert to response format
                    tracks = []
                    for index, (name, midi_bytes) in enumerate(midi_files.items()):
                        config = get_track_config(name, index)
                        tracks.append({
                            "name": name,
                            "midi_data": midi_to_base64(midi_bytes),
                            "channel": config["channel"],
                            "program_number": config["program"],
                        })
                    
                    # Send final result
                    result_data = {
                        'type': 'complete',
                        'tracks': tracks,
                        'metadata': {
                            'tempo': params['tempo'],
                            'key': params['key'],
                            'time_signature': '4/4'
                        },
                        'message': f'Generated {len(tracks)} track(s)'
                    }
                    yield f"data: {json.dumps(result_data)}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No code was generated'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/regenerate", response_model=RegenerateResponse)
async def regenerate_track(request: RegenerateRequest):
    """Regenerate a single track based on instruction."""

    context = request.context
    last_error = None

    # For regenerate, default to composition_agent
    # Could add agent_type to RegenerateRequest if needed in the future
    generate_func = generate_single_track_code_composition

    for attempt in range(MAX_RETRIES):
        try:
            # Generate code for single track
            code = await generate_func(
                track_name=request.track_name,
                instruction=request.instruction,
                context=context,
            )

            # Execute the code
            midi_files = execute_midi_generation(
                code=code,
                tempo=context.get("tempo", 120),
                key=context.get("key", "Am"),
            )

            # Get the generated track
            track_name = request.track_name.lower()
            if track_name not in midi_files:
                # Try to find any generated file
                if midi_files:
                    track_name = list(midi_files.keys())[0]
                else:
                    raise MIDIExecutionError("No MIDI file was generated")

            midi_bytes = midi_files[track_name]
            config = get_track_config(track_name, 0)

            return RegenerateResponse(
                track=TrackData(
                    name=request.track_name,
                    midi_data=midi_to_base64(midi_bytes),
                    channel=config["channel"],
                    program_number=config["program"],
                ),
                message=f"Regenerated {request.track_name} track",
            )

        except MIDIExecutionError as e:
            last_error = e
            error_msg = str(e).lower()
            # Retry on transient generation errors
            if "overlapping notes" in error_msg or "syntax error" in error_msg:
                continue
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Regeneration failed: {str(e)}"
            )

    raise HTTPException(
        status_code=422,
        detail=f"Regeneration failed after {MAX_RETRIES} attempts: {str(last_error)}",
    )


# ============================================================================
# Deep Agent Architecture Endpoints
# ============================================================================


class DeepGenerateResponse(BaseModel):
    """Response from deep generation endpoint."""

    tracks: list[TrackData]
    metadata: SongMetadata
    message: str
    attempt_logs: list[AttemptLog]
    spec_used: Optional[dict] = None  # The SongSpec used (for debugging)


@router.post("/generate/deep", response_model=DeepGenerateResponse)
async def generate_song_deep_endpoint(request: GenerateRequest):
    """
    Generate a multi-track song using the deep agent architecture.

    This endpoint uses:
    1. Planning stage to create a structured song specification
    2. Spec-driven code generation
    3. MIDI quality validation
    4. Iterative refinement (up to 5 attempts)

    Returns detailed attempt logs for transparency.
    """
    params = parse_prompt(request.prompt)

    try:
        result = await generate_song_deep(
            prompt=params["style"],
            tempo=params["tempo"],
            key=params["key"],
        )

        # Convert MIDI files to response format
        midi_files = result["midi_files"]
        spec = result["spec"]
        attempt_logs = result["attempt_logs"]

        tracks = []
        for index, (name, midi_bytes) in enumerate(midi_files.items()):
            config = get_track_config(name, index)
            tracks.append(
                TrackData(
                    name=name,
                    midi_data=midi_to_base64(midi_bytes),
                    channel=config["channel"],
                    program_number=config["program"],
                )
            )

        return DeepGenerateResponse(
            tracks=tracks,
            metadata=SongMetadata(
                tempo=spec.tempo, key=spec.key, time_signature=spec.time_signature
            ),
            message=f"Generated {len(tracks)} track(s) after {len(attempt_logs)} attempt(s)",
            attempt_logs=attempt_logs,
            spec_used=spec.model_dump(),
        )

    except GenerationError as e:
        # Return detailed failure information
        raise HTTPException(
            status_code=422,
            detail={
                "error": str(e),
                "attempt_logs": [log.model_dump() for log in e.attempt_logs],
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/generate/stream")
async def generate_song_stream(request: GenerateRequest, req: Request):
    """
    Generate a song with real-time progress updates via SSE.

    Returns Server-Sent Events with the following event types:
    - progress: Stage updates (planning, generating, validating, refining)
    - complete: Final result with tracks
    - error: Generation failed
    """
    params = parse_prompt(request.prompt)
    agent_type = request.agent_type or "composition_agent"

    async def event_generator():
        progress_queue: asyncio.Queue = asyncio.Queue()

        async def progress_callback(stage: str, data: dict):
            await progress_queue.put({"stage": stage, **data})

        # Start generation in background
        async def run_generation():
            try:
                # Handle audio-based agents
                if agent_type == "per_instrument" or agent_type == "stem_separation":
                    # For minimal implementation, use simple progress updates
                    await progress_queue.put({"stage": "generating", "message": "Generating audio tracks..."})
                    
                    if agent_type == "per_instrument":
                        audio_files = await generate_per_instrument(
                            prompt=params["style"], tempo=params["tempo"], key=params["key"]
                        )
                    else:
                        audio_files = await generate_with_stem_separation(
                            prompt=params["style"], tempo=params["tempo"], key=params["key"]
                        )
                    
                    if len(audio_files) > MAX_TRACKS:
                        audio_files = dict(list(audio_files.items())[:MAX_TRACKS])
                    
                    tracks = []
                    for index, (name, audio_bytes) in enumerate(audio_files.items()):
                        config = get_track_config(name, index)
                        tracks.append({
                            "name": name,
                            "audio_data": base64.b64encode(audio_bytes).decode("utf-8"),
                            "channel": config["channel"],
                            "program_number": config["program"],
                            "data_type": "audio",
                        })
                    
                    await progress_queue.put({
                        "stage": "complete",
                        "result": {
                            "tracks": tracks,
                            "metadata": {
                                "tempo": params["tempo"],
                                "key": params["key"],
                                "time_signature": "4/4",
                            },
                            "message": f"Generated {len(tracks)} track(s)",
                        },
                        "attempt_logs": [],
                    })
                    return
                
                # MIDI-based agents (existing logic)
                result = await generate_song_deep(
                    prompt=params["style"],
                    tempo=params["tempo"],
                    key=params["key"],
                    progress_callback=progress_callback,
                )

                # Convert MIDI files to response format
                midi_files = result["midi_files"]
                spec = result["spec"]
                attempt_logs = result["attempt_logs"]

                tracks = []
                for index, (name, midi_bytes) in enumerate(midi_files.items()):
                    config = get_track_config(name, index)
                    tracks.append({
                        "name": name,
                        "midi_data": midi_to_base64(midi_bytes),
                        "channel": config["channel"],
                        "program_number": config["program"],
                    })

                await progress_queue.put(
                    {
                        "stage": "complete",
                        "result": {
                            "tracks": tracks,
                            "metadata": {
                                "tempo": spec.tempo,
                                "key": spec.key,
                                "time_signature": spec.time_signature,
                            },
                            "message": f"Generated {len(tracks)} track(s)",
                        },
                        "attempt_logs": [log.model_dump() for log in attempt_logs],
                    }
                )
            except GenerationError as e:
                await progress_queue.put(
                    {
                        "stage": "error",
                        "error": str(e),
                        "attempt_logs": [log.model_dump() for log in e.attempt_logs],
                    }
                )
            except Exception as e:
                await progress_queue.put(
                    {
                        "stage": "error",
                        "error": str(e),
                    }
                )
            finally:
                await progress_queue.put(None)  # Signal end

        # Start background task
        task = asyncio.create_task(run_generation())

        try:
            while True:
                # Check if client disconnected
                if await req.is_disconnected():
                    task.cancel()
                    break

                try:
                    event = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield ": keepalive\n\n"
                    continue

                if event is None:
                    break

                yield f"data: {json.dumps(event)}\n\n"

        except asyncio.CancelledError:
            task.cancel()
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Hybrid Agent Endpoint (frontend tool execution)
# ============================================================================


@router.post("/agent/step", response_model=AgentStepResponse)
async def agent_step(request: AgentStepRequest):
    """
    Execute a single step of the hybrid agent.

    This endpoint supports two modes:
    1. Start new conversation: Send { prompt: "..." }
    2. Resume after tool execution: Send { thread_id: "...", tool_results: [...] }

    When the agent needs to execute tools, it returns:
    - done: false
    - tool_calls: Array of tools to execute on frontend

    When the agent completes:
    - done: true
    - message: Final response from agent
    """
    try:
        if request.tool_results and request.thread_id:
            # Resume mode: continue after frontend tool execution
            result = await resume_agent_step(
                thread_id=request.thread_id,
                tool_results=[{"id": tr.id, "result": tr.result} for tr in request.tool_results],
            )
        elif request.prompt:
            # Start mode: new conversation
            result = await start_agent_step(
                prompt=request.prompt,
                thread_id=request.thread_id,
                context=request.context,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'prompt' (to start) or 'tool_results' (to resume)",
            )

        return AgentStepResponse(
            thread_id=result["thread_id"],
            tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
            done=result["done"],
            message=result["message"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent step failed: {str(e)}")


@router.post("/agent/step/stream")
async def agent_step_stream(request: AgentStepRequest, req: Request):
    """
    Execute a hybrid agent step with real-time SSE streaming.

    This endpoint supports two modes:
    1. Start new conversation: Send { prompt: "..." }
    2. Resume after tool execution: Send { thread_id: "...", tool_results: [...] }

    Returns Server-Sent Events with the following event types:
    - thinking: Agent reasoning/processing (streamed tokens)
    - tool_calls: Tools to execute on frontend (pauses stream, wait for resume)
    - tool_results_received: Acknowledgment after frontend sends tool results
    - message: Final response from agent
    - error: Any errors that occurred
    """

    async def event_generator():
        try:
            if request.tool_results and request.thread_id:
                # Resume mode: continue after frontend tool execution
                async for event in stream_agent_resume(
                    thread_id=request.thread_id,
                    tool_results=[{"id": tr.id, "result": tr.result} for tr in request.tool_results],
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            elif request.prompt:
                # Start mode: new conversation
                async for event in stream_agent_step(
                    prompt=request.prompt,
                    thread_id=request.thread_id,
                    context=request.context,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Must provide either prompt (to start) or tool_results (to resume)'})}\n\n"

        except asyncio.CancelledError:
            # Client disconnected
            raise
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Conversational Stem Agent Endpoint (chat-based audio generation)
# ============================================================================


@router.post("/agent/stem/step", response_model=AgentStepResponse)
async def stem_agent_step(request: AgentStepRequest):
    """
    Execute a single step of the conversational stem agent.
    
    This agent has a conversation to gather user preferences before generating
    audio stems. It will ask questions about style, tempo, key, instruments,
    and references before triggering generation.
    
    Supports two modes:
    1. Start new conversation: Send { prompt: "..." }
    2. Resume after tool execution: Send { thread_id: "...", tool_results: [...] }
    
    When the agent needs to execute tools (generateStems), it returns:
    - done: false
    - tool_calls: Array with generateStems tool and parameters
    
    When the agent is still gathering info:
    - done: true
    - message: Agent's question or confirmation request
    """
    try:
        if request.tool_results and request.thread_id:
            # Resume mode: continue after stem generation
            result = await resume_stem_agent_step(
                thread_id=request.thread_id,
                tool_results=[{"id": tr.id, "result": tr.result} for tr in request.tool_results],
            )
        elif request.prompt:
            # Start mode: new conversation or continue existing
            result = await start_stem_agent_step(
                prompt=request.prompt,
                thread_id=request.thread_id,
                context=request.context,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'prompt' (to start) or 'tool_results' (to resume)",
            )
        
        return AgentStepResponse(
            thread_id=result["thread_id"],
            tool_calls=[ToolCall(**tc) for tc in result["tool_calls"]],
            done=result["done"],
            message=result["message"],
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stem agent step failed: {str(e)}")


@router.post("/agent/stem/step/stream")
async def stem_agent_step_stream(request: AgentStepRequest, req: Request):
    """
    Execute a conversational stem agent step with real-time SSE streaming.
    
    This agent has a conversation to gather user preferences before generating
    audio stems. It streams its thinking process and questions in real-time.
    
    Supports two modes:
    1. Start new conversation: Send { prompt: "..." }
    2. Resume after tool execution: Send { thread_id: "...", tool_results: [...] }
    
    Returns Server-Sent Events with the following event types:
    - thinking: Agent reasoning/questions (streamed tokens)
    - tool_calls: generateStems tool to execute with gathered parameters
    - tool_results_received: Acknowledgment after frontend sends generation results
    - message: Final response from agent
    - error: Any errors that occurred
    """
    
    async def event_generator():
        try:
            if request.tool_results and request.thread_id:
                # Resume mode: continue after stem generation completed
                async for event in stream_stem_agent_resume(
                    thread_id=request.thread_id,
                    tool_results=[{"id": tr.id, "result": tr.result} for tr in request.tool_results],
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            elif request.prompt:
                # Start mode: new conversation or continue existing
                async for event in stream_stem_agent_step(
                    prompt=request.prompt,
                    thread_id=request.thread_id,
                    context=request.context,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'error': 'Must provide either prompt (to start) or tool_results (to resume)'})}\n\n"
        
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Audio-to-Audio One-Shot Endpoint (mic input + text prompt -> generated audio)
# ============================================================================


class AudioToAudioRequest(BaseModel):
    """Request for audio-to-audio generation."""
    prompt: str  # Text description of what to generate (e.g., "drum beat", "synthwave lead")
    audio_data: str  # Base64 encoded WAV audio from microphone
    duration: int = 20  # Output duration in seconds (1-190)
    strength: float = 0.25  # How much to transform (0=keep original, 1=ignore original)
    cfg_scale: float = 12.0  # Prompt adherence (1-25, recommended 7-15)
    steps: int = 80  # Denoising steps for quality (30-100, recommended 50-80)
    seed: Optional[int] = None  # Optional seed for reproducibility
    
    # Cohesion settings (for constraining to existing tracks)
    target_tempo: Optional[int] = None  # Target BPM - enables pre/post processing
    style: Optional[str] = None  # Style context to add to prompt (e.g., "dark synthwave")
    mood: Optional[str] = None  # Mood keywords (e.g., "moody, atmospheric")
    
    # Processing options (enabled when target_tempo is set)
    preprocess_input: bool = True  # Time-stretch user recording to target tempo
    postprocess_output: bool = True  # Time-stretch output & trim to exact duration
    max_bpm_drift: int = 10  # Max BPM drift before post-processing corrects it


class CohesionGenerateRequest(BaseModel):
    """Request for generating a track with cohesion constraints."""
    instrument: str  # Instrument to generate (e.g., "bass", "drums", "synth")
    prompt: str  # Style/mood description
    reference_audio: str  # Base64 encoded reference track (anchor)
    
    # Cohesion constraints
    style: str  # Locked style for consistency (e.g., "dark synthwave")
    tempo: int = 120  # Target BPM
    duration: int = 20  # Track duration
    strength_profile: str = "medium"  # "tight", "medium", "loose"
    mood: Optional[str] = None  # Additional mood keywords
    
    # Validation
    validate_bpm: bool = True  # Validate output matches tempo
    max_bpm_drift: int = 10  # Max allowed BPM difference
    max_retries: int = 2  # Retry on validation failure


class CohesionGenerateResponse(BaseModel):
    """Response from cohesion-constrained generation."""
    audio_data: str  # Base64 encoded WAV output
    name: str  # Track name
    duration: float
    validation: Optional[CohesionValidationResult] = None


async def build_audio_to_audio_prompt(user_prompt: str) -> str:
    """
    Use LLM to build an optimized prompt for Stable Audio's audio-to-audio model.
    
    Based on patterns from: https://stableaudio.com/user-guide/audio-to-audio
    """
    from app.services.llm import get_openrouter_client
    
    system_prompt = """You are a prompt engineer for Stable Audio's audio-to-audio model. Enhance user prompts following these EXACT patterns from the official Stable Audio guide.

REAL EXAMPLES FROM STABLE AUDIO GUIDE (https://stableaudio.com/user-guide/audio-to-audio):

Simple (these work great):
- "Drums"
- "Bass guitar"
- "Heavy metal guitar"
- "Upright bass"
- "Guitar"
- "Choir"
- "Racecar"
- "Racing car"

Structured format:
- "format: solo | instruments: vibraphone"
- "Genre: UK Bass | Instruments: 707 Drum Machine, Strings, 808 bass stabs, Beautiful Synths"
- "Instruments: Strings, Drum Kit, Electric Bass, Choir, String Section, Flute, Harp"

Descriptive (genre + instruments + mood words):
- "Post rock, guitars, bass, strings, euphoric, up-lifting, moody, flowing, raw, epic"
- "Lofi hip hop beat, chillhop"
- "Electronic, orchestral, relaxed, synth, soft, piano, bass, 808 bass stabs"

MOOD WORDS THAT WORK WELL:
euphoric, up-lifting, moody, flowing, raw, epic, relaxed, soft, beautiful, chillhop

RULES:
1. Match the user's intent exactly - if they say "drums", output might just be "Drums" or "Drums, punchy, tight"
2. Simple prompts are often better - don't over-complicate
3. If adding descriptors, use mood words from the guide: euphoric, moody, raw, soft, relaxed, etc.
4. For structured output, use "format: X | instruments: Y" or "Genre: X | Instruments: Y"
5. NO quality suffixes like "high fidelity" or "studio quality" - the guide doesn't use these
6. Keep under 30 words unless listing multiple instruments
7. Works for anything: instruments, SFX, vocals, nature sounds

OUTPUT: Return ONLY the enhanced prompt, nothing else. No quotes."""

    try:
        client = get_openrouter_client()
        response = await client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Enhance this prompt: {user_prompt}"}
            ],
            max_tokens=100,
            temperature=0.3,
        )
        enhanced = response.choices[0].message.content.strip()
        # Remove any quotes the LLM might add
        enhanced = enhanced.strip('"\'')
        return enhanced
    except Exception as e:
        # Fallback to user prompt as-is if LLM fails (simple prompts work fine per the guide)
        return user_prompt


class AudioToAudioResponse(BaseModel):
    """Response from audio-to-audio generation."""
    audio_data: str  # Base64 encoded WAV output
    name: str  # Track name
    duration: float  # Duration in seconds
    
    # Processing metadata (when cohesion is enabled)
    source_bpm: Optional[float] = None  # Detected BPM of user input
    output_bpm: Optional[float] = None  # Detected BPM of generated audio
    was_input_stretched: bool = False  # Whether input was time-stretched
    was_output_stretched: bool = False  # Whether output was time-stretched


@router.post("/audio-to-audio/generate", response_model=AudioToAudioResponse)
async def audio_to_audio_generate(request: AudioToAudioRequest):
    """
    Audio-to-audio generation with optional cohesion processing.
    
    Takes a user recording (hum, beatbox, etc.) and transforms it into
    the requested instrument. When target_tempo is provided, pre-processes
    the input and post-processes the output to ensure it fits with existing tracks.
    
    **Basic mode** (no target_tempo):
    - User recording → Stable Audio → generated instrument
    
    **Cohesion mode** (target_tempo provided):
    - User recording → detect BPM → stretch to target → Stable Audio → 
      detect output BPM → stretch to target → trim to duration → output
    
    Args:
        prompt: What instrument to generate (e.g., "drums", "bass")
        audio_data: Base64 WAV of user recording
        duration: Output duration in seconds
        strength: Transformation amount (0=keep original, 1=ignore original)
        target_tempo: Target BPM (enables cohesion processing)
        style: Style context added to prompt
        mood: Mood keywords added to prompt
        preprocess_input: Stretch user recording to target tempo
        postprocess_output: Stretch output to target tempo & trim duration
    
    Returns:
        Generated audio with processing metadata
    """
    from app.services.audio_renderer import generate_audio_to_audio
    from app.services.audio_processing import (
        preprocess_user_recording,
        postprocess_generated_audio,
    )
    
    try:
        # Decode the base64 audio
        audio_bytes = base64.b64decode(request.audio_data)
        
        source_bpm = None
        was_input_stretched = False
        
        # PRE-PROCESSING: Stretch user input to target tempo
        if request.target_tempo and request.preprocess_input:
            audio_bytes, source_bpm = preprocess_user_recording(
                audio_bytes,
                target_bpm=request.target_tempo,
            )
            was_input_stretched = source_bpm is not None and abs(source_bpm - request.target_tempo) > 2
        
        # Build prompt with style/mood context
        prompt_parts = [request.prompt]
        if request.style:
            prompt_parts.append(request.style)
        if request.mood:
            prompt_parts.append(request.mood)
        if request.target_tempo:
            prompt_parts.append(f"{request.target_tempo} BPM")
        
        full_prompt = ", ".join(prompt_parts)
        enhanced_prompt = await build_audio_to_audio_prompt(full_prompt)
        
        # Generate using Stable Audio
        generated_audio = await generate_audio_to_audio(
            reference_audio=audio_bytes,
            prompt=enhanced_prompt,
            duration=request.duration,
            strength=request.strength,
            cfg_scale=request.cfg_scale,
            steps=request.steps,
            seed=request.seed,
        )
        
        output_bpm = None
        was_output_stretched = False
        
        # POST-PROCESSING: Stretch output to target tempo & trim
        if request.target_tempo and request.postprocess_output:
            generated_audio, metadata = postprocess_generated_audio(
                generated_audio,
                target_bpm=request.target_tempo,
                target_duration=float(request.duration),
                max_bpm_drift=request.max_bpm_drift,
            )
            output_bpm = metadata.get("detected_bpm")
            was_output_stretched = metadata.get("was_stretched", False)
        
        # Encode result
        audio_b64 = base64.b64encode(generated_audio).decode()
        
        name = request.prompt[:30].strip()
        if len(request.prompt) > 30:
            name += "..."
        
        return AudioToAudioResponse(
            audio_data=audio_b64,
            name=name,
            duration=request.duration,
            source_bpm=source_bpm,
            output_bpm=output_bpm,
            was_input_stretched=was_input_stretched,
            was_output_stretched=was_output_stretched,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio-to-audio generation failed: {str(e)}")


@router.post("/audio-to-audio/generate/stream")
async def audio_to_audio_generate_stream(request: AudioToAudioRequest, req: Request):
    """
    Audio-to-audio generation with SSE streaming and optional cohesion processing.
    
    Same as /audio-to-audio/generate but streams progress events.
    
    Returns Server-Sent Events:
    - stage: "preprocessing" | "generating" | "postprocessing" | "complete" | "error"
    - message: Progress message
    - result: Final audio data with metadata (on complete)
    """
    from app.services.audio_renderer import generate_audio_to_audio
    from app.services.audio_processing import (
        preprocess_user_recording,
        postprocess_generated_audio,
    )
    
    async def event_generator():
        try:
            source_bpm = None
            was_input_stretched = False
            output_bpm = None
            was_output_stretched = False
            
            yield f"data: {json.dumps({'stage': 'processing', 'message': 'Decoding audio input...'})}\n\n"
            
            # Decode the base64 audio
            audio_bytes = base64.b64decode(request.audio_data)
            
            # PRE-PROCESSING
            if request.target_tempo and request.preprocess_input:
                yield f"data: {json.dumps({'stage': 'preprocessing', 'message': f'Aligning to {request.target_tempo} BPM...'})}\n\n"
                audio_bytes, source_bpm = preprocess_user_recording(
                    audio_bytes,
                    target_bpm=request.target_tempo,
                )
                was_input_stretched = source_bpm is not None and abs(source_bpm - request.target_tempo) > 2
            
            # Build prompt with style/mood context
            prompt_parts = [request.prompt]
            if request.style:
                prompt_parts.append(request.style)
            if request.mood:
                prompt_parts.append(request.mood)
            if request.target_tempo:
                prompt_parts.append(f"{request.target_tempo} BPM")
            
            full_prompt = ", ".join(prompt_parts)
            enhanced_prompt = await build_audio_to_audio_prompt(full_prompt)
            
            yield f"data: {json.dumps({'stage': 'generating', 'message': f'Generating: {enhanced_prompt[:60]}...'})}\n\n"
            
            # Generate using Stable Audio
            generated_audio = await generate_audio_to_audio(
                reference_audio=audio_bytes,
                prompt=enhanced_prompt,
                duration=request.duration,
                strength=request.strength,
                cfg_scale=request.cfg_scale,
                steps=request.steps,
                seed=request.seed,
            )
            
            # POST-PROCESSING
            if request.target_tempo and request.postprocess_output:
                yield f"data: {json.dumps({'stage': 'postprocessing', 'message': 'Aligning output to target tempo...'})}\n\n"
                generated_audio, metadata = postprocess_generated_audio(
                    generated_audio,
                    target_bpm=request.target_tempo,
                    target_duration=float(request.duration),
                    max_bpm_drift=request.max_bpm_drift,
                )
                output_bpm = metadata.get("detected_bpm")
                was_output_stretched = metadata.get("was_stretched", False)
            
            # Encode result
            audio_b64 = base64.b64encode(generated_audio).decode()
            
            name = request.prompt[:30].strip()
            if len(request.prompt) > 30:
                name += "..."
            
            result = {
                'audio_data': audio_b64,
                'name': name,
                'duration': request.duration,
                'source_bpm': source_bpm,
                'output_bpm': output_bpm,
                'was_input_stretched': was_input_stretched,
                'was_output_stretched': was_output_stretched,
            }
            
            yield f"data: {json.dumps({'stage': 'complete', 'message': 'Generation complete!', 'result': result})}\n\n"
            
        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Cohesion-Constrained Generation Endpoint
# ============================================================================


@router.post("/audio-to-audio/generate/cohesion", response_model=CohesionGenerateResponse)
async def audio_to_audio_generate_with_cohesion(request: CohesionGenerateRequest):
    """
    Generate a new instrument track constrained to match existing tracks.
    
    This endpoint uses CohesionSpec to ensure the new track fits with
    previously generated audio. It:
    
    1. Uses the reference_audio as a guide for audio-to-audio generation
    2. Applies strength profile based on instrument type (tight/medium/loose)
    3. Validates output BPM matches target tempo
    4. Retries if validation fails (up to max_retries)
    
    Use this when adding new instruments to an existing song to ensure
    all tracks play well together.
    
    Args:
        instrument: What instrument to generate (e.g., "bass", "drums")
        prompt: Additional style guidance
        reference_audio: Base64 WAV of existing track to match
        style: Locked style/genre (e.g., "dark synthwave")
        tempo: Target BPM (used for validation)
        duration: Output duration in seconds
        strength_profile: How closely to match reference ("tight"/"medium"/"loose")
        mood: Optional mood keywords
        validate_bpm: Whether to check output BPM
        max_bpm_drift: Max allowed BPM difference from target
        max_retries: Retry attempts on validation failure
    
    Returns:
        Generated audio with validation results
    """
    from app.services.per_instrument_agent import generate_single_instrument
    
    try:
        # Parse strength profile
        try:
            profile = StrengthProfile(request.strength_profile.lower())
        except ValueError:
            profile = StrengthProfile.MEDIUM
        
        # Build cohesion spec
        cohesion_spec = CohesionSpec(
            style=request.style,
            tempo=request.tempo,
            duration=request.duration,
            reference_audio=request.reference_audio,
            strength_profile=profile,
            max_bpm_drift=request.max_bpm_drift,
            mood=request.mood,
        )
        
        # Generate with cohesion constraints
        audio_bytes, validation_result = await generate_single_instrument(
            instrument=request.instrument,
            prompt=request.prompt,
            cohesion_spec=cohesion_spec,
            validate_cohesion=request.validate_bpm,
            max_retries=request.max_retries,
        )
        
        # Encode result
        audio_b64 = base64.b64encode(audio_bytes).decode()
        
        return CohesionGenerateResponse(
            audio_data=audio_b64,
            name=request.instrument,
            duration=request.duration,
            validation=validation_result,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohesion generation failed: {str(e)}")
