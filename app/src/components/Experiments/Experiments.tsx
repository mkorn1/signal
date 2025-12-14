import styled from "@emotion/styled"
import FastForward from "mdi-react/FastForwardIcon"
import FastRewind from "mdi-react/FastRewindIcon"
import Pause from "mdi-react/PauseIcon"
import PlayArrow from "mdi-react/PlayArrowIcon"
import Stop from "mdi-react/StopIcon"
import VolumeHigh from "mdi-react/VolumeHighIcon"
import VolumeMute from "mdi-react/VolumeMuteIcon"
import { FC, useCallback, useEffect, useMemo, useRef, useState } from "react"

type AgentType = "per_instrument" | "stem_separation" | "conversational_stem"

interface AudioTrack {
  name: string
  audioData: string // base64 WAV
  audioUrl?: string // blob URL for playback
  duration?: number // duration in seconds
  muted: boolean
  solo: boolean
  volume: number
}

// Styled Components
const Container = styled.div`
  height: 100vh;
  width: 100vw;
  display: flex;
  flex-direction: column;
  background: ${({ theme }) => theme.backgroundColor};
  color: ${({ theme }) => theme.textColor};
`

const Header = styled.div`
  padding: 12px 20px;
  border-bottom: 1px solid ${({ theme }) => theme.dividerColor};
  display: flex;
  gap: 12px;
  align-items: center;
  flex-shrink: 0;
`

const Title = styled.h1`
  font-size: 16px;
  margin: 0;
  margin-right: 12px;
`

const Select = styled.select`
  padding: 6px 10px;
  border-radius: 4px;
  border: 1px solid ${({ theme }) => theme.dividerColor};
  background: ${({ theme }) => theme.secondaryBackgroundColor};
  color: ${({ theme }) => theme.textColor};
  font-size: 13px;
`

const Input = styled.input`
  flex: 1;
  padding: 6px 10px;
  border-radius: 4px;
  border: 1px solid ${({ theme }) => theme.dividerColor};
  background: ${({ theme }) => theme.secondaryBackgroundColor};
  color: ${({ theme }) => theme.textColor};
  font-size: 13px;
  max-width: 400px;
`

const Button = styled.button<{ disabled?: boolean }>`
  padding: 6px 14px;
  border-radius: 4px;
  border: none;
  background: ${({ theme, disabled }) =>
    disabled ? theme.dividerColor : theme.themeColor};
  color: white;
  font-size: 13px;
  cursor: ${({ disabled }) => (disabled ? "not-allowed" : "pointer")};
  opacity: ${({ disabled }) => (disabled ? 0.6 : 1)};

  &:hover:not(:disabled) {
    opacity: 0.9;
  }
`

const StatusBadge = styled.span<{ status: "idle" | "loading" | "done" | "error" }>`
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 11px;
  background: ${({ status }) =>
    status === "loading"
      ? "#f59e0b"
      : status === "done"
        ? "#10b981"
        : status === "error"
          ? "#ef4444"
          : "#6b7280"};
  color: white;
`

const MainContent = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`

// Chat Panel - Top Half
const ChatPanel = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 200px;
  max-height: 50vh;
  border-bottom: 2px solid ${({ theme }) => theme.dividerColor};
`

const ChatMessages = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  background: ${({ theme }) => theme.darkBackgroundColor};
`

const ChatMessage = styled.div<{ role: "user" | "agent" | "system" }>`
  padding: 10px 14px;
  border-radius: 12px;
  max-width: 85%;
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
  word-wrap: break-word;
  align-self: ${({ role }) => (role === "user" ? "flex-end" : "flex-start")};
  background: ${({ role, theme }) =>
    role === "user"
      ? theme.themeColor
      : role === "system"
        ? theme.secondaryBackgroundColor
        : theme.backgroundColor};
  color: ${({ role, theme }) =>
    role === "user" ? "white" : theme.textColor};
  border: ${({ role, theme }) =>
    role === "agent" ? `1px solid ${theme.dividerColor}` : "none"};
`

const ChatInputArea = styled.div`
  padding: 12px;
  border-top: 1px solid ${({ theme }) => theme.dividerColor};
  background: ${({ theme }) => theme.backgroundColor};
  display: flex;
  gap: 8px;
  align-items: flex-end;
`

const ChatInput = styled.textarea`
  flex: 1;
  padding: 10px 14px;
  border-radius: 8px;
  border: 1px solid ${({ theme }) => theme.dividerColor};
  background: ${({ theme }) => theme.secondaryBackgroundColor};
  color: ${({ theme }) => theme.textColor};
  font-size: 13px;
  font-family: inherit;
  resize: none;
  min-height: 44px;
  max-height: 120px;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.themeColor};
  }
`

const ChatButton = styled.button<{ variant?: "primary" | "secondary" }>`
  padding: 10px 18px;
  border-radius: 8px;
  border: none;
  background: ${({ theme, variant }) =>
    variant === "secondary" ? theme.secondaryBackgroundColor : theme.themeColor};
  color: ${({ variant }) => (variant === "secondary" ? "inherit" : "white")};
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;

  &:hover:not(:disabled) {
    opacity: 0.9;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`

// Timeline Panel - Bottom Half
const TimelinePanel = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-height: 200px;
`

const StreamLog = styled.div`
  font-family: monospace;
  font-size: 11px;
  background: ${({ theme }) => theme.secondaryBackgroundColor};
  padding: 8px 12px;
  white-space: pre-wrap;
  max-height: 60px;
  overflow-y: auto;
  border-bottom: 1px solid ${({ theme }) => theme.dividerColor};
  flex-shrink: 0;
`

// Transport Panel
const TransportBar = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px 16px;
  background: ${({ theme }) => theme.backgroundColor};
  border-bottom: 1px solid ${({ theme }) => theme.dividerColor};
  gap: 4px;
  flex-shrink: 0;
`

const CircleButton = styled.div<{ active?: boolean }>`
  border-radius: 100%;
  margin: 0 2px;
  padding: 6px;
  color: ${({ theme, active }) => (active ? theme.themeColor : theme.textColor)};
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  background: ${({ active }) => (active ? "rgba(255,255,255,0.1)" : "transparent")};

  &:hover {
    background: ${({ theme }) => theme.highlightColor};
  }

  svg {
    width: 18px;
    height: 18px;
  }
`

const PlayButtonStyled = styled(CircleButton)`
  background: ${({ theme }) => theme.themeColor};
  color: white;

  &:hover {
    background: ${({ theme }) => theme.themeColor};
    opacity: 0.85;
  }
`

const TransportSeparator = styled.div`
  background: ${({ theme }) => theme.dividerColor};
  margin: 0 12px;
  width: 1px;
  height: 16px;
`

const TimeDisplay = styled.div`
  font-family: monospace;
  font-size: 13px;
  color: ${({ theme }) => theme.secondaryTextColor};
  min-width: 80px;
`

const ZoomControls = styled.div`
  display: flex;
  align-items: center;
  gap: 4px;
  margin-left: auto;
`

const ZoomButton = styled.button`
  padding: 4px 8px;
  border-radius: 4px;
  border: 1px solid ${({ theme }) => theme.dividerColor};
  background: ${({ theme }) => theme.secondaryBackgroundColor};
  color: ${({ theme }) => theme.textColor};
  font-size: 12px;
  cursor: pointer;

  &:hover {
    background: ${({ theme }) => theme.highlightColor};
  }
`

// Timeline Area
const TimelineContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: ${({ theme }) => theme.darkBackgroundColor};
`

const TimelineRuler = styled.div`
  height: 24px;
  background: ${({ theme }) => theme.backgroundColor};
  border-bottom: 1px solid ${({ theme }) => theme.dividerColor};
  position: relative;
  flex-shrink: 0;
  margin-left: 140px;
`

const RulerCanvas = styled.canvas`
  width: 100%;
  height: 100%;
`

const TracksArea = styled.div`
  flex: 1;
  display: flex;
  overflow-y: auto;
  overflow-x: hidden;
`

const TrackList = styled.div`
  width: 140px;
  flex-shrink: 0;
  border-right: 1px solid ${({ theme }) => theme.dividerColor};
  background: ${({ theme }) => theme.backgroundColor};
`

const TrackHeader = styled.div<{ selected?: boolean }>`
  height: 80px;
  padding: 8px;
  border-bottom: 1px solid ${({ theme }) => theme.dividerColor};
  display: flex;
  flex-direction: column;
  gap: 4px;
  background: ${({ theme, selected }) =>
    selected ? theme.secondaryBackgroundColor : theme.backgroundColor};
`

const TrackName = styled.div`
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`

const TrackControls = styled.div`
  display: flex;
  align-items: center;
  gap: 4px;
`

const SmallButton = styled.button<{ active?: boolean }>`
  padding: 2px 6px;
  border-radius: 2px;
  border: 1px solid ${({ theme }) => theme.dividerColor};
  background: ${({ active, theme }) =>
    active ? theme.themeColor : theme.secondaryBackgroundColor};
  color: ${({ active }) => (active ? "white" : "inherit")};
  font-size: 10px;
  cursor: pointer;

  &:hover {
    opacity: 0.8;
  }
`

const VolumeSlider = styled.input`
  width: 60px;
  height: 4px;
  cursor: pointer;
`

const WaveformArea = styled.div`
  flex: 1;
  overflow-x: auto;
  overflow-y: hidden;
  position: relative;
`

const WaveformScrollContainer = styled.div<{ width: number }>`
  min-width: ${({ width }) => width}px;
  position: relative;
`

const WaveformTrack = styled.div`
  height: 80px;
  border-bottom: 1px solid ${({ theme }) => theme.dividerColor};
  position: relative;
  background: ${({ theme }) => theme.editorBackgroundColor};
`

const WaveformCanvas = styled.canvas`
  position: absolute;
  top: 4px;
  left: 0;
  height: 72px;
`

const Playhead = styled.div<{ position: number }>`
  position: absolute;
  top: 0;
  left: ${({ position }) => position}px;
  width: 1px;
  height: 100%;
  background: ${({ theme }) => theme.themeColor};
  pointer-events: none;
  z-index: 10;

  &::before {
    content: "";
    position: absolute;
    top: 0;
    left: -4px;
    width: 9px;
    height: 9px;
    background: ${({ theme }) => theme.themeColor};
    clip-path: polygon(50% 100%, 0 0, 100% 0);
  }
`

const EmptyState = styled.div`
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${({ theme }) => theme.secondaryTextColor};
  font-size: 14px;
`

// Helper to format time
const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  const ms = Math.floor((seconds % 1) * 100)
  return `${mins}:${secs.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`
}

interface ChatMessageData {
  role: "user" | "agent" | "system"
  content: string
}

export const Experiments: FC = () => {
  const [agentType, setAgentType] = useState<AgentType>("conversational_stem")
  const [prompt, setPrompt] = useState("")
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle")
  const [streamLog, setStreamLog] = useState<string[]>([])
  const [audioTracks, setAudioTracks] = useState<AudioTrack[]>([])
  const [error, setError] = useState<string | null>(null)
  const [threadId, setThreadId] = useState<string | null>(null)
  const [conversationMode, setConversationMode] = useState(false)
  const [chatMessages, setChatMessages] = useState<ChatMessageData[]>([])
  const [streamingMessage, setStreamingMessage] = useState("")
  const logRef = useRef<HTMLDivElement>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const audioUrlsRef = useRef<string[]>([])

  // Playback state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const audioRefs = useRef<HTMLAudioElement[]>([])
  const animationRef = useRef<number | null>(null)

  // Timeline state
  const [pixelsPerSecond, setPixelsPerSecond] = useState(100)
  const [selectedTrackIndex, setSelectedTrackIndex] = useState<number | null>(null)
  const waveformAreaRef = useRef<HTMLDivElement>(null)
  const rulerCanvasRef = useRef<HTMLCanvasElement>(null)

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      audioUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [])

  const base64ToAudioUrl = (base64: string): string => {
    try {
      const binary = atob(base64)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i)
      }
      const blob = new Blob([bytes], { type: "audio/wav" })
      const url = URL.createObjectURL(blob)
      audioUrlsRef.current.push(url)
      return url
    } catch (e) {
      console.error("Failed to decode audio:", e)
      return ""
    }
  }

  // Convert tracks to playable URLs
  const playableTracks = useMemo(() => {
    return audioTracks.map((track) => ({
      ...track,
      audioUrl: track.audioUrl || base64ToAudioUrl(track.audioData),
    }))
  }, [audioTracks])

  // Setup audio elements
  useEffect(() => {
    audioRefs.current = playableTracks.map((track, i) => {
      const existing = audioRefs.current[i]
      if (existing && existing.src === track.audioUrl) {
        return existing
      }
      const audio = new Audio(track.audioUrl)
      audio.preload = "auto"
      return audio
    })

    // Get max duration
    Promise.all(
      audioRefs.current.map(
        (audio) =>
          new Promise<number>((resolve) => {
            if (audio.duration && !isNaN(audio.duration)) {
              resolve(audio.duration)
            } else {
              audio.addEventListener("loadedmetadata", () => resolve(audio.duration), {
                once: true,
              })
            }
          }),
      ),
    ).then((durations) => {
      const maxDuration = Math.max(...durations, 0)
      setDuration(maxDuration)
    })
  }, [playableTracks])

  // Update track volumes/mutes
  useEffect(() => {
    audioRefs.current.forEach((audio, i) => {
      const track = audioTracks[i]
      if (track) {
        const hasSolo = audioTracks.some((t) => t.solo)
        const shouldMute = track.muted || (hasSolo && !track.solo)
        audio.muted = shouldMute
        audio.volume = track.volume
      }
    })
  }, [audioTracks])

  // Playback animation
  useEffect(() => {
    if (isPlaying) {
      const updateTime = () => {
        const firstAudio = audioRefs.current[0]
        if (firstAudio) {
          setCurrentTime(firstAudio.currentTime)
          if (firstAudio.ended) {
            setIsPlaying(false)
            setCurrentTime(0)
            audioRefs.current.forEach((a) => (a.currentTime = 0))
          }
        }
        animationRef.current = requestAnimationFrame(updateTime)
      }
      animationRef.current = requestAnimationFrame(updateTime)
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isPlaying])

  // Draw ruler
  useEffect(() => {
    const canvas = rulerCanvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * window.devicePixelRatio
    canvas.height = rect.height * window.devicePixelRatio
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio)

    ctx.fillStyle = "#1a1a1a"
    ctx.fillRect(0, 0, rect.width, rect.height)

    ctx.fillStyle = "#888"
    ctx.font = "10px monospace"

    const step = pixelsPerSecond >= 50 ? 1 : pixelsPerSecond >= 20 ? 5 : 10
    for (let sec = 0; sec <= duration + 10; sec += step) {
      const x = sec * pixelsPerSecond
      if (x > rect.width + 100) break

      ctx.fillStyle = "#666"
      ctx.fillRect(x, 16, 1, 8)

      ctx.fillStyle = "#888"
      ctx.fillText(formatTime(sec), x + 2, 12)
    }
  }, [duration, pixelsPerSecond])

  const appendLog = (msg: string) => {
    const timestamp = new Date().toLocaleTimeString()
    setStreamLog((prev) => [...prev.slice(-50), `[${timestamp}] ${msg}`])
    setTimeout(() => {
      logRef.current?.scrollTo(0, logRef.current.scrollHeight)
    }, 10)
  }

  const handleGenerate = async () => {
    if (!prompt.trim()) return

    // Stop any current playback
    handleStop()

    // For non-conversational agents, cleanup and reset
    if (agentType !== "conversational_stem") {
      audioUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
      audioUrlsRef.current = []
      setAudioTracks([])
      setStreamLog([])
    }

    setStatus("loading")
    setError(null)

    // Conversational stem uses different endpoint and flow
    if (agentType === "conversational_stem") {
      await handleConversationalStem()
      return
    }

    const endpoint =
      agentType === "per_instrument"
        ? "/api/generate/per-instrument/stream"
        : "/api/generate/stem-separation/stream"

    appendLog(`Starting generation with ${agentType} agent`)

    try {
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      appendLog("Connected to stream...")

      const reader = response.body?.getReader()
      if (!reader) throw new Error("No response body")

      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6))
              appendLog(`${data.stage}: ${data.message || ""}`)

              if (data.stage === "complete" && data.result) {
                const tracks: AudioTrack[] = data.result.tracks.map(
                  (t: { name: string; audio_data: string }) => ({
                    name: t.name,
                    audioData: t.audio_data,
                    muted: false,
                    solo: false,
                    volume: 1,
                  }),
                )
                appendLog(`Received ${tracks.length} tracks`)
                setAudioTracks(tracks)
                setStatus("done")
              } else if (data.stage === "error") {
                throw new Error(data.error || "Generation failed")
              }
            } catch (parseErr) {
              if (!(parseErr instanceof SyntaxError)) {
                throw parseErr
              }
            }
          }
        }
      }

      reader.releaseLock()
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      appendLog(`Error: ${msg}`)
      setError(msg)
      setStatus("error")
    }
  }

  // Scroll chat to bottom
  const scrollChatToBottom = () => {
    setTimeout(() => {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
    }, 50)
  }

  // Handle conversational stem agent - multi-turn conversation
  const handleConversationalStem = async () => {
    const userMessage = prompt.trim()
    setConversationMode(true)
    setChatMessages((prev) => [...prev, { role: "user", content: userMessage }])
    setPrompt("") // Clear input for next message
    setStreamingMessage("")
    scrollChatToBottom()

    try {
      const response = await fetch(`http://localhost:8000/api/agent/stem/step/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: userMessage,
          thread_id: threadId,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error("No response body")

      const decoder = new TextDecoder()
      let buffer = ""
      let agentMessage = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6))

              // Track thread ID for conversation continuity
              if (data.thread_id) {
                setThreadId(data.thread_id)
              }

              if (data.type === "thinking" && data.content) {
                agentMessage += data.content
                setStreamingMessage(agentMessage)
                scrollChatToBottom()
              } else if (data.type === "message" && data.content) {
                // Final message from agent
                setChatMessages((prev) => [...prev, { role: "agent", content: data.content }])
                setStreamingMessage("")
                setStatus("idle") // Ready for next message
                scrollChatToBottom()
              } else if (data.type === "tool_calls" && data.tool_calls?.length > 0) {
                // Agent wants to generate - show thinking message first
                if (agentMessage) {
                  setChatMessages((prev) => [...prev, { role: "agent", content: agentMessage }])
                }
                setStreamingMessage("")

                // Show generation status
                const toolCall = data.tool_calls[0]
                const instruments = toolCall.args?.instruments?.join(", ") || "drums, bass, melody, keys"
                setChatMessages((prev) => [
                  ...prev,
                  { role: "system", content: `🎵 Generating stems: ${instruments}...` },
                ])
                appendLog(`Generating: ${toolCall.args?.style || "N/A"}`)
                scrollChatToBottom()

                // Execute the generation
                await executeGenerateStems(toolCall, data.thread_id)
              } else if (data.type === "error") {
                throw new Error(data.error || "Agent error")
              }
            } catch (parseErr) {
              if (!(parseErr instanceof SyntaxError)) {
                throw parseErr
              }
            }
          }
        }
      }

      // If we got thinking content but no final message, display it
      if (agentMessage && streamingMessage) {
        setChatMessages((prev) => [...prev, { role: "agent", content: agentMessage }])
        setStreamingMessage("")
      }
      setStatus("idle")

      reader.releaseLock()
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      setChatMessages((prev) => [...prev, { role: "system", content: `❌ Error: ${msg}` }])
      setError(msg)
      setStatus("error")
    }
  }

  // Execute the generateStems tool call
  const executeGenerateStems = async (
    toolCall: { id: string; name: string; args: Record<string, unknown> },
    currentThreadId: string,
  ) => {
    const { style, tempo, instruments } = toolCall.args as {
      style?: string
      tempo?: number
      instruments?: string[]
    }

    appendLog(`Generating with: ${style}, ${tempo} BPM`)

    try {
      // Call the per-instrument generation endpoint
      const genResponse = await fetch(
        `http://localhost:8000/api/generate/per-instrument/stream`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: `${style}, with ${instruments?.join(", ") || "drums, bass, melody, keys"}`,
          }),
        },
      )

      if (!genResponse.ok) {
        throw new Error("Generation failed")
      }

      const reader = genResponse.body?.getReader()
      if (!reader) throw new Error("No response body")

      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.stage) {
                appendLog(`${data.stage}: ${data.message || ""}`)
              }

              if (data.stage === "complete" && data.result) {
                const tracks: AudioTrack[] = data.result.tracks.map(
                  (t: { name: string; audio_data: string }) => ({
                    name: t.name,
                    audioData: t.audio_data,
                    muted: false,
                    solo: false,
                    volume: 1,
                  }),
                )
                setChatMessages((prev) => [
                  ...prev,
                  {
                    role: "system",
                    content: `✅ Generated ${tracks.length} tracks: ${tracks.map((t) => t.name).join(", ")}`,
                  },
                ])
                scrollChatToBottom()
                setAudioTracks(tracks)
                setStatus("done")

                // Resume agent with tool results
                await resumeAgentWithResults(currentThreadId, toolCall.id, tracks)
              }
            } catch (parseErr) {
              if (!(parseErr instanceof SyntaxError)) {
                throw parseErr
              }
            }
          }
        }
      }

      reader.releaseLock()
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Generation failed"
      appendLog(`Error: ${msg}`)
      setError(msg)
      setStatus("error")
    }
  }

  // Resume agent conversation after tool execution
  const resumeAgentWithResults = async (
    currentThreadId: string,
    toolCallId: string,
    tracks: AudioTrack[],
  ) => {
    try {
      const response = await fetch(`http://localhost:8000/api/agent/stem/step/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thread_id: currentThreadId,
          tool_results: [
            {
              id: toolCallId,
              result: JSON.stringify({
                success: true,
                trackCount: tracks.length,
                tracks: tracks.map((t) => t.name),
              }),
            },
          ],
        }),
      })

      if (!response.ok) return

      const reader = response.body?.getReader()
      if (!reader) return

      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.type === "message" && data.content) {
                appendLog(`[Agent]: ${data.content}`)
              }
            } catch {
              // Ignore parse errors
            }
          }
        }
      }

      reader.releaseLock()
    } catch {
      // Silent fail for resume
    }
  }

  // New chat for conversational mode
  const handleNewChat = () => {
    setThreadId(null)
    setConversationMode(false)
    setStreamLog([])
    setChatMessages([])
    setStreamingMessage("")
    setAudioTracks([])
    setStatus("idle")
    setError(null)
    audioUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
    audioUrlsRef.current = []
  }

  // Transport controls
  const handlePlay = useCallback(() => {
    if (audioRefs.current.length === 0) return

    if (isPlaying) {
      audioRefs.current.forEach((a) => a.pause())
      setIsPlaying(false)
    } else {
      audioRefs.current.forEach((a) => a.play())
      setIsPlaying(true)
    }
  }, [isPlaying])

  const handleStop = useCallback(() => {
    audioRefs.current.forEach((a) => {
      a.pause()
      a.currentTime = 0
    })
    setIsPlaying(false)
    setCurrentTime(0)
  }, [])

  const handleRewind = useCallback(() => {
    const newTime = Math.max(0, currentTime - 5)
    audioRefs.current.forEach((a) => (a.currentTime = newTime))
    setCurrentTime(newTime)
  }, [currentTime])

  const handleFastForward = useCallback(() => {
    const newTime = Math.min(duration, currentTime + 5)
    audioRefs.current.forEach((a) => (a.currentTime = newTime))
    setCurrentTime(newTime)
  }, [currentTime, duration])

  const handleSeek = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left + (waveformAreaRef.current?.scrollLeft || 0)
      const newTime = x / pixelsPerSecond
      audioRefs.current.forEach((a) => (a.currentTime = newTime))
      setCurrentTime(newTime)
    },
    [pixelsPerSecond],
  )

  // Track controls
  const toggleMute = useCallback((index: number) => {
    setAudioTracks((prev) =>
      prev.map((t, i) => (i === index ? { ...t, muted: !t.muted } : t)),
    )
  }, [])

  const toggleSolo = useCallback((index: number) => {
    setAudioTracks((prev) =>
      prev.map((t, i) => (i === index ? { ...t, solo: !t.solo } : t)),
    )
  }, [])

  const setTrackVolume = useCallback((index: number, volume: number) => {
    setAudioTracks((prev) =>
      prev.map((t, i) => (i === index ? { ...t, volume } : t)),
    )
  }, [])

  // Zoom controls
  const zoomIn = useCallback(() => {
    setPixelsPerSecond((prev) => Math.min(prev * 1.5, 400))
  }, [])

  const zoomOut = useCallback(() => {
    setPixelsPerSecond((prev) => Math.max(prev / 1.5, 20))
  }, [])

  const zoomFit = useCallback(() => {
    if (duration > 0 && waveformAreaRef.current) {
      const width = waveformAreaRef.current.clientWidth - 20
      setPixelsPerSecond(width / duration)
    }
  }, [duration])

  const timelineWidth = Math.max(duration * pixelsPerSecond + 100, 800)
  const playheadPosition = currentTime * pixelsPerSecond

  return (
    <Container>
      <Header>
        <Title>Audio Generation</Title>
        <Select
          value={agentType}
          onChange={(e) => {
            setAgentType(e.target.value as AgentType)
            handleNewChat() // Reset when switching agents
          }}
        >
          <option value="conversational_stem">Conversational Stem</option>
          <option value="per_instrument">Per-Instrument (Direct)</option>
          <option value="stem_separation">Stem Separation (Direct)</option>
        </Select>
        {conversationMode && (
          <ChatButton variant="secondary" onClick={handleNewChat} disabled={status === "loading"}>
            New Chat
          </ChatButton>
        )}
        <StatusBadge status={status}>
          {status === "idle"
            ? "Ready"
            : status === "loading"
              ? "Processing"
              : status === "done"
                ? "Complete"
                : "Error"}
        </StatusBadge>
      </Header>

      <MainContent>
        {/* Chat Panel - Top Half (for conversational mode) */}
        {agentType === "conversational_stem" ? (
          <ChatPanel>
            <ChatMessages>
              {chatMessages.length === 0 && !streamingMessage && (
                <ChatMessage role="agent">
                  👋 Hi! I'm your music producer assistant. Tell me about the music you want to create, and I'll ask some questions to make sure I get it just right.

                  Try something like: "I want a chill lo-fi beat" or "Make me something like Daft Punk"
                </ChatMessage>
              )}
              {chatMessages.map((msg, i) => (
                <ChatMessage key={i} role={msg.role}>
                  {msg.content}
                </ChatMessage>
              ))}
              {streamingMessage && (
                <ChatMessage role="agent">
                  {streamingMessage}
                  <span style={{ opacity: 0.5 }}>▌</span>
                </ChatMessage>
              )}
              <div ref={chatEndRef} />
            </ChatMessages>
            <ChatInputArea>
              <ChatInput
                placeholder={conversationMode ? "Reply to continue..." : "Describe your music idea..."}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault()
                    handleGenerate()
                  }
                }}
                rows={2}
              />
              <ChatButton onClick={handleGenerate} disabled={status === "loading" || !prompt.trim()}>
                {status === "loading" ? "..." : "Send"}
              </ChatButton>
            </ChatInputArea>
          </ChatPanel>
        ) : (
          /* Direct mode - simple input */
          <div style={{ padding: "12px", borderBottom: "1px solid #333" }}>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <Input
                placeholder="Describe your music..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
                style={{ maxWidth: "none", flex: 1 }}
              />
              <Button onClick={handleGenerate} disabled={status === "loading" || !prompt.trim()}>
                {status === "loading" ? "Generating..." : "Generate"}
              </Button>
            </div>
            <StreamLog ref={logRef} style={{ marginTop: "8px", maxHeight: "80px" }}>
              {streamLog.length === 0 ? "Generation log..." : streamLog.join("\n")}
            </StreamLog>
          </div>
        )}

        {/* Timeline Panel - Bottom Half */}
        <TimelinePanel>
          {/* Transport Bar */}
          <TransportBar>
            <CircleButton onClick={handleRewind}>
              <FastRewind />
            </CircleButton>
            <CircleButton onClick={handleStop}>
              <Stop />
            </CircleButton>
            <PlayButtonStyled onClick={handlePlay}>
              {isPlaying ? <Pause /> : <PlayArrow />}
            </PlayButtonStyled>
            <CircleButton onClick={handleFastForward}>
              <FastForward />
            </CircleButton>

            <TransportSeparator />
            <TimeDisplay>
              {formatTime(currentTime)} / {formatTime(duration)}
            </TimeDisplay>

            <ZoomControls>
              <ZoomButton onClick={zoomOut}>-</ZoomButton>
              <ZoomButton onClick={zoomFit}>Fit</ZoomButton>
              <ZoomButton onClick={zoomIn}>+</ZoomButton>
            </ZoomControls>
          </TransportBar>

          {/* Timeline */}
          <TimelineContainer>
            <TimelineRuler>
              <RulerCanvas ref={rulerCanvasRef} />
            </TimelineRuler>

            {playableTracks.length > 0 ? (
              <TracksArea>
                <TrackList>
                  {playableTracks.map((track, i) => (
                    <TrackHeader
                      key={i}
                      selected={selectedTrackIndex === i}
                      onClick={() => setSelectedTrackIndex(i)}
                    >
                      <TrackName title={track.name}>{track.name}</TrackName>
                      <TrackControls>
                        <SmallButton active={track.muted} onClick={() => toggleMute(i)}>
                          M
                        </SmallButton>
                        <SmallButton active={track.solo} onClick={() => toggleSolo(i)}>
                          S
                        </SmallButton>
                        {track.muted ? (
                          <VolumeMute style={{ width: 14, height: 14, opacity: 0.5 }} />
                        ) : (
                          <VolumeHigh style={{ width: 14, height: 14 }} />
                        )}
                        <VolumeSlider
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={track.volume}
                          onChange={(e) => setTrackVolume(i, parseFloat(e.target.value))}
                        />
                      </TrackControls>
                    </TrackHeader>
                  ))}
                </TrackList>

                <WaveformArea ref={waveformAreaRef} onClick={handleSeek}>
                  <WaveformScrollContainer width={timelineWidth}>
                    <Playhead position={playheadPosition} />
                    {playableTracks.map((track, i) => (
                      <WaveformTrackView
                        key={i}
                        track={track}
                        width={timelineWidth}
                        pixelsPerSecond={pixelsPerSecond}
                        muted={track.muted || (audioTracks.some((t) => t.solo) && !track.solo)}
                      />
                    ))}
                  </WaveformScrollContainer>
                </WaveformArea>
              </TracksArea>
            ) : (
              <EmptyState>
                Generate audio to see tracks on the timeline
              </EmptyState>
            )}
          </TimelineContainer>
        </TimelinePanel>
      </MainContent>
    </Container>
  )
}

// Waveform Track Component
interface WaveformTrackViewProps {
  track: AudioTrack & { audioUrl?: string }
  width: number
  pixelsPerSecond: number
  muted: boolean
}

const WaveformTrackView: FC<WaveformTrackViewProps> = ({
  track,
  width,
  pixelsPerSecond,
  muted,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null)

  // Decode audio data
  useEffect(() => {
    if (!track.audioUrl) return

    fetch(track.audioUrl)
      .then((res) => res.arrayBuffer())
      .then((buffer) => {
        const audioCtx = new AudioContext()
        return audioCtx.decodeAudioData(buffer)
      })
      .then(setAudioBuffer)
      .catch(console.error)
  }, [track.audioUrl])

  // Draw waveform
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !audioBuffer) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const duration = audioBuffer.duration
    const trackWidth = duration * pixelsPerSecond

    canvas.width = trackWidth * dpr
    canvas.height = 72 * dpr
    canvas.style.width = `${trackWidth}px`
    ctx.scale(dpr, dpr)

    // Clear
    ctx.fillStyle = "rgba(0, 0, 0, 0)"
    ctx.clearRect(0, 0, trackWidth, 72)

    // Get channel data (use first channel)
    const channelData = audioBuffer.getChannelData(0)
    const samplesPerPixel = Math.ceil(channelData.length / trackWidth)

    // Draw waveform
    const color = muted ? "rgba(100, 100, 100, 0.5)" : "rgba(100, 180, 255, 0.8)"
    ctx.fillStyle = color

    const midY = 36
    for (let x = 0; x < trackWidth; x++) {
      const startSample = Math.floor(x * samplesPerPixel)
      const endSample = Math.min(startSample + samplesPerPixel, channelData.length)

      let min = 0
      let max = 0
      for (let i = startSample; i < endSample; i++) {
        const sample = channelData[i]
        if (sample < min) min = sample
        if (sample > max) max = sample
      }

      const y1 = midY + min * 32
      const y2 = midY + max * 32
      const height = Math.max(y2 - y1, 1)

      ctx.fillRect(x, y1, 1, height)
    }
  }, [audioBuffer, pixelsPerSecond, muted])

  return (
    <WaveformTrack>
      <WaveformCanvas ref={canvasRef} />
    </WaveformTrack>
  )
}
