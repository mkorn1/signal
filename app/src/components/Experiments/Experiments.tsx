import styled from "@emotion/styled"
import DeleteSweep from "mdi-react/DeleteSweepIcon"
import FastForward from "mdi-react/FastForwardIcon"
import FastRewind from "mdi-react/FastRewindIcon"
import Microphone from "mdi-react/MicrophoneIcon"
import MicrophoneOff from "mdi-react/MicrophoneOffIcon"
import Pause from "mdi-react/PauseIcon"
import PlayArrow from "mdi-react/PlayArrowIcon"
import Refresh from "mdi-react/RefreshIcon"
import Stop from "mdi-react/StopIcon"
import VolumeHigh from "mdi-react/VolumeHighIcon"
import VolumeMute from "mdi-react/VolumeMuteIcon"
import { FC, useCallback, useEffect, useMemo, useRef, useState } from "react"

// IndexedDB helpers for storing large audio data (localStorage has 5-10MB limit)
const DB_NAME = "experiments_audio_db"
const DB_VERSION = 1
const STORE_NAME = "audio_tracks"

const openDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION)
    request.onerror = () => reject(request.error)
    request.onsuccess = () => resolve(request.result)
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "id" })
      }
    }
  })
}

const saveTracksToIDB = async (tracks: AudioTrack[]): Promise<void> => {
  try {
    const db = await openDB()
    const tx = db.transaction(STORE_NAME, "readwrite")
    const store = tx.objectStore(STORE_NAME)
    // Clear existing and save new
    store.clear()
    tracks.forEach((track, i) => {
      if (track.audioData) {
        store.put({ id: i, ...track })
      }
    })
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve()
      tx.onerror = () => reject(tx.error)
    })
    db.close()
  } catch (e) {
    console.error("Failed to save tracks to IndexedDB:", e)
  }
}

const loadTracksFromIDB = async (): Promise<AudioTrack[]> => {
  try {
    const db = await openDB()
    const tx = db.transaction(STORE_NAME, "readonly")
    const store = tx.objectStore(STORE_NAME)
    const request = store.getAll()
    const tracks = await new Promise<AudioTrack[]>((resolve, reject) => {
      request.onsuccess = () => resolve(request.result || [])
      request.onerror = () => reject(request.error)
    })
    db.close()
    // Filter out corrupted tracks and sort by id
    // IndexedDB stores tracks with an auto-increment id field
    type StoredTrack = AudioTrack & { id?: number }
    return (tracks as StoredTrack[])
      .filter((t) => t && t.audioData && t.audioData.length > 0)
      .sort((a, b) => (a.id ?? 0) - (b.id ?? 0))
      .map(({ id, ...track }) => track as AudioTrack)
  } catch (e) {
    console.error("Failed to load tracks from IndexedDB:", e)
    return []
  }
}

type AgentType = "conversational_stem" | "audio_to_audio"

// Parameters used to generate a track (for regeneration)
interface GenerationParams {
  prompt: string
  audioData: string // base64 WAV of user recording
  duration: number
  strength: number
  cfgScale: number
  steps: number
}

interface AudioTrack {
  name: string
  audioData: string // base64 WAV
  audioUrl?: string // blob URL for playback
  duration?: number // duration in seconds
  muted: boolean
  solo: boolean
  volume: number
  generationParams?: GenerationParams // params used to generate this track
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

const ClearAllButton = styled.button<{ disabled?: boolean }>`
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 4px;
  border: 1px solid ${({ theme }) => theme.dividerColor};
  background: transparent;
  color: ${({ theme }) => theme.secondaryTextColor};
  font-size: 12px;
  cursor: ${({ disabled }) => (disabled ? "not-allowed" : "pointer")};
  opacity: ${({ disabled }) => (disabled ? 0.5 : 1)};
  transition: all 0.15s ease;

  &:hover:not(:disabled) {
    background: #ef444420;
    border-color: #ef4444;
    color: #ef4444;
  }
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

const RegenerateButton = styled.button<{ disabled?: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2px;
  border-radius: 3px;
  border: 1px solid ${({ theme }) => theme.dividerColor};
  background: ${({ theme }) => theme.secondaryBackgroundColor};
  color: ${({ theme }) => theme.textColor};
  cursor: ${({ disabled }) => (disabled ? "not-allowed" : "pointer")};
  opacity: ${({ disabled }) => (disabled ? 0.4 : 1)};

  &:hover:not(:disabled) {
    background: ${({ theme }) => theme.themeColor};
    color: white;
    border-color: ${({ theme }) => theme.themeColor};
  }

  svg {
    width: 12px;
    height: 12px;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  &.spinning svg {
    animation: spin 1s linear infinite;
  }
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

// Audio-to-Audio Input Panel
const AudioToAudioPanel = styled.div`
  padding: 20px;
  border-bottom: 2px solid ${({ theme }) => theme.dividerColor};
  background: ${({ theme }) => theme.backgroundColor};
`

const AudioToAudioTitle = styled.h3`
  margin: 0 0 16px 0;
  font-size: 14px;
  font-weight: 500;
  color: ${({ theme }) => theme.textColor};
`

const AudioToAudioRow = styled.div`
  display: flex;
  gap: 12px;
  align-items: flex-end;
  margin-bottom: 16px;
`

const AudioToAudioInput = styled.input`
  flex: 1;
  padding: 10px 14px;
  border-radius: 8px;
  border: 1px solid ${({ theme }) => theme.dividerColor};
  background: ${({ theme }) => theme.secondaryBackgroundColor};
  color: ${({ theme }) => theme.textColor};
  font-size: 14px;

  &:focus {
    outline: none;
    border-color: ${({ theme }) => theme.themeColor};
  }
`

const RecordButton = styled.button<{ isRecording?: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 10px 18px;
  border-radius: 8px;
  border: none;
  background: ${({ isRecording }) => (isRecording ? "#ef4444" : "#10b981")};
  color: white;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  min-width: 140px;
  transition: all 0.2s;

  &:hover {
    opacity: 0.9;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  svg {
    width: 18px;
    height: 18px;
  }
`

const GenerateButton = styled.button`
  padding: 10px 24px;
  border-radius: 8px;
  border: none;
  background: ${({ theme }) => theme.themeColor};
  color: white;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;

  &:hover:not(:disabled) {
    opacity: 0.9;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`

const AddInstrumentButton = styled.button`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  width: 100%;
  padding: 8px 12px;
  margin-top: 4px;
  border-radius: 6px;
  border: 1px dashed ${({ theme }) => theme.dividerColor};
  background: transparent;
  color: ${({ theme }) => theme.secondaryTextColor};
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s ease;

  &:hover {
    border-color: ${({ theme }) => theme.themeColor};
    color: ${({ theme }) => theme.themeColor};
    background: ${({ theme }) => theme.themeColor}10;
  }
`

const RecordingStatus = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  border-radius: 8px;
  background: ${({ theme }) => theme.secondaryBackgroundColor};
  font-size: 13px;
`

const RecordingIndicator = styled.div<{ active?: boolean }>`
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: ${({ active }) => (active ? "#ef4444" : "#6b7280")};
  animation: ${({ active }) => (active ? "pulse 1s infinite" : "none")};

  @keyframes pulse {
    0%,
    100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }
`

const AudioPreview = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-radius: 8px;
  background: ${({ theme }) => theme.secondaryBackgroundColor};
  margin-top: 12px;
`

const AudioPreviewButton = styled.button`
  padding: 6px 12px;
  border-radius: 4px;
  border: 1px solid ${({ theme }) => theme.dividerColor};
  background: transparent;
  color: ${({ theme }) => theme.textColor};
  font-size: 12px;
  cursor: pointer;

  &:hover {
    background: ${({ theme }) => theme.highlightColor};
  }
`

const SliderGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
`

const SliderLabel = styled.label`
  font-size: 11px;
  color: ${({ theme }) => theme.secondaryTextColor};
`

const Slider = styled.input`
  width: 120px;
  cursor: pointer;
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

// LocalStorage keys
const STORAGE_KEYS = {
  AGENT_TYPE: "experiments_agent_type",
  // AUDIO_TRACKS moved to IndexedDB - too large for localStorage
  CHAT_MESSAGES: "experiments_chat_messages",
  STREAM_LOG: "experiments_stream_log",
  THREAD_ID: "experiments_thread_id",
  CONVERSATION_MODE: "experiments_conversation_mode",
}

// Clean up old localStorage audio data (migrated to IndexedDB)
if (localStorage.getItem("experiments_audio_tracks")) {
  localStorage.removeItem("experiments_audio_tracks")
}

export const Experiments: FC = () => {
  // Load persisted state from localStorage
  const [agentType, setAgentType] = useState<AgentType>(() => {
    const saved = localStorage.getItem(STORAGE_KEYS.AGENT_TYPE)
    return (saved as AgentType) || "audio_to_audio"
  })
  const [prompt, setPrompt] = useState("")
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">("idle")
  const [streamLog, setStreamLog] = useState<string[]>(() => {
    const saved = localStorage.getItem(STORAGE_KEYS.STREAM_LOG)
    return saved ? JSON.parse(saved) : []
  })
  const [audioTracks, setAudioTracks] = useState<AudioTrack[]>([])
  const [tracksLoaded, setTracksLoaded] = useState(false)

  // Load tracks from IndexedDB on mount
  useEffect(() => {
    loadTracksFromIDB().then((tracks) => {
      setAudioTracks(tracks)
      setTracksLoaded(true)
    })
  }, [])
  const [error, setError] = useState<string | null>(null)
  const [threadId, setThreadId] = useState<string | null>(() => {
    return localStorage.getItem(STORAGE_KEYS.THREAD_ID)
  })
  const [conversationMode, setConversationMode] = useState(() => {
    return localStorage.getItem(STORAGE_KEYS.CONVERSATION_MODE) === "true"
  })
  const [chatMessages, setChatMessages] = useState<ChatMessageData[]>(() => {
    const saved = localStorage.getItem(STORAGE_KEYS.CHAT_MESSAGES)
    return saved ? JSON.parse(saved) : []
  })
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

  // Audio-to-audio recording state
  const [isRecording, setIsRecording] = useState(false)
  const [recordedAudio, setRecordedAudio] = useState<Blob | null>(null)
  const [recordedAudioUrl, setRecordedAudioUrl] = useState<string | null>(null)
  const [recordingDuration, setRecordingDuration] = useState(0)
  const [transformStrength, setTransformStrength] = useState(0.25)
  const [outputDuration, setOutputDuration] = useState(20)
  const [cfgScale, setCfgScale] = useState(12) // Prompt adherence (1-25)
  const [steps, setSteps] = useState(80) // Quality steps (30-100)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const recordingTimerRef = useRef<number | null>(null)
  const previewAudioRef = useRef<HTMLAudioElement | null>(null)
  const promptInputRef = useRef<HTMLInputElement>(null)

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      audioUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
    }
  }, [])

  // Persist state to localStorage
  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.AGENT_TYPE, agentType)
  }, [agentType])

  useEffect(() => {
    // Only save after initial load to avoid overwriting with empty array
    if (!tracksLoaded) return
    // Save to IndexedDB (handles large audio data)
    const validTracks = audioTracks.filter((t) => t.audioData && t.audioData.length > 0)
    saveTracksToIDB(validTracks)
  }, [audioTracks, tracksLoaded])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.CHAT_MESSAGES, JSON.stringify(chatMessages))
  }, [chatMessages])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.STREAM_LOG, JSON.stringify(streamLog))
  }, [streamLog])

  useEffect(() => {
    if (threadId) {
      localStorage.setItem(STORAGE_KEYS.THREAD_ID, threadId)
    } else {
      localStorage.removeItem(STORAGE_KEYS.THREAD_ID)
    }
  }, [threadId])

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.CONVERSATION_MODE, String(conversationMode))
  }, [conversationMode])

  const base64ToAudioUrl = (base64: string | undefined): string => {
    if (!base64) {
      console.error("base64ToAudioUrl: No audio data provided")
      return ""
    }
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
    // Filter out tracks with invalid audioUrls
    const validTracks = playableTracks.filter((t) => t.audioUrl)

    audioRefs.current = validTracks.map((track, i) => {
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

    setStatus("loading")
    setError(null)

    // Conversational stem uses different endpoint and flow
    if (agentType === "conversational_stem") {
      await handleConversationalStem()
      return
    }

    // Audio-to-audio has its own generate button, skip here
    if (agentType === "audio_to_audio") {
      return
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

  // Audio-to-audio: Start recording from microphone
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" })
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []
      setRecordingDuration(0)

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop())

        // Combine chunks into a blob
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" })

        // Convert webm to wav using AudioContext
        const arrayBuffer = await audioBlob.arrayBuffer()
        const audioContext = new AudioContext()
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)

        // Convert to WAV
        const wavBlob = await audioBufferToWav(audioBuffer)
        setRecordedAudio(wavBlob)

        // Create URL for preview
        if (recordedAudioUrl) {
          URL.revokeObjectURL(recordedAudioUrl)
        }
        const url = URL.createObjectURL(wavBlob)
        setRecordedAudioUrl(url)

        // Clear timer
        if (recordingTimerRef.current) {
          clearInterval(recordingTimerRef.current)
          recordingTimerRef.current = null
        }
      }

      // Start recording
      mediaRecorder.start(100) // Collect data every 100ms
      setIsRecording(true)

      // Start duration timer
      const startTime = Date.now()
      recordingTimerRef.current = window.setInterval(() => {
        setRecordingDuration(Math.floor((Date.now() - startTime) / 1000))
      }, 1000)

      appendLog("Recording started...")
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to access microphone"
      setError(msg)
      appendLog(`Recording error: ${msg}`)
    }
  }

  // Audio-to-audio: Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      appendLog("Recording stopped")
    }
  }

  // Convert AudioBuffer to WAV Blob
  const audioBufferToWav = async (audioBuffer: AudioBuffer): Promise<Blob> => {
    const numChannels = audioBuffer.numberOfChannels
    const sampleRate = audioBuffer.sampleRate
    const format = 1 // PCM
    const bitDepth = 16

    const bytesPerSample = bitDepth / 8
    const blockAlign = numChannels * bytesPerSample
    const dataLength = audioBuffer.length * blockAlign
    const buffer = new ArrayBuffer(44 + dataLength)
    const view = new DataView(buffer)

    // WAV header
    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i))
      }
    }

    writeString(0, "RIFF")
    view.setUint32(4, 36 + dataLength, true)
    writeString(8, "WAVE")
    writeString(12, "fmt ")
    view.setUint32(16, 16, true) // fmt chunk size
    view.setUint16(20, format, true)
    view.setUint16(22, numChannels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * blockAlign, true)
    view.setUint16(32, blockAlign, true)
    view.setUint16(34, bitDepth, true)
    writeString(36, "data")
    view.setUint32(40, dataLength, true)

    // Interleave channels and write samples
    const channels: Float32Array[] = []
    for (let i = 0; i < numChannels; i++) {
      channels.push(audioBuffer.getChannelData(i))
    }

    let offset = 44
    for (let i = 0; i < audioBuffer.length; i++) {
      for (let ch = 0; ch < numChannels; ch++) {
        const sample = Math.max(-1, Math.min(1, channels[ch][i]))
        const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7fff
        view.setInt16(offset, intSample, true)
        offset += 2
      }
    }

    return new Blob([buffer], { type: "audio/wav" })
  }

  // Audio-to-audio: Play preview of recorded audio
  const playPreview = () => {
    if (recordedAudioUrl) {
      if (previewAudioRef.current) {
        previewAudioRef.current.pause()
      }
      previewAudioRef.current = new Audio(recordedAudioUrl)
      previewAudioRef.current.play()
    }
  }

  // Audio-to-audio: Clear recorded audio
  const clearRecording = () => {
    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl)
    }
    setRecordedAudio(null)
    setRecordedAudioUrl(null)
    setRecordingDuration(0)
    if (previewAudioRef.current) {
      previewAudioRef.current.pause()
      previewAudioRef.current = null
    }
  }

  // Audio-to-audio: Add another instrument (clear input and focus)
  const handleAddInstrument = () => {
    setPrompt("")
    clearRecording()
    promptInputRef.current?.focus()
  }

  // Handle audio-to-audio one-shot generation
  const handleAudioToAudio = async () => {
    if (!recordedAudio || !prompt.trim()) {
      setError("Please record audio and enter a prompt")
      return
    }

    setStatus("loading")
    setError(null)
    setStreamLog([])
    appendLog("Starting audio-to-audio generation...")

    try {
      // Convert recorded audio blob to base64
      const arrayBuffer = await recordedAudio.arrayBuffer()
      const base64Audio = btoa(
        new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), ""),
      )

      appendLog(`Sending ${Math.round(arrayBuffer.byteLength / 1024)}KB audio | cfg=${cfgScale}, steps=${steps}`)

      const response = await fetch(`http://localhost:8000/api/audio-to-audio/generate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt.trim(),
          audio_data: base64Audio,
          duration: outputDuration,
          strength: transformStrength,
          cfg_scale: cfgScale,
          steps: steps,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error("No response body")

      const decoder = new TextDecoder()
      let buffer = ""
      let generationCompleted = false

      try {
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

                if (data.stage === "processing" || data.stage === "generating" || data.stage === "preprocessing" || data.stage === "postprocessing") {
                  appendLog(data.message)
                } else if (data.stage === "complete" && data.result) {
                  // Validate result has required fields
                  if (!data.result.audio_data) {
                    throw new Error("Server returned empty audio data")
                  }

                  appendLog("Generation complete!")

                  // Add track to timeline with generation params for regeneration
                  const track: AudioTrack = {
                    name: data.result.name || prompt.trim().slice(0, 30),
                    audioData: data.result.audio_data,
                    muted: false,
                    solo: false,
                    volume: 1,
                    generationParams: {
                      prompt: prompt.trim(),
                      audioData: base64Audio,
                      duration: outputDuration,
                      strength: transformStrength,
                      cfgScale: cfgScale,
                      steps: steps,
                    },
                  }
                  setAudioTracks((prev) => [...prev, track])
                  generationCompleted = true
                  setStatus("done")
                } else if (data.stage === "error") {
                  throw new Error(data.error || "Generation failed")
                }
              } catch (parseErr) {
                if (!(parseErr instanceof SyntaxError)) {
                  throw parseErr
                }
                // Log malformed SSE data for debugging
                console.warn("Malformed SSE data:", line)
              }
            }
          }
        }
      } finally {
        reader.releaseLock()
      }

      // If stream ended without completion, treat as error
      if (!generationCompleted) {
        throw new Error("Generation stream ended without completing - please try again")
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      appendLog(`Error: ${msg}`)
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

  // Resume agent conversation after tool execution (for conversational_stem)
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

  // Clear all - clears everything including IndexedDB, localStorage, and recorded audio
  const handleClearAll = async () => {
    if (!confirm("Clear all generated audio, chat history, and start fresh?")) {
      return
    }

    // Stop any playback
    handleStop()

    // Clear IndexedDB audio tracks
    try {
      const deleteRequest = indexedDB.deleteDatabase(DB_NAME)
      deleteRequest.onerror = () => console.error("Failed to clear audio database")
      deleteRequest.onsuccess = () => console.log("Audio database cleared")
    } catch (err) {
      console.error("Error clearing IndexedDB:", err)
    }

    // Clear all localStorage keys
    Object.values(STORAGE_KEYS).forEach((key) => localStorage.removeItem(key))

    // Clear recorded audio
    if (recordedAudioUrl) {
      URL.revokeObjectURL(recordedAudioUrl)
    }
    setRecordedAudio(null)
    setRecordedAudioUrl(null)
    setRecordingDuration(0)
    if (previewAudioRef.current) {
      previewAudioRef.current.pause()
      previewAudioRef.current = null
    }

    // Reset all state
    setThreadId(null)
    setConversationMode(false)
    setStreamLog([])
    setChatMessages([])
    setStreamingMessage("")
    setAudioTracks([])
    setStatus("idle")
    setError(null)
    setPrompt("")
    setTransformStrength(0.25)
    setOutputDuration(20)
    setCfgScale(12)
    setSteps(80)
    setCurrentTime(0)
    setDuration(0)
    setIsPlaying(false)

    // Clear blob URLs
    audioUrlsRef.current.forEach((url) => URL.revokeObjectURL(url))
    audioUrlsRef.current = []

    appendLog("Cleared all data")
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

  // Regenerate a track using its stored generation params
  const [regeneratingIndex, setRegeneratingIndex] = useState<number | null>(null)

  const handleRegenerateTrack = useCallback(async (index: number) => {
    const track = audioTracks[index]
    if (!track.generationParams) {
      appendLog("Cannot regenerate: no generation params stored")
      return
    }

    setRegeneratingIndex(index)
    appendLog(`Regenerating ${track.name}...`)

    try {
      const { prompt, audioData, duration, strength, cfgScale, steps } = track.generationParams

      const response = await fetch(`http://localhost:8000/api/audio-to-audio/generate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          audio_data: audioData,
          duration,
          strength,
          cfg_scale: cfgScale,
          steps,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error("No response body")

      const decoder = new TextDecoder()
      let buffer = ""

      try {
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

                if (data.stage === "processing" || data.stage === "generating" || data.stage === "preprocessing" || data.stage === "postprocessing") {
                  appendLog(data.message)
                } else if (data.stage === "complete" && data.result) {
                  if (!data.result.audio_data) {
                    throw new Error("Server returned empty audio data")
                  }

                  appendLog(`Regenerated ${track.name}!`)

                  // Update the track in place with new audio but keep same params
                  setAudioTracks((prev) =>
                    prev.map((t, i) =>
                      i === index
                        ? {
                          ...t,
                          audioData: data.result.audio_data,
                          audioUrl: undefined, // Clear cached URL so it gets regenerated
                        }
                        : t
                    )
                  )
                } else if (data.stage === "error") {
                  throw new Error(data.error || "Regeneration failed")
                }
              } catch (parseErr) {
                if (!(parseErr instanceof SyntaxError)) {
                  throw parseErr
                }
              }
            }
          }
        }
      } finally {
        reader.releaseLock()
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error"
      appendLog(`Regeneration error: ${msg}`)
    } finally {
      setRegeneratingIndex(null)
    }
  }, [audioTracks])

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
          <option value="audio_to_audio">Audio to Audio</option>
        </Select>
        {conversationMode && (
          <ChatButton variant="secondary" onClick={handleNewChat} disabled={status === "loading"}>
            New Chat
          </ChatButton>
        )}
        {agentType === "audio_to_audio" && (
          <ClearAllButton onClick={handleClearAll} disabled={status === "loading"}>
            <DeleteSweep style={{ width: 16, height: 16 }} />
            Clear All
          </ClearAllButton>
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
        {/* Audio-to-Audio Panel - Mic input + text prompt */}
        {agentType === "audio_to_audio" ? (
          <AudioToAudioPanel>
            <AudioToAudioTitle>
              🎤 Record your sound (beatbox, hum, etc.) and describe what you want to create
            </AudioToAudioTitle>

            <AudioToAudioRow>
              <AudioToAudioInput
                ref={promptInputRef}
                placeholder="e.g., 'punchy drum beat', 'synthwave lead melody', 'deep bass line'..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={status === "loading"}
              />
              <RecordButton
                isRecording={isRecording}
                onClick={isRecording ? stopRecording : startRecording}
                disabled={status === "loading"}
              >
                {isRecording ? (
                  <>
                    <MicrophoneOff /> Stop ({recordingDuration}s)
                  </>
                ) : (
                  <>
                    <Microphone /> Record
                  </>
                )}
              </RecordButton>
            </AudioToAudioRow>

            {recordedAudio && (
              <AudioPreview>
                <RecordingIndicator />
                <span style={{ flex: 1 }}>
                  Recorded: {recordingDuration}s ({Math.round(recordedAudio.size / 1024)}KB)
                </span>
                <AudioPreviewButton onClick={playPreview}>▶ Preview</AudioPreviewButton>
                <AudioPreviewButton onClick={clearRecording}>✕ Clear</AudioPreviewButton>
              </AudioPreview>
            )}

            <AudioToAudioRow style={{ marginTop: "16px" }}>
              <SliderGroup>
                <SliderLabel>Strength: {Math.round(transformStrength * 100)}%</SliderLabel>
                <Slider
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={transformStrength}
                  onChange={(e) => setTransformStrength(parseFloat(e.target.value))}
                  disabled={status === "loading"}
                  title="How much to transform from input (0=keep original, 1=ignore input)"
                />
              </SliderGroup>
              <SliderGroup>
                <SliderLabel>Duration: {outputDuration}s</SliderLabel>
                <Slider
                  type="range"
                  min="5"
                  max="60"
                  step="5"
                  value={outputDuration}
                  onChange={(e) => setOutputDuration(parseInt(e.target.value))}
                  disabled={status === "loading"}
                  title="Output audio duration in seconds"
                />
              </SliderGroup>
              <SliderGroup>
                <SliderLabel>Prompt: {cfgScale}</SliderLabel>
                <Slider
                  type="range"
                  min="1"
                  max="25"
                  step="1"
                  value={cfgScale}
                  onChange={(e) => setCfgScale(parseInt(e.target.value))}
                  disabled={status === "loading"}
                  title="How closely to follow your prompt (7-15 recommended)"
                />
              </SliderGroup>
              <SliderGroup>
                <SliderLabel>Quality: {steps}</SliderLabel>
                <Slider
                  type="range"
                  min="30"
                  max="100"
                  step="10"
                  value={steps}
                  onChange={(e) => setSteps(parseInt(e.target.value))}
                  disabled={status === "loading"}
                  title="More steps = better quality but slower (50-80 recommended)"
                />
              </SliderGroup>
              <GenerateButton
                onClick={handleAudioToAudio}
                disabled={status === "loading" || !recordedAudio || !prompt.trim()}
              >
                {status === "loading" ? "Generating..." : "Generate"}
              </GenerateButton>
            </AudioToAudioRow>

            <StreamLog ref={logRef} style={{ marginTop: "12px" }}>
              {streamLog.length === 0
                ? "Record audio → describe what you want → generate"
                : streamLog.join("\n")}
            </StreamLog>
          </AudioToAudioPanel>
        ) : agentType === "conversational_stem" ? (
          /* Chat Panel - Conversational mode */
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
                        {track.generationParams && (
                          <RegenerateButton
                            className={regeneratingIndex === i ? "spinning" : ""}
                            disabled={regeneratingIndex !== null}
                            onClick={(e) => {
                              e.stopPropagation()
                              handleRegenerateTrack(i)
                            }}
                            title="Regenerate this track"
                          >
                            <Refresh />
                          </RegenerateButton>
                        )}
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
                  {agentType === "audio_to_audio" && (
                    <AddInstrumentButton onClick={handleAddInstrument}>
                      + Add instrument
                    </AddInstrumentButton>
                  )}
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
