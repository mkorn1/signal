/**
 * Streaming stem agent loop - uses SSE for real-time conversation updates.
 *
 * Consumes SSE from /api/agent/stem/step/stream and handles tool execution
 * (generateStems) when the agent decides to generate audio.
 *
 * Supports multi-turn conversations for gathering user preferences.
 */

import {
    executeStemToolCall,
    type GeneratedStem,
    type StemToolCall,
    type StemToolResult,
} from "./stemToolExecutor"

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"

/** SSE event types from the backend */
type SSEEventType =
  | "thinking"
  | "tool_calls"
  | "tool_results_received"
  | "message"
  | "error"

interface SSEEvent {
  type: SSEEventType
  thread_id: string
  content?: string
  tool_calls?: StemToolCall[]
  count?: number
  done?: boolean
  error?: string
}

/** Reason why the stem agent loop ended */
export type StemAgentStopReason =
  | "complete" // Agent finished (may continue in next turn)
  | "generated" // Audio was generated successfully
  | "error" // An error occurred
  | "aborted" // User aborted

/** Result of running the stem agent loop */
export interface StemAgentResult {
  success: boolean
  message?: string
  threadId?: string
  stopReason: StemAgentStopReason
  generatedStems?: GeneratedStem[]
}

export interface StemAgentCallbacks {
  /** Called when agent is thinking/processing (streamed tokens) */
  onThinking?: (content: string) => void
  /** Called when tool calls need to be executed */
  onToolCalls?: (toolCalls: StemToolCall[]) => void
  /** Called after stems are generated */
  onStemsGenerated?: (stems: GeneratedStem[]) => void
  /** Called when agent sends final message */
  onMessage?: (message: string) => void
  /** Called on any error */
  onError?: (error: Error) => void
  /** Called when stream completes */
  onComplete?: (result: StemAgentResult) => void
}

/**
 * Parse SSE data from a text chunk.
 */
function parseSSEEvents(text: string): SSEEvent[] {
  const events: SSEEvent[] = []
  const lines = text.split("\n")

  for (const line of lines) {
    if (line.startsWith("data: ")) {
      try {
        const json = line.slice(6)
        if (json.trim()) {
          events.push(JSON.parse(json))
        }
      } catch (e) {
        console.warn("[StemAgent] Failed to parse SSE event:", line, e)
      }
    }
  }

  return events
}

/**
 * Consume SSE stream from a fetch response.
 */
async function* consumeSSEStream(
  response: Response,
  abortSignal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error("No response body")
  }

  const decoder = new TextDecoder()
  let buffer = ""

  try {
    while (true) {
      if (abortSignal?.aborted) {
        break
      }

      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      const parts = buffer.split("\n\n")
      buffer = parts.pop() ?? ""

      for (const part of parts) {
        const events = parseSSEEvents(part + "\n")
        for (const event of events) {
          yield event
        }
      }
    }

    if (buffer.trim()) {
      const events = parseSSEEvents(buffer)
      for (const event of events) {
        yield event
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/**
 * Start a streaming stem agent request.
 */
async function startStreamRequest(
  prompt: string,
  threadId?: string,
  abortSignal?: AbortSignal,
): Promise<Response> {
  const response = await fetch(`${API_BASE}/api/agent/stem/step/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      thread_id: threadId,
    }),
    signal: abortSignal,
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `HTTP ${response.status}`)
  }

  return response
}

/**
 * Resume streaming after tool execution.
 */
async function resumeStreamRequest(
  threadId: string,
  toolResults: StemToolResult[],
  abortSignal?: AbortSignal,
): Promise<Response> {
  const response = await fetch(`${API_BASE}/api/agent/stem/step/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      thread_id: threadId,
      tool_results: toolResults,
    }),
    signal: abortSignal,
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    throw new Error(errorData.detail || `HTTP ${response.status}`)
  }

  return response
}

/**
 * Run the streaming stem agent loop.
 *
 * Streams events from the backend, executes generateStems tool calls
 * when the agent decides to generate, and reconnects to stream results.
 *
 * @param prompt - The user's message
 * @param options - Configuration options
 * @param options.threadId - Optional thread ID to continue an existing conversation
 * @param options.callbacks - Event callbacks
 * @param options.abortSignal - Signal to abort the operation
 */
export async function runStemAgentLoop(
  prompt: string,
  options?: {
    threadId?: string
    callbacks?: StemAgentCallbacks
    abortSignal?: AbortSignal
  },
): Promise<StemAgentResult> {
  const { threadId: existingThreadId, callbacks, abortSignal } = options ?? {}
  let threadId: string | null = existingThreadId ?? null
  let thinkingBuffer = ""
  let allGeneratedStems: GeneratedStem[] = []

  const makeResult = (
    success: boolean,
    stopReason: StemAgentStopReason,
    message?: string,
  ): StemAgentResult => {
    const result: StemAgentResult = {
      success,
      stopReason,
      message,
      threadId: threadId ?? undefined,
      generatedStems:
        allGeneratedStems.length > 0 ? allGeneratedStems : undefined,
    }
    callbacks?.onComplete?.(result)
    return result
  }

  try {
    // Start stream
    let response = await startStreamRequest(
      prompt,
      threadId ?? undefined,
      abortSignal,
    )

    while (true) {
      if (abortSignal?.aborted) {
        throw new Error("Aborted")
      }

      let pendingToolCalls: StemToolCall[] | null = null
      let finalMessage: string | null = null
      let hasError = false

      // Consume the stream
      for await (const event of consumeSSEStream(response, abortSignal)) {
        if (event.thread_id) {
          threadId = event.thread_id
        }

        switch (event.type) {
          case "thinking":
            if (event.content) {
              thinkingBuffer += event.content
              callbacks?.onThinking?.(event.content)
            }
            break

          case "tool_calls":
            if (event.tool_calls && event.tool_calls.length > 0) {
              pendingToolCalls = event.tool_calls
              callbacks?.onToolCalls?.(event.tool_calls)
            }
            break

          case "tool_results_received":
            break

          case "message":
            if (event.content) {
              finalMessage = event.content
              callbacks?.onMessage?.(event.content)
            }
            if (event.done) {
              const stopReason =
                allGeneratedStems.length > 0 ? "generated" : "complete"
              return makeResult(true, stopReason, finalMessage ?? undefined)
            }
            break

          case "error":
            hasError = true
            const error = new Error(event.error ?? "Unknown error")
            callbacks?.onError?.(error)
            return makeResult(false, "error", event.error)
        }
      }

      // Stream ended - check what to do next
      if (hasError) {
        return makeResult(false, "error", "Stream error")
      }

      if (pendingToolCalls && pendingToolCalls.length > 0 && threadId) {
        // Execute all tool calls (should be generateStems)
        const toolResults: StemToolResult[] = []

        for (const toolCall of pendingToolCalls) {
          const { result, stems } = await executeStemToolCall(toolCall)
          toolResults.push(result)

          if (stems.length > 0) {
            allGeneratedStems.push(...stems)
            callbacks?.onStemsGenerated?.(stems)
          }
        }

        // Reset thinking buffer for next round
        thinkingBuffer = ""

        // Resume with tool results
        response = await resumeStreamRequest(threadId, toolResults, abortSignal)
        continue
      }

      // No more work to do
      if (finalMessage) {
        const stopReason =
          allGeneratedStems.length > 0 ? "generated" : "complete"
        return makeResult(true, stopReason, finalMessage)
      }

      // Unexpected end
      const stopReason =
        allGeneratedStems.length > 0 ? "generated" : "complete"
      return makeResult(true, stopReason, thinkingBuffer || undefined)
    }
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error))
    if (err.message === "Aborted") {
      return makeResult(false, "aborted", "Aborted by user")
    }
    callbacks?.onError?.(err)
    return makeResult(false, "error", err.message)
  }
}
