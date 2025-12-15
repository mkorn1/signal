/**
 * Hybrid agent loop - communicates with backend and executes tools on frontend.
 * Supports multi-turn conversation via thread_id persistence.
 */

import type { Song } from "@signal-app/core"
import {
  executeToolCalls,
  type ToolCall,
  type ToolResult,
} from "./toolExecutor"
import {
  serializeSongState,
  formatSongStateForPrompt,
} from "./songStateSerializer"

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"

interface AgentStepResponse {
  thread_id: string
  tool_calls: ToolCall[]
  done: boolean
  message: string | null
}

interface AgentLoopCallbacks {
  onToolsExecuted?: (toolCalls: ToolCall[], results: ToolResult[]) => void
  onMessage?: (message: string) => void
  onError?: (error: Error) => void
}

export interface AgentLoopResult {
  success: boolean
  message?: string
  threadId?: string
}

/**
 * Run the hybrid agent loop.
 *
 * Sends prompt to backend, executes returned tool calls on the song store,
 * and continues until the agent is done. Supports multi-turn via threadId.
 */
export async function runAgentLoop(
  prompt: string,
  song: Song,
  options?: {
    threadId?: string
    callbacks?: AgentLoopCallbacks
    abortSignal?: AbortSignal
  },
): Promise<AgentLoopResult> {
  const { threadId: existingThreadId, callbacks, abortSignal } = options ?? {}
  let threadId: string | null = existingThreadId ?? null

  console.log(
    `%c[AgentLoop] ╔══════════════════════════════════════════════════════════╗`,
    "color: #3F51B5; font-weight: bold"
  )
  console.log(
    `%c[AgentLoop] ║           HYBRID AGENT LOOP STARTED                      ║`,
    "color: #3F51B5; font-weight: bold"
  )
  console.log(
    `%c[AgentLoop] ╚══════════════════════════════════════════════════════════╝`,
    "color: #3F51B5; font-weight: bold"
  )
  console.log(`%c[AgentLoop] Thread ID: ${threadId ?? 'NEW'}`, "color: #3F51B5")
  console.log(`%c[AgentLoop] Prompt: "${prompt.substring(0, 100)}${prompt.length > 100 ? '...' : ''}"`, "color: #3F51B5")

  try {
    // Serialize current song state for agent context
    const songState = serializeSongState(song)
    const context = formatSongStateForPrompt(songState)
    console.log(`%c[AgentLoop] Song state: ${songState.trackCount} tracks, ${songState.tempo} BPM`, "color: #3F51B5")

    // Initial request
    console.log(`%c[AgentLoop] Sending initial request to backend...`, "color: #3F51B5")
    let response = await fetch(`${API_BASE}/api/agent/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        context,
        thread_id: threadId, // Continue existing thread if provided
      }),
      signal: abortSignal,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP ${response.status}`)
    }

    let result: AgentStepResponse = await response.json()
    threadId = result.thread_id
    console.log(`[HybridAgent] Initial response:`, result)

    // Agent loop
    while (!result.done) {
      if (abortSignal?.aborted) {
        throw new Error("Aborted")
      }

      if (result.tool_calls.length === 0) {
        console.log(`[HybridAgent] No tool calls but not done - breaking`)
        break
      }

      console.log(
        `%c[AgentLoop] ═══════════════════════════════════════════════════`,
        "color: #3F51B5; font-weight: bold"
      )
      console.log(
        `%c[AgentLoop] Executing ${result.tool_calls.length} tool call(s)...`,
        "color: #3F51B5; font-weight: bold"
      )
      // Execute tools on the frontend (may use async Magenta generation)
      const toolResults = await executeToolCalls(song, result.tool_calls)
      console.log(`%c[AgentLoop] Tool results:`, "color: #3F51B5", toolResults)
      callbacks?.onToolsExecuted?.(result.tool_calls, toolResults)

      // Resume the agent with tool results
      response = await fetch(`${API_BASE}/api/agent/step`, {
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

      result = await response.json()
    }

    // Agent completed
    console.log(
      `%c[AgentLoop] ╔══════════════════════════════════════════════════════════╗`,
      "color: #4CAF50; font-weight: bold"
    )
    console.log(
      `%c[AgentLoop] ║           AGENT LOOP COMPLETED SUCCESSFULLY              ║`,
      "color: #4CAF50; font-weight: bold"
    )
    console.log(
      `%c[AgentLoop] ╚══════════════════════════════════════════════════════════╝`,
      "color: #4CAF50; font-weight: bold"
    )
    if (result.message) {
      console.log(`%c[AgentLoop] Final message: "${result.message.substring(0, 100)}..."`, "color: #4CAF50")
      callbacks?.onMessage?.(result.message)
    }

    return {
      success: true,
      message: result.message ?? undefined,
      threadId: threadId ?? undefined,
    }
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error))
    console.error(
      `%c[AgentLoop] ╔══════════════════════════════════════════════════════════╗`,
      "color: #F44336; font-weight: bold"
    )
    console.error(
      `%c[AgentLoop] ║           AGENT LOOP FAILED                              ║`,
      "color: #F44336; font-weight: bold"
    )
    console.error(
      `%c[AgentLoop] ╚══════════════════════════════════════════════════════════╝`,
      "color: #F44336; font-weight: bold"
    )
    console.error(`%c[AgentLoop] Error: ${err.message}`, "color: #F44336", err)
    callbacks?.onError?.(err)
    return {
      success: false,
      message: err.message,
      threadId: threadId ?? undefined,
    }
  }
}
