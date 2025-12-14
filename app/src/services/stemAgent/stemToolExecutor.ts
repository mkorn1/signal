/**
 * Stem tool executor for the conversational stem agent.
 * Handles the generateStems tool call by invoking the audio generation API.
 */

import { aiBackend } from "../aiBackend"

export interface StemToolCall {
  id: string
  name: string
  args: {
    style: string
    tempo?: number
    instruments?: string[]
  }
}

export interface StemToolResult {
  id: string
  result: string // JSON string
}

export interface GeneratedStem {
  name: string
  audioData: string // base64 encoded audio
  channel: number
  programNumber: number
}

/**
 * Execute a stem tool call (generateStems).
 *
 * This calls the backend to actually generate the audio stems using
 * the parameters provided by the agent.
 */
export async function executeStemToolCall(
  toolCall: StemToolCall,
  onProgress?: (instrument: string, status: string) => void,
): Promise<{ result: StemToolResult; stems: GeneratedStem[] }> {
  const { id, name, args } = toolCall

  if (name !== "generateStems") {
    return {
      result: {
        id,
        result: JSON.stringify({ error: `Unknown tool: ${name}` }),
      },
      stems: [],
    }
  }

  console.log("[StemAgent] Executing generateStems:", args)

  try {
    // The style prompt from the agent should already be detailed and include BPM
    // We use it directly since it was crafted based on the conversation
    const style = args.style || "electronic music, 120 BPM, energetic mood"
    const tempo = args.tempo || 120
    const instruments = args.instruments || ["melody", "drums", "bass", "keys"]

    // The style prompt should already be rich and detailed from the agent
    // Just add instrument info if not already implied
    const prompt = `${style}, with ${instruments.join(", ")}`

    console.log("[StemAgent] Generation prompt:", prompt)

    // Use the per-instrument agent to generate stems
    const response = await aiBackend.generate({
      prompt,
      agentType: "per_instrument",
    })

    if (!response.tracks || response.tracks.length === 0) {
      return {
        result: {
          id,
          result: JSON.stringify({
            error: "No tracks were generated",
            style,
            tempo,
            instruments,
          }),
        },
        stems: [],
      }
    }

    // Convert tracks to generated stems
    const stems: GeneratedStem[] = response.tracks.map((track) => ({
      name: track.name,
      audioData: track.audioData || "",
      channel: track.channel,
      programNumber: track.programNumber || 0,
    }))

    console.log(
      "[StemAgent] Generated stems:",
      stems.map((s) => s.name),
    )

    return {
      result: {
        id,
        result: JSON.stringify({
          success: true,
          trackCount: stems.length,
          tracks: stems.map((s) => s.name),
          style,
          tempo,
          instruments,
        }),
      },
      stems,
    }
  } catch (error) {
    console.error("[StemAgent] Generation failed:", error)
    const errorMessage =
      error instanceof Error ? error.message : "Generation failed"

    return {
      result: {
        id,
        result: JSON.stringify({
          error: errorMessage,
          style: args.style,
          tempo: args.tempo,
          instruments: args.instruments,
        }),
      },
      stems: [],
    }
  }
}
