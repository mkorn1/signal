/**
 * Stem Agent - Conversational audio stem generation.
 *
 * Uses a chat-based approach to gather user preferences before generating
 * audio stems via Stable Audio.
 */

export { runStemAgentLoop } from "./stemAgentLoop"
export type { StemAgentCallbacks, StemAgentResult } from "./stemAgentLoop"
export { executeStemToolCall } from "./stemToolExecutor"
export type {
    GeneratedStem,
    StemToolCall,
    StemToolResult
} from "./stemToolExecutor"

