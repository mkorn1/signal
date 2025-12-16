/**
 * Global state for music properties set by the AI agent.
 * This allows the UI to track and animate changes made by the agent.
 */

import { atom } from "jotai"
import type { Scale } from "../entities/scale/Scale"

export interface AgentKeySignature {
  readonly key: number // 0 is C, 1 is C#, 2 is D, etc.
  readonly scale: Scale
}

// Key names for display
const KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

export function formatKeySignature(key: AgentKeySignature | null): string {
  if (!key) return "--"
  const keyName = KEY_NAMES[key.key]
  const scaleSuffix = key.scale === "minor" ? "m" : key.scale === "major" ? "" : ` ${key.scale}`
  return `${keyName}${scaleSuffix}`
}

// Atom for the key signature set by the agent
export const agentKeySignatureAtom = atom<AgentKeySignature | null>(null)

// Atom to track when values were last updated by agent (for animations)
export const agentBpmUpdatedAtom = atom<number>(0) // timestamp
export const agentKeyUpdatedAtom = atom<number>(0) // timestamp

