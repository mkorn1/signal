import { useAtomValue, useSetAtom } from "jotai"
import { focusAtom } from "jotai-optics"
import { atomWithStorage } from "jotai/utils"
import { Language } from "../localize/useLocalization"
import { ThemeType } from "../theme/Theme"

export type AgentType = "llm" | "composition_agent" | "hybrid" | "hybrid_legacy" | "deep_agent_2"

export function useSettings() {
  return {
    get language() {
      return useAtomValue(languageAtom)
    },
    get showNoteLabels() {
      return useAtomValue(showNoteLabelsAtom)
    },
    get themeType() {
      return useAtomValue(themeTypeAtom)
    },
    get agentType() {
      return useAtomValue(agentTypeAtom)
    },
    setLanguage: useSetAtom(languageAtom),
    setShowNoteLabels: useSetAtom(showNoteLabelsAtom),
    setThemeType: useSetAtom(themeTypeAtom),
    setAgentType: useSetAtom(agentTypeAtom),
  }
}

// atoms with storage
const settingStorageAtom = atomWithStorage<{
  language: Language | null
  showNoteLabels: boolean
}>("SettingStore", {
  language: null,
  showNoteLabels: true,
})
const themeStorageAtom = atomWithStorage<{
  themeType: ThemeType
}>("ThemeStore", {
  themeType: "dark",
})
const agentTypeAtom = atomWithStorage<AgentType>("ai_chat_agent_type", "hybrid")

// focused atoms
const languageAtom = focusAtom(settingStorageAtom, (optic) =>
  optic.prop("language"),
)
const showNoteLabelsAtom = focusAtom(settingStorageAtom, (optic) =>
  optic.prop("showNoteLabels"),
)
const themeTypeAtom = focusAtom(themeStorageAtom, (optic) =>
  optic.prop("themeType"),
)
