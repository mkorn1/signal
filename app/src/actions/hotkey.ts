import {
  copyControlSelection,
  deleteControlSelection,
  pasteControlSelection,
} from "../actions/control"
import {
  useCopySelection,
  useDeleteSelection,
  usePasteSelection,
} from "../actions/selection"
import {
  isControlEventsClipboardData,
  isPianoNotesClipboardData,
} from "../clipboard/clipboardTypes"
import { useStores } from "../hooks/useStores"
import Clipboard from "../services/Clipboard"
import {
  useArrangeCopySelection,
  useArrangeDeleteSelection,
  useArrangePasteSelection,
} from "./arrangeView"
import {
  useCopyTempoSelection,
  useDeleteTempoSelection,
  usePasteTempoSelection,
} from "./tempo"

export const useCopySelectionGlobal = () => {
  const rootStore = useStores()
  const copySelection = useCopySelection()
  const arrangeCopySelection = useArrangeCopySelection()
  const copyTempoSelection = useCopyTempoSelection()

  return () => {
    switch (rootStore.router.path) {
      case "/track":
        if (rootStore.pianoRollStore.selectedNoteIds.length > 0) {
          copySelection()
        } else if (rootStore.controlStore.selectedEventIds.length > 0) {
          copyControlSelection(rootStore)()
        }
        break
      case "/arrange":
        arrangeCopySelection()
        break
      case "/tempo":
        copyTempoSelection()
        break
    }
  }
}

export const useCutSelectionGlobal = () => {
  const rootStore = useStores()
  const copySelection = useCopySelection()
  const deleteSelection = useDeleteSelection()
  const arrangeCopySelection = useArrangeCopySelection()
  const arrangeDeleteSelection = useArrangeDeleteSelection()
  const copyTempoSelection = useCopyTempoSelection()
  const deleteTempoSelection = useDeleteTempoSelection()

  return () => {
    switch (rootStore.router.path) {
      case "/track":
        if (rootStore.pianoRollStore.selectedNoteIds.length > 0) {
          copySelection()
          deleteSelection()
        } else if (rootStore.controlStore.selectedEventIds.length > 0) {
          copyControlSelection(rootStore)()
          deleteControlSelection(rootStore)()
        }
        break
      case "/arrange":
        arrangeCopySelection()
        arrangeDeleteSelection()
        break
      case "/tempo":
        copyTempoSelection()
        deleteTempoSelection()
        break
    }
  }
}

export const usePasteSelectionGlobal = () => {
  const rootStore = useStores()
  const pasteSelection = usePasteSelection()
  const arrangePasteSelection = useArrangePasteSelection()
  const pasteTempoSelection = usePasteTempoSelection()

  return () => {
    switch (rootStore.router.path) {
      case "/track":
        const text = Clipboard.readText()
        if (!text || text.length === 0) {
          return
        }
        const obj = JSON.parse(text)
        if (isPianoNotesClipboardData(obj)) {
          pasteSelection()
        } else if (isControlEventsClipboardData(obj)) {
          pasteControlSelection(rootStore)()
        }
        break
      case "/arrange":
        arrangePasteSelection()
        break
      case "/tempo":
        pasteTempoSelection()
    }
  }
}
