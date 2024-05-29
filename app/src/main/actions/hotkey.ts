// global keyboard shortcuts or menu actions

import {
  isControlEventsClipboardData,
  isPianoNotesClipboardData,
} from "../clipboard/clipboardTypes"
import Clipboard from "../services/Clipboard"
import RootStore from "../stores/RootStore"
import {
  arrangeCopySelection,
  arrangeDeleteSelection,
  arrangePasteSelection,
} from "./arrangeView"
import {
  copyControlSelection,
  deleteControlSelection,
  pasteControlSelection,
} from "./control"
import { copySelection, deleteSelection, pasteSelection } from "./selection"
import {
  copyTempoSelection,
  deleteTempoSelection,
  pasteTempoSelection,
} from "./tempo"

export const copySelectionGlobal = (rootStore: RootStore) => () => {
  switch (rootStore.router.path) {
    case "/track":
      if (rootStore.pianoRollStore.selectedNoteIds.length > 0) {
        copySelection(rootStore)()
      } else if (rootStore.controlStore.selectedEventIds.length > 0) {
        copyControlSelection(rootStore)()
      }
      break
    case "/arrange":
      arrangeCopySelection(rootStore)()
      break
    case "/tempo":
      copyTempoSelection(rootStore)()
      break
  }
}

export const cutSelectionGlobal = (rootStore: RootStore) => () => {
  switch (rootStore.router.path) {
    case "/track":
      if (rootStore.pianoRollStore.selectedNoteIds.length > 0) {
        copySelection(rootStore)()
        deleteSelection(rootStore)()
      } else if (rootStore.controlStore.selectedEventIds.length > 0) {
        copyControlSelection(rootStore)()
        deleteControlSelection(rootStore)()
      }
      break
    case "/arrange":
      arrangeCopySelection(rootStore)()
      arrangeDeleteSelection(rootStore)()
      break
    case "/tempo":
      copyTempoSelection(rootStore)()
      deleteTempoSelection(rootStore)()
      break
  }
}

export const pasteSelectionGlobal = (rootStore: RootStore) => () => {
  switch (rootStore.router.path) {
    case "/track":
      const text = Clipboard.readText()
      if (!text || text.length === 0) {
        return
      }
      const obj = JSON.parse(text)
      if (isPianoNotesClipboardData(obj)) {
        pasteSelection(rootStore)()
      } else if (isControlEventsClipboardData(obj)) {
        pasteControlSelection(rootStore)()
      }
      break
    case "/arrange":
      arrangePasteSelection(rootStore)()
      break
    case "/tempo":
      pasteTempoSelection(rootStore)()
  }
}
