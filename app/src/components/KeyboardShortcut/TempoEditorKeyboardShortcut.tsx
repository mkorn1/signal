import { observer } from "mobx-react-lite"
import { FC } from "react"
import {
  useCopyTempoSelection,
  useDeleteTempoSelection,
  useDuplicateTempoSelection,
  usePasteTempoSelection,
  useResetTempoSelection,
} from "../../actions/tempo"
import { isTempoEventsClipboardData } from "../../clipboard/clipboardTypes"
import { useStores } from "../../hooks/useStores"
import clipboard from "../../services/Clipboard"
import { KeyboardShortcut } from "./KeyboardShortcut"
import { isFocusable } from "./isFocusable"

export const TempoEditorKeyboardShortcut: FC = observer(() => {
  const rootStore = useStores()
  const { tempoEditorStore } = rootStore
  const resetTempoSelection = useResetTempoSelection()
  const deleteTempoSelection = useDeleteTempoSelection()
  const copyTempoSelection = useCopyTempoSelection()
  const duplicateTempoSelection = useDuplicateTempoSelection()
  const pasteTempoSelection = usePasteTempoSelection()

  return (
    <KeyboardShortcut
      actions={[
        {
          code: "Digit1",
          run: () => (tempoEditorStore.mouseMode = "pencil"),
        },
        {
          code: "Digit2",
          run: () => (tempoEditorStore.mouseMode = "selection"),
        },
        { code: "Escape", run: resetTempoSelection },
        { code: "Backspace", run: deleteTempoSelection },
        { code: "Delete", run: deleteTempoSelection },
        {
          code: "KeyC",
          metaKey: true,
          run: () => copyTempoSelection(),
        },
        {
          code: "KeyX",
          metaKey: true,
          run: () => {
            {
              copyTempoSelection()
              deleteTempoSelection()
            }
          },
        },
        {
          code: "KeyD",
          metaKey: true,
          run: duplicateTempoSelection,
        },
      ]}
      onPaste={(e) => {
        if (e.target !== null && isFocusable(e.target)) {
          return
        }

        const text = clipboard.readText()

        if (!text || text.length === 0) {
          return
        }

        const obj = JSON.parse(text)

        if (isTempoEventsClipboardData(obj)) {
          pasteTempoSelection()
        }
      }}
    />
  )
})
