import { observer } from "mobx-react-lite"
import { FC } from "react"
import {
  useArrangeCopySelection,
  useArrangeDeleteSelection,
  useArrangeDuplicateSelection,
  useArrangePasteSelection,
} from "../../actions"
import { useStores } from "../../hooks/useStores"
import { KeyboardShortcut } from "./KeyboardShortcut"

const SCROLL_DELTA = 24

export const ArrangeViewKeyboardShortcut: FC = observer(() => {
  const { arrangeViewStore } = useStores()
  const arrangeDeleteSelection = useArrangeDeleteSelection()
  const arrangeCopySelection = useArrangeCopySelection()
  const arrangePasteSelection = useArrangePasteSelection()
  const arrangeDuplicateSelection = useArrangeDuplicateSelection()

  return (
    <KeyboardShortcut
      actions={[
        { code: "Escape", run: () => arrangeViewStore.resetSelection() },
        { code: "Delete", run: () => arrangeDeleteSelection() },
        { code: "Backspace", run: () => arrangeDeleteSelection() },
        {
          code: "ArrowUp",
          metaKey: true,
          run: () => arrangeViewStore.scrollBy(0, SCROLL_DELTA),
        },
        {
          code: "ArrowDown",
          metaKey: true,
          run: () => arrangeViewStore.scrollBy(0, -SCROLL_DELTA),
        },
        {
          code: "KeyT",
          run: () => (arrangeViewStore.openTransposeDialog = true),
        },
        {
          code: "KeyD",
          metaKey: true,
          run: () => arrangeDuplicateSelection(),
        },
      ]}
      onCut={() => {
        arrangeCopySelection()
        arrangeDeleteSelection()
      }}
      onCopy={() => arrangeCopySelection()}
      onPaste={() => arrangePasteSelection()}
    />
  )
})
