import { useState } from "react"
import { useStores } from "../../../hooks/useStores"
import { usePencilGesture } from "./PencilMouseHandler"
import { useSelectionGesture } from "./SelectionMouseHandler"
import { useChangeToolGesture } from "./gestures/useChangeToolGesture"
import { useDragScrollGesture } from "./gestures/useDragScrollGesture"

export interface MouseGesture<Params extends any[] = []> {
  onMouseDown(e: MouseEvent, ...params: Params): void
  onMouseMove?(e: MouseEvent): void
  onMouseUp?(e: MouseEvent): void
}

export const useNoteMouseGesture = () => {
  const rootStore = useStores()
  const {
    pianoRollStore,
    pianoRollStore: { mouseMode },
  } = rootStore
  const [isMouseDown, setMouseDown] = useState(false)
  const pencilGesture = usePencilGesture()
  const selectionGesture = useSelectionGesture()
  const currentGesture = (() => {
    switch (mouseMode) {
      case "pencil":
        return pencilGesture
      case "selection":
        return selectionGesture
    }
  })()
  const dragScrollAction = useDragScrollGesture()
  const changeToolAction = useChangeToolGesture()

  function getGestureForMouseDown(e: MouseEvent) {
    // Common Action

    // wheel drag to start scrolling
    if (e.button === 1) {
      return dragScrollAction
    }

    // Right Double-click
    if (e.button === 2 && e.detail % 2 === 0) {
      return changeToolAction
    }

    return currentGesture
  }

  return {
    onMouseDown(ev: React.MouseEvent) {
      const e = ev.nativeEvent
      setMouseDown(true)
      getGestureForMouseDown(e).onMouseDown(e)
    },

    onMouseMove(ev: React.MouseEvent) {
      const e = ev.nativeEvent
      if (!isMouseDown) {
        const cursor = currentGesture.getCursor(e)
        pianoRollStore.notesCursor = cursor
      }
    },

    onMouseUp() {
      setMouseDown(false)
    },
  }
}
