import { useState } from "react"
import { observeDrag } from "../../../helpers/observeDrag"
import { useStores } from "../../../hooks/useStores"
import RootStore from "../../../stores/RootStore"
import { usePencilGesture } from "./PencilMouseHandler"
import { useSelectionGesture } from "./SelectionMouseHandler"

export type MouseGesture = (rootStore: RootStore) => (e: MouseEvent) => void

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

    onMouseUp(_ev: React.MouseEvent) {
      setMouseDown(false)
    },
  }
}

const useDragScrollGesture = () => {
  const { pianoRollStore } = useStores()
  return {
    onMouseDown(_e: MouseEvent) {
      observeDrag({
        onMouseMove: (e: MouseEvent) => {
          pianoRollStore.scrollBy(e.movementX, e.movementY)
          pianoRollStore.autoScroll = false
        },
      })
    },
  }
}

const useChangeToolGesture = () => {
  const { pianoRollStore } = useStores()
  return {
    onMouseDown(_e: MouseEvent) {
      pianoRollStore.toggleTool()
      pianoRollStore.notesCursor = "crosshair"
    },
  }
}
