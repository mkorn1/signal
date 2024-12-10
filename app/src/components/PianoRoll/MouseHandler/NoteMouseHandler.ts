import { useState } from "react"
import { observeDrag } from "../../../helpers/observeDrag"
import { useStores } from "../../../hooks/useStores"
import RootStore from "../../../stores/RootStore"
import { usePencilGesture } from "./PencilMouseHandler"
import { useSelectionGesture } from "./SelectionMouseHandler"

export type MouseGesture = (rootStore: RootStore) => (e: MouseEvent) => void

export const useNoteMouseHandler = () => {
  const rootStore = useStores()
  const [isMouseDown, setMouseDown] = useState(false)
  const pencilGesture = usePencilGesture()
  const selectionGesture = useSelectionGesture()
  const currentGesture = (() => {
    switch (rootStore.pianoRollStore.mouseMode) {
      case "pencil":
        return pencilGesture
      case "selection":
        return selectionGesture
    }
  })()

  // mousedown 以降に行う MouseAction を返す
  // Returns a MouseAction to do after MouseDown
  function actionForMouseDown(e: MouseEvent): MouseGesture | null {
    // 共通の action
    // Common Action

    // wheel drag to start scrolling
    if (e.button === 1) {
      return dragScrollAction
    }

    // 右ダブルクリック
    // Right Double-click
    if (e.button === 2 && e.detail % 2 === 0) {
      return changeToolAction
    }

    return currentGesture.onMouseDown(e)
  }

  return {
    onMouseDown(ev: React.MouseEvent) {
      const e = ev.nativeEvent
      setMouseDown(true)
      actionForMouseDown(e)?.(rootStore)(e)
    },

    onMouseMove(ev: React.MouseEvent) {
      const e = ev.nativeEvent
      if (!isMouseDown) {
        const cursor = currentGesture.getCursor(e)
        rootStore.pianoRollStore.notesCursor = cursor
      }
    },

    onMouseUp(_ev: React.MouseEvent) {
      setMouseDown(false)
    },
  }
}

const dragScrollAction: MouseGesture =
  ({ pianoRollStore }) =>
  () => {
    observeDrag({
      onMouseMove: (e: MouseEvent) => {
        pianoRollStore.scrollBy(e.movementX, e.movementY)
        pianoRollStore.autoScroll = false
      },
    })
  }

const changeToolAction: MouseGesture =
  ({ pianoRollStore }) =>
  () => {
    pianoRollStore.toggleTool()
    pianoRollStore.notesCursor = "crosshair"
  }
