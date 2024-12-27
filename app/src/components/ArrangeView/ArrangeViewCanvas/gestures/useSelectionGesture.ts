import { MouseEvent, useCallback } from "react"
import { Point } from "../../../../entities/geometry/Point"
import { Rect } from "../../../../entities/geometry/Rect"
import { MouseGesture } from "../../../../gesture/MouseGesture"
import { getClientPos } from "../../../../helpers/mouseEvent"
import { useStores } from "../../../../hooks/useStores"
import { useCreateSelectionGesture } from "./useCreateSelectionGesture"
import { useMoveSelectionGesture } from "./useMoveSelectionGesture"

export const useSelectionGesture = (): MouseGesture<[], MouseEvent> => {
  const {
    arrangeViewStore: { scrollLeft, scrollTop, selectionRect },
  } = useStores()

  const moveSelectionGesture = useMoveSelectionGesture()
  const createSelectionGesture = useCreateSelectionGesture()

  const onMouseDown = useCallback(
    (e: MouseEvent) => {
      const startPosPx: Point = {
        x: e.nativeEvent.offsetX + scrollLeft,
        y: e.nativeEvent.offsetY + scrollTop,
      }
      const startClientPos = getClientPos(e.nativeEvent)

      const isSelectionSelected =
        selectionRect != null && Rect.containsPoint(selectionRect, startPosPx)

      if (isSelectionSelected) {
        moveSelectionGesture.onMouseDown(e, startClientPos, selectionRect)
      } else {
        createSelectionGesture.onMouseDown(e, startClientPos, startPosPx)
      }
    },
    [
      selectionRect,
      scrollLeft,
      scrollTop,
      moveSelectionGesture,
      createSelectionGesture,
    ],
  )

  return {
    onMouseDown,
  }
}
