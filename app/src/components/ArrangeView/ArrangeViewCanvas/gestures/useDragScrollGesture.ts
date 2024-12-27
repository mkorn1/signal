import { useCallback } from "react"
import { Point } from "../../../../entities/geometry/Point"
import { MouseGesture } from "../../../../gesture/MouseGesture"
import { observeDrag } from "../../../../helpers/observeDrag"
import { useStores } from "../../../../hooks/useStores"

export const useDragScrollGesture = (): MouseGesture<[], React.MouseEvent> => {
  const {
    arrangeViewStore,
    arrangeViewStore: { scrollLeft, scrollTop },
  } = useStores()

  const setScrollLeft = useCallback((v: number) => {
    arrangeViewStore.setScrollLeftInPixels(v)
    arrangeViewStore.autoScroll = false
  }, [])

  const setScrollTop = useCallback(
    (v: number) => arrangeViewStore.setScrollTop(v),
    [],
  )

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      function createPoint(e: MouseEvent) {
        return { x: e.clientX, y: e.clientY }
      }
      const startPos = createPoint(e.nativeEvent)

      observeDrag({
        onMouseMove(e) {
          const pos = createPoint(e)
          const delta = Point.sub(pos, startPos)
          setScrollLeft(Math.max(0, scrollLeft - delta.x))
          setScrollTop(Math.max(0, scrollTop - delta.y))
        },
      })
    },
    [scrollLeft, scrollTop, setScrollLeft, setScrollTop],
  )
  return {
    onMouseDown,
  }
}
