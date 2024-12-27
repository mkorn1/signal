import { MouseEvent } from "react"
import {
  useArrangeEndSelection,
  useArrangeResizeSelection,
} from "../../../../actions"
import { Point } from "../../../../entities/geometry/Point"
import { MouseGesture } from "../../../../gesture/MouseGesture"
import { getClientPos } from "../../../../helpers/mouseEvent"
import { observeDrag } from "../../../../helpers/observeDrag"
import { useStores } from "../../../../hooks/useStores"

export const useCreateSelectionGesture = (): MouseGesture<
  [Point, Point],
  MouseEvent
> => {
  const {
    player,
    arrangeViewStore,
    arrangeViewStore: { trackTransform },
  } = useStores()

  const arrangeEndSelection = useArrangeEndSelection()
  const arrangeResizeSelection = useArrangeResizeSelection()

  return {
    onMouseDown(_e, startClientPos, startPosPx) {
      const startPos = trackTransform.getArrangePoint(startPosPx)
      arrangeViewStore.resetSelection()

      if (!player.isPlaying) {
        player.position = arrangeViewStore.quantizer.round(startPos.tick)
      }

      arrangeViewStore.selectedTrackIndex = Math.floor(startPos.trackIndex)

      observeDrag({
        onMouseMove: (e) => {
          const deltaPx = Point.sub(getClientPos(e), startClientPos)
          const selectionToPx = Point.add(startPosPx, deltaPx)
          arrangeResizeSelection(
            startPos,
            trackTransform.getArrangePoint(selectionToPx),
          )
        },
        onMouseUp: (e) => {
          arrangeEndSelection()
        },
      })
    },
  }
}
