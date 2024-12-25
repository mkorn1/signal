import { Point } from "../../../entities/geometry/Point"
import { Range } from "../../../entities/geometry/Range"
import { TempoCoordTransform } from "../../../entities/transform/TempoCoordTransform"
import { MouseGesture } from "../../../gesture/MouseGesture"
import { isEventInRange } from "../../../helpers/filterEvents"
import { getClientPos } from "../../../helpers/mouseEvent"
import { observeDrag } from "../../../helpers/observeDrag"
import { useStores } from "../../../hooks/useStores"
import { isSetTempoEvent } from "../../../track"

export const useCreateSelectionGesture = (): MouseGesture<
  [Point, TempoCoordTransform]
> => {
  const {
    song: { conductorTrack },
    tempoEditorStore,
  } = useStores()

  return {
    onMouseDown(e, startPoint, transform) {
      if (conductorTrack === undefined) {
        return
      }

      const start = transform.fromPosition(startPoint)
      const startClientPos = getClientPos(e)

      tempoEditorStore.selectedEventIds = []

      tempoEditorStore.selection = {
        fromTick: start.tick,
        toTick: start.tick,
      }

      observeDrag({
        onMouseMove: (e) => {
          const posPx = getClientPos(e)
          const deltaPx = Point.sub(posPx, startClientPos)
          const local = Point.add(startPoint, deltaPx)
          const end = transform.fromPosition(local)
          tempoEditorStore.selection = {
            fromTick: Math.min(start.tick, end.tick),
            toTick: Math.max(start.tick, end.tick),
          }
        },
        onMouseUp: (e) => {
          const { selection } = tempoEditorStore
          if (selection === null) {
            return
          }

          tempoEditorStore.selectedEventIds = conductorTrack.events
            .filter(isSetTempoEvent)
            .filter(
              isEventInRange(
                Range.create(selection.fromTick, selection.toTick),
              ),
            )
            .map((e) => e.id)

          tempoEditorStore.selection = null
        },
      })
    },
  }
}
