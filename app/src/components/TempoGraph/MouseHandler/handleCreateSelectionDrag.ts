import { IPoint, pointAdd, pointSub } from "../../../entities/geometry/Point"
import { TempoCoordTransform } from "../../../entities/transform/TempoCoordTransform"
import { filterEventsWithRange } from "../../../helpers/filterEvents"
import { getClientPos } from "../../../helpers/mouseEvent"
import { observeDrag } from "../../../helpers/observeDrag"
import RootStore from "../../../stores/RootStore"
import { isSetTempoEvent } from "../../../track"

export const handleCreateSelectionDrag =
  (rootStore: RootStore) =>
  (e: MouseEvent, startPoint: IPoint, transform: TempoCoordTransform) => {
    const {
      song: { conductorTrack },
      tempoEditorStore,
    } = rootStore

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
        const deltaPx = pointSub(posPx, startClientPos)
        const local = pointAdd(startPoint, deltaPx)
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

        tempoEditorStore.selectedEventIds = filterEventsWithRange(
          conductorTrack.events.filter(isSetTempoEvent),
          selection.fromTick,
          selection.toTick,
        ).map((e) => e.id)

        tempoEditorStore.selection = null
      },
    })
  }
