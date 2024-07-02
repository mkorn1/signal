import { Point, pointAdd } from "../../../../entities/geometry/Point"
import { ControlSelection } from "../../../../entities/selection/ControlSelection"
import { ControlCoordTransform } from "../../../../entities/transform/ControlCoordTransform"
import { observeDrag2 } from "../../../../helpers/observeDrag"
import RootStore from "../../../../stores/RootStore"

export const handleCreateSelectionDrag =
  (rootStore: RootStore) =>
  (
    e: MouseEvent,
    startPoint: Point,
    controlTransform: ControlCoordTransform,
    getControllerEventIdsInSelection: (selection: ControlSelection) => number[],
  ) => {
    const {
      pianoRollStore,
      controlStore,
      controlStore: { quantizer },
      player,
    } = rootStore
    controlStore.selectedEventIds = []

    const startTick = quantizer.round(controlTransform.getTicks(startPoint.x))

    pianoRollStore.selection = null
    pianoRollStore.selectedNoteIds = []

    if (!player.isPlaying) {
      player.position = startTick
    }

    controlStore.selection = {
      fromTick: startTick,
      toTick: startTick,
    }

    observeDrag2(e, {
      onMouseMove: (_e, delta) => {
        const local = pointAdd(startPoint, delta)
        const endTick = quantizer.round(controlTransform.getTicks(local.x))
        controlStore.selection = {
          fromTick: Math.min(startTick, endTick),
          toTick: Math.max(startTick, endTick),
        }
      },
      onMouseUp: (_e) => {
        const { selection } = controlStore
        if (selection === null) {
          return
        }

        controlStore.selectedEventIds =
          getControllerEventIdsInSelection(selection)
        controlStore.selection = null
      },
    })
  }
