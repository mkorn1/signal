import { ControllerEvent, PitchBendEvent } from "midifile-ts"
import { observer } from "mobx-react-lite"
import React, { MouseEventHandler, useCallback, useMemo } from "react"
import { createOrUpdateControlEventsValue } from "../../../actions/control"
import { Point } from "../../../entities/geometry/Point"
import { Rect } from "../../../entities/geometry/Rect"
import { ControlCoordTransform } from "../../../entities/transform/ControlCoordTransform"
import { filterEventsWithRange } from "../../../helpers/filterEvents"
import { ValueEventType, createValueEvent } from "../../../helpers/valueEvent"
import { useContextMenu } from "../../../hooks/useContextMenu"
import { useStores } from "../../../hooks/useStores"
import { pointToCircleRect } from "../../../stores/TempoEditorStore"
import { TrackEventOf } from "../../../track"
import { ControlSelectionContextMenu } from "../ControlSelectionContextMenu"
import { handleCreateSelectionDrag } from "../Graph/MouseHandler/handleCreateSelectionDrag"
import { handlePencilMouseDown } from "../Graph/MouseHandler/handlePencilMouseDown"
import { handleSelectionDragEvents } from "../Graph/MouseHandler/handleSelectionDragEvents"
import { GraphAxis } from "./GraphAxis"
import { LineGraphCanvas } from "./LineGraphCanvas"

export interface ItemValue {
  tick: number
  value: number
}

export interface LineGraphProps<T extends ControllerEvent | PitchBendEvent> {
  width: number
  height: number
  maxValue: number
  events: TrackEventOf<T>[]
  eventType: ValueEventType
  lineWidth?: number
  circleRadius?: number
  axis: number[]
  axisLabelFormatter?: (value: number) => string
}

const LineGraph = observer(
  <T extends ControllerEvent | PitchBendEvent>({
    maxValue,
    events,
    eventType,
    width,
    height,
    lineWidth = 2,
    circleRadius = 4,
    axis,
    axisLabelFormatter = (v) => v.toString(),
  }: LineGraphProps<T>) => {
    const rootStore = useStores()
    const {
      controlStore: { scrollLeft, transform, cursor, mouseMode },
    } = rootStore

    const controlTransform = useMemo(
      () => new ControlCoordTransform(transform, maxValue, height, lineWidth),
      [transform.horizontalId, maxValue, height, lineWidth],
    )

    const items = events.map((e) => ({
      id: e.id,
      ...controlTransform.toPosition(e.tick, e.value),
    }))

    const controlPoints = items.map((p) => ({
      ...pointToCircleRect(p, circleRadius),
      id: p.id,
    }))

    const hitTest = useCallback(
      (point: Point) =>
        controlPoints.find((r) => Rect.containsPoint(r, point))?.id,
      [controlPoints],
    )

    const getLocal = (e: MouseEvent): Point => ({
      x: e.offsetX + scrollLeft,
      y: e.offsetY,
    })

    const pencilMouseDown: MouseEventHandler = useCallback(
      (ev) => {
        const local = getLocal(ev.nativeEvent)

        handlePencilMouseDown(rootStore)(
          ev.nativeEvent,
          local,
          controlTransform,
          eventType,
        )
      },
      [rootStore, scrollLeft, controlTransform, eventType],
    )

    const selectionMouseDown: MouseEventHandler = useCallback(
      (ev) => {
        const local = getLocal(ev.nativeEvent)
        const hitEventId = hitTest(local)

        if (hitEventId !== undefined) {
          handleSelectionDragEvents(rootStore)(
            ev.nativeEvent,
            hitEventId,
            local,
            controlTransform,
            eventType,
          )
        } else {
          handleCreateSelectionDrag(rootStore)(
            ev.nativeEvent,
            local,
            controlTransform,
            (s) =>
              filterEventsWithRange(events, s.fromTick, s.toTick).map(
                (e) => e.id,
              ),
          )
        }
      },
      [rootStore, controlTransform, scrollLeft, events, eventType, hitTest],
    )

    const onMouseDown =
      mouseMode === "pencil" ? pencilMouseDown : selectionMouseDown

    const onClickAxis = useCallback(
      (value: number) => {
        const event = createValueEvent(eventType)(value)
        createOrUpdateControlEventsValue(rootStore)(event)
      },
      [eventType],
    )

    const { onContextMenu, menuProps } = useContextMenu()

    return (
      <div
        style={{
          display: "flex",
        }}
      >
        <GraphAxis
          values={axis}
          valueFormatter={axisLabelFormatter}
          onClick={onClickAxis}
        />
        <LineGraphCanvas
          style={{ cursor }}
          onMouseDown={onMouseDown}
          onContextMenu={onContextMenu}
          width={width}
          height={height}
          maxValue={maxValue}
          controlPoints={controlPoints}
          items={items}
        />
        <ControlSelectionContextMenu {...menuProps} />
      </div>
    )
  },
)

export default React.memo(LineGraph)
