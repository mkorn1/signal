import isEqual from "lodash/isEqual"
import React, { FC, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import { TrackEvent } from "../../track"
import { getEventController } from "./EventController"
import { Cell, Row } from "./EventList"
import { EventListInput } from "./EventListInput"

interface EventListItemProps {
  item: TrackEvent
  isSelected: boolean
  style?: React.CSSProperties
  onClick: (e: React.MouseEvent, ev: TrackEvent) => void
}

const equalEventListItemProps = (
  a: EventListItemProps,
  b: EventListItemProps,
) =>
  isEqual(a.item, b.item) &&
  a.isSelected === b.isSelected &&
  isEqual(a.style, b.style) &&
  a.onClick === b.onClick

export const EventListItem: FC<EventListItemProps> = React.memo(
  ({ item, isSelected, style, onClick }) => {
    const rootStore = useStores()
    const {
      pianoRollStore: { selectedTrack },
    } = rootStore

    const controller = getEventController(item)

    const onDelete = useCallback(
      (e: TrackEvent) => {
        selectedTrack?.removeEvent(e.id)
      },
      [rootStore],
    )

    const onChangeTick = useCallback(
      (value: number | string) => {
        if (typeof value === "string") {
          return
        }
        selectedTrack?.updateEvent(item.id, { tick: value })
      },
      [rootStore, item],
    )

    const onChangeGate = useCallback(
      (value: number | string) => {
        if (controller.gate === undefined) {
          return
        }
        const obj = controller.gate.update(value)
        selectedTrack?.updateEvent(item.id, obj)
      },
      [rootStore, item],
    )

    const onChangeValue = useCallback(
      (value: number | string) => {
        if (controller.value === undefined) {
          return
        }
        const obj = controller.value.update(value)
        selectedTrack?.updateEvent(item.id, obj)
      },
      [rootStore, item],
    )

    return (
      <Row
        style={style}
        onClick={useCallback((e: React.MouseEvent) => onClick(e, item), [item])}
        onKeyDown={useCallback(
          (e: React.KeyboardEvent) => {
            if (
              e.target === e.currentTarget &&
              (e.key === "Delete" || e.key === "Backspace")
            ) {
              onDelete(item)
              e.stopPropagation()
            }
          },
          [item],
        )}
        tabIndex={-1}
      >
        <Cell>
          <EventListInput
            value={item.tick}
            type="number"
            minValue={0}
            maxValue={Infinity}
            onChange={onChangeTick}
          />
        </Cell>
        <Cell
          style={{
            whiteSpace: "nowrap",
            textOverflow: "ellipsis",
            overflow: "hidden",
          }}
        >
          {controller.name}
        </Cell>
        <Cell>
          {controller.gate !== undefined && (
            <EventListInput {...controller.gate} onChange={onChangeGate} />
          )}
        </Cell>
        <Cell>
          {controller.value !== undefined && (
            <EventListInput {...controller.value} onChange={onChangeValue} />
          )}
        </Cell>
      </Row>
    )
  },
  equalEventListItemProps,
)
