import styled from "@emotion/styled"
import useComponentSize from "@rehooks/component-size"
import { observer } from "mobx-react-lite"
import React, { FC, useCallback, useMemo, useRef } from "react"
import { FixedSizeList, ListChildComponentProps } from "react-window"
import { Layout } from "../../Constants"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { TrackEvent } from "../../track"
import { EventListItem } from "./EventListItem"

const Container = styled.div`
  width: 100%;
  height: 100%;
`

const Header = styled.div`
  height: ${Layout.rulerHeight};
  border-bottom: 1px solid ${({ theme }) => theme.dividerColor};
  /* scroll bar width */
  padding-right: 14px;
`

export const Row = styled.div`
  display: flex;
  outline: none;

  &:focus {
    background: ${({ theme }) => theme.highlightColor};
  }
`

export const Cell = styled.div`
  padding: 0.5rem;

  &:focus-within {
    background: ${({ theme }) => theme.highlightColor};
  }
`

export const widthForCell = (index: number) =>
  ["5em", "6em", "4em", "4em"][index]

const EventList: FC = observer(() => {
  const rootStore = useStores()
  const {
    pianoRollStore: { selectedTrack, selectedNoteIds: selectedEventIds = [] },
  } = rootStore

  const events = useMemo(() => {
    const { events = [] } = selectedTrack || {}
    if (selectedEventIds.length > 0) {
      return events.filter((event) => selectedEventIds.indexOf(event.id) >= 0)
    }
    return events
  }, [selectedTrack?.events, selectedEventIds])

  const ref = useRef<HTMLDivElement>(null)
  const size = useComponentSize(ref)

  return (
    <Container ref={ref}>
      <Header>
        <Row>
          <Cell style={{ width: widthForCell(0) }}>
            <Localized name="tick" />
          </Cell>
          <Cell style={{ width: widthForCell(1), flexGrow: 1 }}>
            <Localized name="event" />
          </Cell>
          <Cell style={{ width: widthForCell(2) }}>
            <Localized name="duration" />
          </Cell>
          <Cell style={{ width: widthForCell(3) }}>
            <Localized name="value" />
          </Cell>
        </Row>
      </Header>
      <FixedSizeList
        height={size.height - Layout.rulerHeight}
        itemCount={events.length}
        itemSize={35}
        width={size.width}
        itemData={{ events }}
        itemKey={(index) => events[index].id}
      >
        {ItemRenderer}
      </FixedSizeList>
    </Container>
  )
})

const ItemRenderer = ({ index, style, data }: ListChildComponentProps) => {
  const { events, selectedEventIds = [], setSelectedEventIds } = data
  const e = events[index]

  const onClickRow = useCallback((e: React.MouseEvent, ev: TrackEvent) => {
    if (e.ctrlKey || e.metaKey) {
      setSelectedEventIds?.((ids: number[]) => [...ids, ev.id])
    } else {
      setSelectedEventIds?.([ev.id])
    }
  }, [])

  return (
    <EventListItem
      style={style}
      item={e}
      key={e.id}
      isSelected={selectedEventIds.includes(e.id)}
      onClick={onClickRow}
    />
  )
}

export default EventList
