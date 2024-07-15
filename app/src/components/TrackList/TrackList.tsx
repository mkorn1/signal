import styled from "@emotion/styled"
import { observer } from "mobx-react-lite"
import { FC } from "react"
import { isNotNull } from "../../helpers/array"
import { useStores } from "../../hooks/useStores"
import { DraggableList } from "../ControlSettingDialog/DraggableList"
import { AddTrackButton } from "./AddTrackButton"
import { TrackListItem } from "./TrackListItem"

const List = styled.div`
  overflow-y: auto;
  background: ${({ theme }) => theme.backgroundColor};
  min-width: 14rem;
  flex-grow: 1;
`

export const TrackList: FC = observer(() => {
  const { song } = useStores()

  return (
    <List>
      <DraggableList
        items={song.tracks
          .map((t, i) => (t.isConductorTrack ? null : i))
          .filter(isNotNull)}
        getItemId={(trackId) => trackId}
        onItemMoved={(from, to) => {
          song.moveTrack(from, to)
        }}
        render={(trackId) => <TrackListItem key={trackId} trackId={trackId} />}
      ></DraggableList>
      <AddTrackButton />
    </List>
  )
})
