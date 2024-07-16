import styled from "@emotion/styled"
import Headset from "mdi-react/HeadphonesIcon"
import Layers from "mdi-react/LayersIcon"
import VolumeUp from "mdi-react/VolumeHighIcon"
import VolumeOff from "mdi-react/VolumeOffIcon"
import { observer } from "mobx-react-lite"
import { FC, useCallback, useState } from "react"
import {
  selectTrack,
  toggleMuteTrack,
  toggleSoloTrack,
  toogleAllGhostTracks,
  toogleGhostTrack,
} from "../../actions"
import { useContextMenu } from "../../hooks/useContextMenu"
import { useStores } from "../../hooks/useStores"
import { categoryEmojis, getCategoryIndex } from "../../midi/GM"
import Track from "../../track/Track"
import { trackColorToCSSColor } from "../../track/TrackColor"
import { IconButton } from "../ui/IconButton"
import { TrackInstrumentName } from "./InstrumentName"
import { TrackDialog } from "./TrackDialog"
import { TrackListContextMenu } from "./TrackListContextMenu"
import { TrackName } from "./TrackName"

export type TrackListItemProps = {
  track: Track
}

const Container = styled.div<{ selected: boolean }>`
  background-color: ${({ theme, selected }) =>
    selected ? theme.highlightColor : "transparent"};
  display: flex;
  align-items: center;
  padding: 0.5rem 0.5rem;
  border-radius: 0.5rem;
  margin: 0.5rem;
  outline: none;

  &:hover {
    background: ${({ theme }) => theme.highlightColor};
  }
`

const Label = styled.div`
  display: flex;
  padding-bottom: 0.3em;
  align-items: baseline;
`

const Instrument = styled.div`
  color: ${({ theme }) => theme.secondaryTextColor};
  font-size: 0.75rem;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
`

const Name = styled.div<{ selected: boolean }>`
  font-weight: 600;
  color: ${({ theme, selected }) =>
    selected ? theme.textColor : theme.secondaryTextColor};
  padding-right: 0.5em;
  font-size: 0.875rem;
  text-overflow: ellipsis;
  white-space: nowrap;
`

const Controls = styled.div`
  display: flex;
  align-items: center;
`

const ChannelName = styled.div`
  flex-shrink: 0;
  color: ${({ theme }) => theme.secondaryTextColor};
  font-size: 0.625rem;
  display: flex;
  align-items: center;
  border: 1px solid ${({ theme }) => theme.dividerColor};
  padding: 0 0.25rem;
  cursor: pointer;
  height: 1.25rem;
  margin-left: 0.25rem;

  &:hover {
    background: ${({ theme }) => theme.highlightColor};
  }
`

const Icon = styled.div<{ selected: boolean; color: string }>`
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  width: 2.6rem;
  height: 2.6rem;
  border-radius: 1.3rem;
  margin-right: 0.5rem;
  flex-shrink: 0;
  background: ${({ theme, selected }) =>
    selected ? theme.backgroundColor : theme.secondaryBackgroundColor};
  border: 2px solid ${({ color }) => color};
  box-sizing: border-box;
`

const IconInner = styled.div<{ selected: boolean }>`
  opacity: ${({ selected }) => (selected ? 1 : 0.5)};
`

const ControlButton = styled(IconButton)`
  width: 1.9rem;
  height: 1.9rem;
  margin-right: 0.25rem;
`

export const TrackListItem: FC<TrackListItemProps> = observer(({ track }) => {
  const rootStore = useStores()
  const { song, pianoRollStore, rootViewStore, trackMute, router } = rootStore
  const trackIndex = song.tracks.indexOf(track)

  const selected =
    !rootViewStore.isArrangeViewSelected &&
    track.id === pianoRollStore.selectedTrackId
  const mute = trackMute.isMuted(trackIndex)
  const solo = trackMute.isSolo(trackIndex)
  const ghostTrack = !pianoRollStore.notGhostTracks.has(trackIndex)
  const channel = track.channel
  const { onContextMenu, menuProps } = useContextMenu()
  const [isDialogOpened, setDialogOpened] = useState(false)

  const onDoubleClickIcon = useCallback(() => {
    if (track.isConductorTrack) {
      return
    }
    pianoRollStore.openInstrumentBrowser = true
    pianoRollStore.instrumentBrowserSetting = {
      programNumber: track.programNumber ?? 0,
      isRhythmTrack: track.isRhythmTrack,
    }
  }, [])
  const onClickMute: React.MouseEventHandler<HTMLButtonElement> = useCallback(
    (e) => {
      e.stopPropagation()
      toggleMuteTrack(rootStore)(trackIndex)
    },
    [trackIndex],
  )
  const onClickSolo: React.MouseEventHandler<HTMLButtonElement> = useCallback(
    (e) => {
      e.stopPropagation()
      toggleSoloTrack(rootStore)(trackIndex)
    },
    [trackIndex],
  )
  const onClickGhostTrack: React.MouseEventHandler<HTMLButtonElement> =
    useCallback(
      (e) => {
        e.stopPropagation()
        if (e.nativeEvent.altKey) {
          toogleAllGhostTracks(rootStore)()
        } else {
          toogleGhostTrack(rootStore)(trackIndex)
        }
      },
      [trackIndex],
    )
  const onSelectTrack = useCallback(() => {
    router.pushTrack()
    selectTrack(rootStore)(trackIndex)
  }, [trackIndex])
  const onClickChannel = useCallback(() => {
    setDialogOpened(true)
  }, [])

  const emoji = track.isRhythmTrack
    ? "🥁"
    : categoryEmojis[getCategoryIndex(track.programNumber ?? 0)]

  const color =
    track.color !== undefined
      ? trackColorToCSSColor(track.color)
      : "transparent"

  return (
    <>
      <Container
        selected={selected}
        onMouseDown={onSelectTrack}
        onContextMenu={onContextMenu}
        tabIndex={-1}
      >
        <Icon
          selected={selected}
          color={color}
          onDoubleClick={onDoubleClickIcon}
        >
          <IconInner selected={selected}>{emoji}</IconInner>
        </Icon>
        <div>
          <Label>
            <Name selected={selected}>
              <TrackName track={track} />
            </Name>
            <Instrument>
              <TrackInstrumentName track={track} />
            </Instrument>
          </Label>
          <Controls>
            <ControlButton active={solo} onMouseDown={onClickSolo}>
              <Headset />
            </ControlButton>
            <ControlButton active={mute} onMouseDown={onClickMute}>
              {mute ? <VolumeOff /> : <VolumeUp />}
            </ControlButton>
            <ControlButton active={ghostTrack} onMouseDown={onClickGhostTrack}>
              <Layers />
            </ControlButton>
            {channel !== undefined && (
              <ChannelName onClick={onClickChannel}>
                CH {channel + 1}
              </ChannelName>
            )}
          </Controls>
        </div>
      </Container>
      <TrackListContextMenu {...menuProps} trackIndex={trackIndex} />
      <TrackDialog
        trackIndex={trackIndex}
        open={isDialogOpened}
        onClose={() => setDialogOpened(false)}
      />
    </>
  )
})
