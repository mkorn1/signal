import styled from "@emotion/styled"
import { observer } from "mobx-react-lite"
import { FC } from "react"
import { useStores } from "../../hooks/useStores"
import InstrumentBrowser from "../InstrumentBrowser/InstrumentBrowser"
import { Toolbar } from "../Toolbar/Toolbar"
import { TrackListMenuButton } from "../TrackList/TrackListMenuButton"
import { EventListButton } from "./EventListButton"
import { InstrumentButton } from "./InstrumentButton"
import { PanSlider } from "./PanSlider"
import { PianoRollAutoScrollButton } from "./PianoRollAutoScrollButton"
import { PianoRollQuantizeSelector } from "./PianoRollQuantizeSelector"
import { PianoRollToolSelector } from "./PianoRollToolSelector"
import { TrackNameInput } from "./TrackNameInput"
import { VolumeSlider } from "./VolumeSlider"

const Spacer = styled.div`
  width: 1rem;
`

const FlexibleSpacer = styled.div`
  flex-grow: 1;
`

export const PianoRollToolbar: FC = observer(() => {
  const { pianoRollStore } = useStores()

  const { selectedTrackId } = pianoRollStore

  return (
    <Toolbar>
      <TrackListMenuButton />

      <TrackNameInput />

      <EventListButton />

      <Spacer />

      <InstrumentButton />
      <InstrumentBrowser />

      <VolumeSlider trackId={selectedTrackId} />
      <PanSlider trackId={selectedTrackId} />

      <FlexibleSpacer />

      <PianoRollToolSelector />

      <PianoRollQuantizeSelector />

      <PianoRollAutoScrollButton />
    </Toolbar>
  )
})
