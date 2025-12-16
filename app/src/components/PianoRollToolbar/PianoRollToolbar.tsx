import styled from "@emotion/styled"
import { FC } from "react"
import { useEditorMode } from "../../hooks/useEditorMode"
import { InstrumentBrowser } from "../InstrumentBrowser/InstrumentBrowser"
import { AutoScrollButton } from "../Toolbar/AutoScrollButton"
import { Toolbar } from "../Toolbar/Toolbar"
import { EventListButton } from "./EventListButton"
import { InstrumentButton } from "./InstrumentButton"
import { PanSlider } from "./PanSlider"
import { PianoRollToolSelector } from "./PianoRollToolSelector"
import { VolumeSlider } from "./VolumeSlider"

const Spacer = styled.div`
  width: 1rem;
`

const FlexibleSpacer = styled.div`
  flex-grow: 1;
`

export const PianoRollToolbar: FC = () => {
  const { isAdvanced } = useEditorMode()

  return (
    <Toolbar>
      <InstrumentButton />
      <InstrumentBrowser />

      {isAdvanced && <EventListButton />}

      <Spacer />

      <VolumeSlider />
      {isAdvanced && <PanSlider />}

      <FlexibleSpacer />

      <PianoRollToolSelector />

      <AutoScrollButton />
    </Toolbar>
  )
}
