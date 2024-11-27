import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import { categoryEmojis, getCategoryIndex } from "../../midi/GM"
import { ToolbarButton } from "../Toolbar/ToolbarButton"
import { TrackInstrumentName } from "../TrackList/InstrumentName"

export const InstrumentButton: FC = observer(() => {
  const {
    pianoRollStore,
    pianoRollStore: { selectedTrack },
  } = useStores()

  const onClickInstrument = useCallback(() => {
    const track = selectedTrack
    if (track === undefined) {
      return
    }
    const programNumber = track.programNumber
    pianoRollStore.instrumentBrowserSetting = {
      isRhythmTrack: track.isRhythmTrack,
      programNumber: programNumber ?? 0,
    }
    pianoRollStore.openInstrumentBrowser = true
  }, [pianoRollStore])

  if (selectedTrack === undefined) {
    return <></>
  }

  const { programNumber } = selectedTrack
  const emoji = categoryEmojis[getCategoryIndex(programNumber ?? 0)]

  return (
    <ToolbarButton
      onMouseDown={(e) => {
        e.preventDefault()
        onClickInstrument()
      }}
    >
      <span style={{ marginRight: "0.5rem" }}>{emoji}</span>
      <span>
        <TrackInstrumentName track={selectedTrack} />
      </span>
    </ToolbarButton>
  )
})
