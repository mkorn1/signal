import styled from "@emotion/styled"
import { FC, useCallback } from "react"
import { useInstrumentBrowser } from "../../hooks/useInstrumentBrowser"
import { usePianoRoll } from "../../hooks/usePianoRoll"
import { useTrack } from "../../hooks/useTrack"
import { categoryEmojis, getCategoryIndex } from "../../midi/GM"
import { Tooltip } from "../ui/Tooltip"
import { InstrumentName } from "../TrackList/InstrumentName"

const Button = styled.button`
  display: flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.25rem 0.5rem;
  border: 1px solid ${({ theme }) => theme.themeColor};
  border-radius: 0.25rem;
  background: rgba(77, 166, 255, 0.1);
  color: ${({ theme }) => theme.themeColor};
  font-size: 0.6875rem;
  font-weight: 600;
  font-family: inherit;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  cursor: pointer;
  transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
  white-space: nowrap;

  &:hover {
    background: rgba(77, 166, 255, 0.15);
  }

  &:active {
    transform: scale(0.98);
  }
`

const Emoji = styled.span`
  font-size: 0.75rem;
`

export const InstrumentButton: FC = () => {
  const { selectedTrackId } = usePianoRoll()
  const { isRhythmTrack, programNumber } = useTrack(selectedTrackId)
  const { setSetting, setOpen } = useInstrumentBrowser()

  const onClickInstrument = useCallback(() => {
    setSetting({
      isRhythmTrack,
      programNumber,
    })
    setOpen(true)
  }, [isRhythmTrack, programNumber, setOpen, setSetting])

  const emoji = isRhythmTrack
    ? "🥁"
    : categoryEmojis[getCategoryIndex(programNumber ?? 0)]

  return (
    <Tooltip title="Change instrument sound">
      <Button
        onMouseDown={(e) => {
          e.preventDefault()
          onClickInstrument()
        }}
      >
        <Emoji>{emoji}</Emoji>
        <InstrumentName
          programNumber={programNumber}
          isRhythmTrack={isRhythmTrack}
        />
      </Button>
    </Tooltip>
  )
}
