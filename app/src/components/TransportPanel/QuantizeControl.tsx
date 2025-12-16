import styled from "@emotion/styled"
import MusicNote from "mdi-react/MusicNoteIcon"
import FiberManualRecord from "mdi-react/FiberManualRecordIcon"
import React, { FC, useCallback } from "react"
import { useQuantizer } from "../../hooks/useQuantizer"
import { Tooltip } from "../ui/Tooltip"
import { QuantizePopup } from "../Toolbar/QuantizeSelector/QuantizePopup"

const Container = styled.div`
  display: flex;
  align-items: center;
  gap: 0.25rem;
`

const SnapButton = styled.button<{ active: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  background: ${({ active }) => (active ? "rgba(255, 255, 255, 0.1)" : "transparent")};
  border: none;
  border-radius: 0.25rem;
  padding: 0.3rem;
  cursor: pointer;
  color: ${({ active }) => (active ? "var(--color-theme)" : "var(--color-text-secondary)")};
  transition: all 150ms;

  &:hover {
    background: rgba(255, 255, 255, 0.08);
  }

  svg {
    width: 1.1rem;
    height: 1.1rem;
  }
`

const ValueButton = styled.button`
  display: flex;
  align-items: center;
  background: transparent;
  border: none;
  padding: 0.3rem 0.5rem;
  cursor: pointer;
  color: var(--color-text-secondary);
  font-family: var(--font-mono);
  font-size: 0.85rem;
  border-radius: 0.25rem;
  transition: all 150ms;

  &:hover {
    background: rgba(255, 255, 255, 0.08);
    color: var(--color-text);
  }
`

const DotLabel = styled(FiberManualRecord)`
  position: relative;
  top: -0.4rem;
  left: 0.05rem;
  width: 0.4rem;
  height: 0.4rem;
  margin: 0 -0.05rem;
`

const TripletLabel = styled.span`
  color: var(--color-text-secondary);
  font-size: 70%;
  padding: 0 0.2em;
`

function calcQuantize(num: number, dot: boolean, triplet: boolean): number {
  let val = num
  if (dot) {
    val /= 1.5
  }
  if (triplet) {
    val *= 1.5
  }
  return val
}

export const QuantizeControl: FC = () => {
  const { quantize, setQuantize, isQuantizeEnabled, setIsQuantizeEnabled } =
    useQuantizer()

  const onClickSwitch = useCallback(() => {
    setIsQuantizeEnabled(!isQuantizeEnabled)
  }, [setIsQuantizeEnabled, isQuantizeEnabled])

  // Calculate dot/triplet display
  const dot = quantize % 1 !== 0 && (quantize * 1.5) % 1 === 0
  const triplet = (quantize / 1.5) % 1 === 0
  const denominator = calcQuantize(quantize, triplet, dot)

  const list = [1, 2, 4, 8, 16, 32, 64, 128]

  return (
    <Container>
      <Tooltip title="Snap notes to grid">
        <SnapButton active={isQuantizeEnabled} onMouseDown={onClickSwitch}>
          <MusicNote />
        </SnapButton>
      </Tooltip>
      <QuantizePopup
        value={denominator}
        values={list}
        dotted={dot}
        triplet={triplet}
        onChangeValue={(d) => setQuantize(calcQuantize(d, dot, triplet))}
        onChangeDotted={(d) => setQuantize(calcQuantize(denominator, d, false))}
        onChangeTriplet={(t) => setQuantize(calcQuantize(denominator, false, t))}
        side="top"
        trigger={
          <ValueButton
            onWheel={(e) => {
              const currentIndex = list.indexOf(denominator)
              const delta = e.deltaY < 0 ? 1 : -1
              const index = Math.min(
                list.length - 1,
                Math.max(0, currentIndex + delta),
              )
              setQuantize(calcQuantize(list[index], dot, triplet))
            }}
          >
            <span>{denominator}</span>
            {triplet && <TripletLabel>3</TripletLabel>}
            {dot && <DotLabel />}
          </ValueButton>
        }
      />
    </Container>
  )
}

