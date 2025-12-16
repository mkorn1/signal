import styled from "@emotion/styled"
import { keyframes } from "@emotion/react"
import { DEFAULT_TEMPO } from "@signal-app/player"
import { FC, useEffect, useState } from "react"
import { useAtomValue } from "jotai"
import { useConductorTrack } from "../../hooks/useConductorTrack"
import { usePlayer } from "../../hooks/usePlayer"
import { agentBpmUpdatedAtom } from "../../stores/AgentMusicState"
import { NumberInput } from "../inputs/NumberInput"

const pulseGlow = keyframes`
  0% {
    transform: scale(1);
    filter: brightness(1);
  }
  25% {
    transform: scale(1.05);
    filter: brightness(1.4) drop-shadow(0 0 8px var(--color-theme));
  }
  50% {
    transform: scale(1.02);
    filter: brightness(1.2) drop-shadow(0 0 4px var(--color-theme));
  }
  100% {
    transform: scale(1);
    filter: brightness(1);
  }
`

const TempoInput = styled(NumberInput)`
  background: transparent;
  -webkit-appearance: none;
  border: none;
  color: inherit;
  font-size: inherit;
  font-family: inherit;
  width: 5em;
  text-align: center;
  outline: none;
  font-family: var(--font-mono);
  font-size: 1rem;
  padding: 0.3rem 0;

  &::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
`

const TempoWrapper = styled.div<{ isAnimating: boolean }>`
  display: flex;
  align-items: center;
  border: 1px solid transparent;
  padding-left: 0.75rem;
  border-radius: 0.25rem;
  animation: ${(props) => (props.isAnimating ? pulseGlow : "none")} 0.6s ease-out;

  label {
    font-size: 0.6rem;
    color: var(--color-text-secondary);
  }

  &:focus-within {
    border: 1px solid var(--color-divider);
    background: #ffffff14;
  }
`

export const TempoForm: FC = () => {
  const { position, setCurrentTempo } = usePlayer()
  const { currentTempo, setTempo } = useConductorTrack()
  const tempo = currentTempo ?? DEFAULT_TEMPO
  const bpmUpdated = useAtomValue(agentBpmUpdatedAtom)
  const [isAnimating, setIsAnimating] = useState(false)

  useEffect(() => {
    if (bpmUpdated > 0) {
      setIsAnimating(true)
      const timer = setTimeout(() => setIsAnimating(false), 600)
      return () => clearTimeout(timer)
    }
  }, [bpmUpdated])

  const changeTempo = (tempo: number) => {
    setTempo(tempo, position)
    setCurrentTempo(tempo)
  }

  return (
    <TempoWrapper isAnimating={isAnimating}>
      <label htmlFor="tempo-input">BPM</label>
      <TempoInput
        id="tempo-input"
        min={1}
        max={512}
        value={Math.round(tempo * 100) / 100}
        step={1}
        onChange={changeTempo}
      />
    </TempoWrapper>
  )
}
