import styled from "@emotion/styled"
import { keyframes } from "@emotion/react"
import { FC, useEffect, useState } from "react"
import { useAtomValue } from "jotai"
import {
  agentKeySignatureAtom,
  agentKeyUpdatedAtom,
  formatKeySignature,
} from "../../stores/AgentMusicState"

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

const KeyWrapper = styled.div<{ isAnimating: boolean }>`
  display: flex;
  align-items: center;
  padding: 0 0.75rem;
  animation: ${(props) => (props.isAnimating ? pulseGlow : "none")} 0.6s ease-out;

  label {
    font-size: 0.6rem;
    color: var(--color-text-secondary);
  }
`

const KeyValue = styled.span`
  font-family: var(--font-mono);
  font-size: 1rem;
  padding: 0.3rem 0;
  padding-left: 0.3rem;
  min-width: 2.5em;
  text-align: center;
`

export const KeyDisplay: FC = () => {
  const keySignature = useAtomValue(agentKeySignatureAtom)
  const keyUpdated = useAtomValue(agentKeyUpdatedAtom)
  const [isAnimating, setIsAnimating] = useState(false)

  useEffect(() => {
    if (keyUpdated > 0) {
      setIsAnimating(true)
      const timer = setTimeout(() => setIsAnimating(false), 600)
      return () => clearTimeout(timer)
    }
  }, [keyUpdated])

  return (
    <KeyWrapper isAnimating={isAnimating}>
      <label>Key</label>
      <KeyValue>{formatKeySignature(keySignature)}</KeyValue>
    </KeyWrapper>
  )
}

