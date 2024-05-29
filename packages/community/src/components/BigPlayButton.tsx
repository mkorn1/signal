import styled from "@emotion/styled"
import Pause from "mdi-react/PauseIcon.js"
import PlayArrow from "mdi-react/PlayArrowIcon.js"
import { FC } from "react"
import { CircleButton } from "./CircleButton.js"

export const StyledButton = styled(CircleButton)`
  background: ${({ theme }) => theme.themeColor};

  &:hover {
    background: ${({ theme }) => theme.themeColor};
    opacity: 0.8;
  }

  &.active {
    background: ${({ theme }) => theme.themeColor};
  }

  svg {
    width: 1.5rem;
    height: 1.5rem;
  }
`

export interface BigPlayButtonProps {
  onMouseDown?: () => void
  isPlaying: boolean
}

export const BigPlayButton: FC<BigPlayButtonProps> = ({
  onMouseDown,
  isPlaying,
}) => {
  return (
    <StyledButton
      onMouseDown={onMouseDown}
      className={isPlaying ? "active" : undefined}
    >
      {isPlaying ? <Pause /> : <PlayArrow />}
    </StyledButton>
  )
}
