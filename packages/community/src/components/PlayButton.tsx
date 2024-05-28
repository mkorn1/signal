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
`

export interface PlayButtonProps {
  onMouseDown?: () => void
  isPlaying: boolean
}

export const PlayButton: FC<PlayButtonProps> = (
  { onMouseDown, isPlaying },
  ref,
) => {
  return (
    <StyledButton
      id="button-play"
      onMouseDown={onMouseDown}
      className={isPlaying ? "active" : undefined}
    >
      {isPlaying ? <Pause /> : <PlayArrow />}
    </StyledButton>
  )
}
