import styled from "@emotion/styled"
import Pause from "mdi-react/PauseIcon"
import PlayArrow from "mdi-react/PlayArrowIcon"
import { FC } from "react"
import { Tooltip } from "../../../components/Tooltip"
import { Localized } from "../../localize/useLocalization"
import { CircleButton } from "./CircleButton"

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
    <Tooltip
      title={
        <>
          <Localized name="play-pause" /> [space]
        </>
      }
      side="top"
    >
      <StyledButton
        id="button-play"
        onMouseDown={onMouseDown}
        className={isPlaying ? "active" : undefined}
      >
        {isPlaying ? <Pause /> : <PlayArrow />}
      </StyledButton>
    </Tooltip>
  )
}
