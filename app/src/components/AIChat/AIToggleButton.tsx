import styled from "@emotion/styled"
import ChevronLeft from "mdi-react/ChevronLeftIcon"
import ChevronRight from "mdi-react/ChevronRightIcon"
import Robot from "mdi-react/RobotIcon"
import { FC, useCallback } from "react"
import { useAIChat } from "../../hooks/useAIChat"
import { Tooltip } from "../ui/Tooltip"

const ToggleButton = styled.button<{ isOpen: boolean }>`
  position: fixed;
  bottom: 1.5rem;
  right: ${({ isOpen }) => (isOpen ? "calc(min(400px, 35vw) + 0.5rem)" : "1rem")};
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
  width: ${({ isOpen }) => (isOpen ? "32px" : "auto")};
  height: 32px;
  padding: ${({ isOpen }) => (isOpen ? "0" : "0 0.75rem")};
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 1rem;
  background: ${({ isOpen, theme }) =>
    isOpen ? "rgba(30, 30, 30, 0.95)" : theme.themeColor};
  color: ${({ isOpen, theme }) =>
    isOpen ? theme.secondaryTextColor : theme.onSurfaceColor};
  font-size: 0.75rem;
  font-weight: 600;
  font-family: inherit;
  cursor: pointer;
  transition: all 200ms cubic-bezier(0.4, 0, 0.2, 1);
  backdrop-filter: blur(8px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);

  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    background: ${({ isOpen, theme }) =>
      isOpen ? "rgba(50, 50, 50, 0.95)" : theme.themeColor};
    filter: ${({ isOpen }) => (isOpen ? "none" : "brightness(1.1)")};
  }

  &:active {
    transform: translateY(0) scale(0.98);
  }

  svg {
    width: 1rem;
    height: 1rem;
    fill: currentColor;
  }
`

const ButtonText = styled.span`
  @media (max-width: 600px) {
    display: none;
  }
`

export const AIToggleButton: FC = () => {
  const { isOpen, toggle } = useAIChat()

  const handleClick = useCallback(() => {
    toggle()
  }, [toggle])

  return (
    <Tooltip
      title={isOpen ? "Collapse AI Composer" : "Open AI Composer"}
      side="left"
    >
      <ToggleButton isOpen={isOpen} onClick={handleClick}>
        {isOpen ? (
          <ChevronRight />
        ) : (
          <>
            <Robot />
            <ButtonText>AI</ButtonText>
          </>
        )}
      </ToggleButton>
    </Tooltip>
  )
}

