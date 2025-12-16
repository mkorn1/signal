import styled from "@emotion/styled"
import ChevronLeft from "mdi-react/ChevronLeftIcon"
import ChevronRight from "mdi-react/ChevronRightIcon"
import { FC, useCallback } from "react"
import { useAIChat } from "../../hooks/useAIChat"
import { Tooltip } from "../ui/Tooltip"

// Tab button attached to the left edge of AI panel (when open)
const AttachedTab = styled.button`
  position: absolute;
  right: 100%;
  top: 50%;
  transform: translateY(-50%);
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 56px;
  padding: 0;
  border: none;
  border-radius: 0.375rem 0 0 0.375rem;
  background: ${({ theme }) => theme.themeColor};
  color: ${({ theme }) => theme.onSurfaceColor};
  font-size: 0.75rem;
  font-weight: 600;
  font-family: inherit;
  cursor: pointer;
  transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);

  &:hover {
    filter: brightness(1.1);
  }

  &:active {
    transform: translateY(-50%) scale(0.95);
  }

  svg {
    width: 0.875rem;
    height: 0.875rem;
    fill: currentColor;
  }
`

// Tab button fixed to the right edge of viewport (when closed)
const EdgeTab = styled.button`
  position: fixed;
  right: 0;
  top: 50%;
  transform: translateY(-50%);
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 56px;
  padding: 0;
  border: none;
  border-radius: 0.375rem 0 0 0.375rem;
  background: ${({ theme }) => theme.themeColor};
  color: ${({ theme }) => theme.onSurfaceColor};
  font-size: 0.75rem;
  font-weight: 600;
  font-family: inherit;
  cursor: pointer;
  transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);

  &:hover {
    filter: brightness(1.1);
  }

  &:active {
    transform: translateY(-50%) scale(0.95);
  }

  svg {
    width: 0.875rem;
    height: 0.875rem;
    fill: currentColor;
  }
`

// Attached toggle button (shown when AI panel is open)
export const AIToggleButtonAttached: FC = () => {
  const { toggle } = useAIChat()

  const handleClick = useCallback(() => {
    toggle()
  }, [toggle])

  return (
    <Tooltip title="Collapse AI Composer" side="left">
      <AttachedTab onClick={handleClick}>
        <ChevronRight />
      </AttachedTab>
    </Tooltip>
  )
}

// Edge toggle button (shown when AI panel is closed)
export const AIToggleButton: FC = () => {
  const { isOpen, toggle } = useAIChat()

  const handleClick = useCallback(() => {
    toggle()
  }, [toggle])

  // Don't render when panel is open (the attached button handles it)
  if (isOpen) {
    return null
  }

  return (
    <Tooltip title="Open AI Composer" side="left">
      <EdgeTab onClick={handleClick}>
        <ChevronLeft />
      </EdgeTab>
    </Tooltip>
  )
}
