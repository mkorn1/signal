import styled from "@emotion/styled"
import { FC, useCallback } from "react"
import { Tooltip } from "../../../components/Tooltip"
import PencilIcon from "../../images/icons/pencil.svg"
import SelectionIcon from "../../images/icons/selection.svg"
import { Localized } from "../../localize/useLocalization"
import { PianoRollMouseMode } from "../../stores/PianoRollStore"
import {
  ToolbarButtonGroup,
  ToolbarButtonGroupItem,
} from "./ToolbarButtonGroup"

const ButtonGroup = styled(ToolbarButtonGroup)`
  background-color: transparent;
  margin-right: 1rem;
`

export interface ToolSelectorProps {
  mouseMode: PianoRollMouseMode
  onSelect: (mouseMode: PianoRollMouseMode) => void
}

const IconWrapper = styled.div`
  display: flex;
`

export const ToolSelector: FC<ToolSelectorProps> = ({
  mouseMode,
  onSelect,
}) => {
  return (
    <ButtonGroup>
      <ToolbarButtonGroupItem
        onMouseDown={useCallback(() => onSelect("pencil"), [])}
        selected={mouseMode === "pencil"}
      >
        <Tooltip
          title={
            <>
              <Localized name="pencil-tool" /> [1]
            </>
          }
        >
          <IconWrapper>
            <PencilIcon
              style={{
                width: "1.3rem",
                height: "1.3rem",
                fill: "currentColor",
              }}
              viewBox="0 0 128 128"
            />
          </IconWrapper>
        </Tooltip>
      </ToolbarButtonGroupItem>
      <ToolbarButtonGroupItem
        onMouseDown={useCallback(() => onSelect("selection"), [])}
        selected={mouseMode === "selection"}
      >
        <Tooltip
          title={
            <>
              <Localized name="selection-tool" /> [2]
            </>
          }
        >
          <IconWrapper>
            <SelectionIcon
              style={{
                width: "1.3rem",
                height: "1.3rem",
                fill: "currentColor",
              }}
              viewBox="0 0 128 128"
            />
          </IconWrapper>
        </Tooltip>
      </ToolbarButtonGroupItem>
    </ButtonGroup>
  )
}
