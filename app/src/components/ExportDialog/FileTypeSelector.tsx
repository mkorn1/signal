import styled from "@emotion/styled"
import { FC } from "react"
import { Localized } from "../../localize/useLocalization"
import {
  ToolbarButtonGroup,
  ToolbarButtonGroupItem,
} from "../Toolbar/ToolbarButtonGroup"

const ButtonGroup = styled(ToolbarButtonGroup)`
  background-color: transparent;
  margin: 0.65rem 0 0.65rem 0;
`

export interface FormatSelectorProps {
  value: "WAV" | "MP3"
  onChange: (format: "WAV" | "MP3") => void
}

export const FileTypeSelector: FC<FormatSelectorProps> = ({
  value,
  onChange,
}) => {
  return (
    <div>
      <Localized name="file-type" />:
      <ButtonGroup>
        <ToolbarButtonGroupItem
          selected={value === "WAV"}
          onMouseDown={() => onChange("WAV")}
        >
          <span>WAV</span>
        </ToolbarButtonGroupItem>
        <ToolbarButtonGroupItem
          selected={value === "MP3"}
          onMouseDown={() => onChange("MP3")}
        >
          <span>MP3</span>
        </ToolbarButtonGroupItem>
      </ButtonGroup>
    </div>
  )
}
