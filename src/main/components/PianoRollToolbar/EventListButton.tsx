import FormatListBulleted from "mdi-react/FormatListBulletedIcon"
import { FC, MouseEvent, useCallback } from "react"
import { Tooltip } from "../../../components/Tooltip"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { ToolbarButton } from "../Toolbar/ToolbarButton"

export const EventListButton: FC = () => {
  const { pianoRollStore } = useStores()

  return (
    <Tooltip title={<Localized name="event-list" />}>
      <ToolbarButton
        onMouseDown={useCallback((e: MouseEvent) => {
          e.preventDefault()
          pianoRollStore.showEventList = !pianoRollStore.showEventList
        }, [])}
      >
        <FormatListBulleted
          style={{
            width: "1.2rem",
            fill: "currentColor",
          }}
        />
      </ToolbarButton>
    </Tooltip>
  )
}
