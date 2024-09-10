import FormatListBulleted from "mdi-react/FormatListBulletedIcon"
import { observer } from "mobx-react-lite"
import { FC, MouseEvent, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { ToolbarButton } from "../Toolbar/ToolbarButton"
import { Tooltip } from "../ui/Tooltip"

export const EventListButton: FC = observer(() => {
  const { pianoRollStore } = useStores()

  return (
    <Tooltip title={<Localized name="event-list" />}>
      <ToolbarButton
        selected={pianoRollStore.showEventList}
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
})
