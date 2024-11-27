import { observer } from "mobx-react-lite"
import { FC } from "react"
import { useStores } from "../../hooks/useStores"
import { AutoScrollButton } from "../Toolbar/AutoScrollButton"

export const TempoGraphAutoScrollButton: FC = observer(() => {
  const { tempoEditorStore } = useStores()
  const { autoScroll } = tempoEditorStore

  return (
    <AutoScrollButton
      onClick={() => (tempoEditorStore.autoScroll = !autoScroll)}
      selected={autoScroll}
    />
  )
})
