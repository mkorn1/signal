import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import { ToolSelector } from "../Toolbar/ToolSelector"

export const TempoGraphToolSelector: FC = observer(() => {
  const { tempoEditorStore } = useStores()
  const { mouseMode } = tempoEditorStore
  return (
    <ToolSelector
      mouseMode={mouseMode}
      onSelect={useCallback(
        (mouseMode: any) => (tempoEditorStore.mouseMode = mouseMode),
        [],
      )}
    />
  )
})
