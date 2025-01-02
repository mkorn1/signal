import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import { AutoScrollButton } from "../Toolbar/AutoScrollButton"

export const PianoRollAutoScrollButton: FC = observer(() => {
  const { pianoRollStore } = useStores()
  const { autoScroll } = pianoRollStore

  const onClickAutoScroll = useCallback(
    () => (pianoRollStore.autoScroll = !pianoRollStore.autoScroll),
    [pianoRollStore],
  )

  return <AutoScrollButton onClick={onClickAutoScroll} selected={autoScroll} />
})
