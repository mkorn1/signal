import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import { AutoScrollButton } from "../Toolbar/AutoScrollButton"

export const ArrangeAutoScrollButton: FC = observer(() => {
  const { arrangeViewStore } = useStores()
  const { autoScroll } = arrangeViewStore

  const onClickAutoScroll = useCallback(
    () => (arrangeViewStore.autoScroll = !arrangeViewStore.autoScroll),
    [arrangeViewStore],
  )

  return <AutoScrollButton onClick={onClickAutoScroll} selected={autoScroll} />
})
