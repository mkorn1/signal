import { observer } from "mobx-react-lite"
import { useCallback } from "react"
import { useArrangeTransposeSelection } from "../../actions"
import { useStores } from "../../hooks/useStores"
import { TransposeDialog } from "./TransposeDialog"

export const ArrangeTransposeDialog = observer(() => {
  const { arrangeViewStore } = useStores()
  const { openTransposeDialog } = arrangeViewStore
  const arrangeTransposeSelection = useArrangeTransposeSelection()

  const onClose = useCallback(
    () => (arrangeViewStore.openTransposeDialog = false),
    [arrangeViewStore],
  )

  const onClickOK = useCallback(
    (value: number) => {
      arrangeTransposeSelection(value)
      arrangeViewStore.openTransposeDialog = false
    },
    [arrangeViewStore, arrangeTransposeSelection],
  )

  return (
    <TransposeDialog
      open={openTransposeDialog}
      onClose={onClose}
      onClickOK={onClickOK}
    />
  )
})
