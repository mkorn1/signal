import { observer } from "mobx-react-lite"
import { useCallback } from "react"
import {
  arrangeBatchUpdateSelectedNotesVelocity,
  BatchUpdateOperation,
} from "../../actions"
import { useStores } from "../../hooks/useStores"
import { VelocityDialog } from "./VelocityDialog"

export const ArrangeVelocityDialog = observer(() => {
  const rootStore = useStores()
  const { arrangeViewStore, pianoRollStore } = rootStore

  const onClose = useCallback(
    () => (arrangeViewStore.openVelocityDialog = false),
    [arrangeViewStore],
  )

  const onClickOK = useCallback(
    (value: number, operationType: BatchUpdateOperation["type"]) => {
      arrangeBatchUpdateSelectedNotesVelocity(rootStore)({
        type: operationType,
        value,
      })
      arrangeViewStore.openVelocityDialog = false
    },
    [arrangeViewStore],
  )

  return (
    <VelocityDialog
      open={arrangeViewStore.openVelocityDialog}
      value={pianoRollStore.newNoteVelocity}
      onClickOK={onClickOK}
      onClose={onClose}
    />
  )
})
