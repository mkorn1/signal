import { observer } from "mobx-react-lite"
import { useCallback } from "react"
import {
  BatchUpdateOperation,
  useArrangeBatchUpdateSelectedNotesVelocity,
} from "../../actions"
import { useStores } from "../../hooks/useStores"
import { VelocityDialog } from "./VelocityDialog"

export const ArrangeVelocityDialog = observer(() => {
  const { arrangeViewStore, pianoRollStore } = useStores()
  const arrangeBatchUpdateSelectedNotesVelocity =
    useArrangeBatchUpdateSelectedNotesVelocity()

  const onClose = useCallback(
    () => (arrangeViewStore.openVelocityDialog = false),
    [arrangeViewStore],
  )

  const onClickOK = useCallback(
    (value: number, operationType: BatchUpdateOperation["type"]) => {
      arrangeBatchUpdateSelectedNotesVelocity({
        type: operationType,
        value,
      })
      arrangeViewStore.openVelocityDialog = false
    },
    [arrangeViewStore, arrangeBatchUpdateSelectedNotesVelocity],
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
