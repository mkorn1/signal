import { observer } from "mobx-react-lite"
import { useCallback } from "react"
import {
  BatchUpdateOperation,
  useBatchUpdateSelectedNotesVelocity,
} from "../../actions"
import { useStores } from "../../hooks/useStores"
import { VelocityDialog } from "./VelocityDialog"

export const PianoRollVelocityDialog = observer(() => {
  const { pianoRollStore } = useStores()
  const batchUpdateSelectedNotesVelocity = useBatchUpdateSelectedNotesVelocity()

  const onClose = useCallback(
    () => (pianoRollStore.openVelocityDialog = false),
    [pianoRollStore],
  )

  const onClickOK = useCallback(
    (value: number, operationType: BatchUpdateOperation["type"]) => {
      batchUpdateSelectedNotesVelocity({
        type: operationType,
        value,
      })
      pianoRollStore.openVelocityDialog = false
    },
    [pianoRollStore, batchUpdateSelectedNotesVelocity],
  )

  return (
    <VelocityDialog
      open={pianoRollStore.openVelocityDialog}
      value={pianoRollStore.newNoteVelocity}
      onClickOK={onClickOK}
      onClose={onClose}
    />
  )
})
