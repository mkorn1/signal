import { observer } from "mobx-react-lite"
import { useCallback } from "react"
import {
  BatchUpdateOperation,
  batchUpdateSelectedNotesVelocity,
} from "../../actions"
import { useStores } from "../../hooks/useStores"
import { VelocityDialog } from "./VelocityDialog"

export const PianoRollVelocityDialog = observer(() => {
  const rootStore = useStores()
  const { pianoRollStore } = rootStore

  const onClose = useCallback(
    () => (pianoRollStore.openVelocityDialog = false),
    [pianoRollStore],
  )

  const onClickOK = useCallback(
    (value: number, operationType: BatchUpdateOperation["type"]) => {
      batchUpdateSelectedNotesVelocity(rootStore)({
        type: operationType,
        value,
      })
      pianoRollStore.openVelocityDialog = false
    },
    [pianoRollStore],
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
