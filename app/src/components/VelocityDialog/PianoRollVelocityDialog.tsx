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
  const { rootViewStore, pianoRollStore } = rootStore

  const onClose = useCallback(
    () => (rootViewStore.openVelocityDialog = false),
    [rootViewStore],
  )

  const onClickOK = useCallback(
    (value: number, operationType: BatchUpdateOperation["type"]) => {
      batchUpdateSelectedNotesVelocity(rootStore)({
        type: operationType,
        value,
      })
      rootViewStore.openVelocityDialog = false
    },
    [rootViewStore],
  )

  return (
    <VelocityDialog
      open={rootViewStore.openVelocityDialog}
      value={pianoRollStore.newNoteVelocity}
      onClickOK={onClickOK}
      onClose={onClose}
    />
  )
})
