import styled from "@emotion/styled"
import { observer } from "mobx-react-lite"
import { useCallback, useEffect, useState } from "react"
import {
  BatchUpdateOperation,
  batchUpdateSelectedNotesVelocity,
} from "../../actions"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import {
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
} from "../Dialog/Dialog"
import { Button } from "../ui/Button"
import { RadioButton } from "../ui/RadioButton"
import { StyledNumberInput } from "../ui/StyledNumberInput"

const Column = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`

const Row = styled.div`
  display: flex;
  gap: 2rem;
`

export const VelocityDialog = observer(() => {
  const rootStore = useStores()
  const { rootViewStore, pianoRollStore } = rootStore
  const { openVelocityDialog: open } = rootViewStore
  const [operationType, setOperationType] =
    useState<BatchUpdateOperation["type"]>("set")
  const [value, setValue] = useState(pianoRollStore.newNoteVelocity)

  useEffect(() => {
    if (open) {
      setValue(pianoRollStore.newNoteVelocity)
    }
  }, [open])

  const onClose = useCallback(
    () => (rootViewStore.openVelocityDialog = false),
    [rootViewStore],
  )

  const onClickOK = useCallback(() => {
    batchUpdateSelectedNotesVelocity(rootStore)({
      type: operationType,
      value,
    })
    rootViewStore.openVelocityDialog = false
  }, [rootViewStore, operationType, value])

  return (
    <Dialog open={open} style={{ minWidth: "20rem" }}>
      <DialogTitle>
        <Localized name="velocity" />
      </DialogTitle>
      <DialogContent>
        <Column>
          <StyledNumberInput
            value={value}
            onChange={setValue}
            style={{ flexGrow: 1 }}
            onEnter={onClickOK}
          />
          <Row>
            <RadioButton
              label={<Localized name="operation-set" />}
              isSelected={operationType === "set"}
              onClick={() => setOperationType("set")}
            />
            <RadioButton
              label={<Localized name="operation-add" />}
              isSelected={operationType === "add"}
              onClick={() => setOperationType("add")}
            />
            <RadioButton
              label={<Localized name="operation-multiply" />}
              isSelected={operationType === "multiply"}
              onClick={() => setOperationType("multiply")}
            />
          </Row>
        </Column>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>
          <Localized name="cancel" />
        </Button>
        <Button onClick={onClickOK}>
          <Localized name="ok" />
        </Button>
      </DialogActions>
    </Dialog>
  )
})
