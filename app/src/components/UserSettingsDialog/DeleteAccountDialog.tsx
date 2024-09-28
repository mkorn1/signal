import { DialogTitle } from "@radix-ui/react-dialog"
import { observer } from "mobx-react-lite"
import { FC } from "react"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { Dialog, DialogActions, DialogContent } from "../Dialog/Dialog"
import { Button, PrimaryButton } from "../ui/Button"

export const DeleteAccountDialog: FC = observer(() => {
  const { rootViewStore, userRepository } = useStores()

  const onClickCancel = () => {
    rootViewStore.openDeleteAccountDialog = false
  }

  const onClickDelete = async () => {
    try {
      await userRepository.delete()
      rootViewStore.openDeleteAccountDialog = false
    } catch (e) {
      alert(`Failed to delete account: ${e}`)
    }
  }

  return (
    <Dialog open={rootViewStore.openDeleteAccountDialog}>
      <DialogTitle>
        <Localized name="delete-account" />
      </DialogTitle>
      <DialogContent>
        <Localized name="delete-account-description" />
      </DialogContent>
      <DialogActions>
        <PrimaryButton onClick={onClickDelete}>
          <Localized name="delete" />
        </PrimaryButton>
        <Button onClick={onClickCancel}>
          <Localized name="cancel" />
        </Button>
      </DialogActions>
    </Dialog>
  )
})
