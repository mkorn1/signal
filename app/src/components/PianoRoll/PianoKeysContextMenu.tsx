import { observer } from "mobx-react-lite"
import { FC, useState } from "react"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { ContextMenu, ContextMenuProps } from "../ContextMenu/ContextMenu"
import { KeySignatureDialog } from "../KeySignatureDialog/KeySignatureDialog"
import { MenuItem } from "../ui/Menu"

export const PianoKeysContextMenu: FC<ContextMenuProps> = observer((props) => {
  const { handleClose } = props
  const rootStore = useStores()
  const { pianoRollStore } = rootStore
  const [isKeySignatureDialogOpen, setKeySignatureDialogOpen] = useState(false)

  const onClickShowScale = () => {
    if (pianoRollStore.keySignature === null) {
      pianoRollStore.keySignature = {
        scale: "major",
        key: 0,
      }
    }

    setKeySignatureDialogOpen(true)
    handleClose()
  }

  const onClickHideScale = () => {
    pianoRollStore.keySignature = null
    handleClose()
  }

  return (
    <>
      <ContextMenu {...props}>
        <MenuItem onClick={onClickShowScale}>
          <Localized name="show-scale" />
        </MenuItem>
        <MenuItem
          onClick={onClickHideScale}
          disabled={pianoRollStore.keySignature === null}
        >
          <Localized name="hide-scale" />
        </MenuItem>
      </ContextMenu>
      <KeySignatureDialog
        open={isKeySignatureDialogOpen}
        onOpenChange={setKeySignatureDialogOpen}
      />
    </>
  )
})
