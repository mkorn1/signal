import React, { FC, useState } from "react"
import { useStores } from "../../hooks/useStores"
import { ContextMenu, ContextMenuProps } from "../ContextMenu/ContextMenu"
import { KeySignatureDialog } from "../KeySignatureDialog/KeySignatureDialog"
import { MenuItem } from "../ui/Menu"

export const PianoKeysContextMenu: FC<ContextMenuProps> = React.memo(
  (props) => {
    const { handleClose } = props
    const rootStore = useStores()
    const { pianoRollStore } = rootStore
    const [isKeySignatureDialogOpen, setKeySignatureDialogOpen] =
      useState(false)

    const onClickScale = () => {
      setKeySignatureDialogOpen(true)
      handleClose()
    }

    return (
      <>
        <ContextMenu {...props}>
          <MenuItem onClick={onClickScale}>
            スケールを表示する
            {/* <Localized name="cut" /> */}
          </MenuItem>
          <MenuItem onClick={onClickScale} disabled={true}>
            スケールを隠す
            {/* <Localized name="cut" /> */}
          </MenuItem>
        </ContextMenu>
        <KeySignatureDialog
          open={isKeySignatureDialogOpen}
          onOpenChange={setKeySignatureDialogOpen}
        />
      </>
    )
  },
)
