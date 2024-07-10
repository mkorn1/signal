import { observer } from "mobx-react-lite"
import { FC, useCallback, useState } from "react"
import { addTimeSignature, setLoopBegin, setLoopEnd } from "../../actions"
import { useStores } from "../../hooks/useStores"
import { envString } from "../../localize/envString"
import { Localized } from "../../localize/useLocalization"
import { RulerStore } from "../../stores/RulerStore"
import {
  ContextMenu,
  ContextMenuProps,
  ContextMenuHotKey as HotKey,
} from "../ContextMenu/ContextMenu"
import { MenuItem } from "../ui/Menu"
import { TimeSignatureDialog } from "./TimeSignatureDialog"

export interface RulerContextMenuProps extends ContextMenuProps {
  rulerStore: RulerStore
  tick: number
}

export const RulerContextMenu: FC<RulerContextMenuProps> = (
  observer(({ rulerStore, tick, ...props }) => {
    const { handleClose } = props
    const rootStore = useStores()
    const { song, player } = rootStore
    const [isOpenTimeSignatureDialog, setOpenTimeSignatureDialog] =
      useState(false)

    const isTimeSignatureSelected =
      rulerStore.selectedTimeSignatureEventIds.length > 0

    const onClickAddTimeSignature = useCallback(() => {
      setOpenTimeSignatureDialog(true)
      handleClose()
    }, [])

    const onClickRemoveTimeSignature = useCallback(() => {
      song.conductorTrack?.removeEvents(
        rulerStore.selectedTimeSignatureEventIds,
      )
      handleClose()
    }, [song])

    const onClickSetLoopStart = useCallback(() => {
      setLoopBegin(rootStore)(tick)
      handleClose()
    }, [tick])

    const onClickSetLoopEnd = useCallback(() => {
      setLoopEnd(rootStore)(tick)
      handleClose()
    }, [tick])

    const closeOpenTimeSignatureDialog = useCallback(() => {
      setOpenTimeSignatureDialog(false)
    }, [])

    return (
      <>
        <ContextMenu {...props}>
          <MenuItem onClick={onClickSetLoopStart}>
            <Localized name="set-loop-start" />
            <HotKey>{envString.cmdOrCtrl}+Click</HotKey>
          </MenuItem>
          <MenuItem onClick={onClickSetLoopEnd}>
            <Localized name="set-loop-end" />
            <HotKey>Alt+Click</HotKey>
          </MenuItem>
          <MenuItem onClick={onClickAddTimeSignature}>
            <Localized name="add-time-signature" />
          </MenuItem>
          <MenuItem
            onClick={onClickRemoveTimeSignature}
            disabled={!isTimeSignatureSelected}
          >
            <Localized name="remove-time-signature" />
          </MenuItem>
        </ContextMenu>
        <TimeSignatureDialog
          open={isOpenTimeSignatureDialog}
          onClose={closeOpenTimeSignatureDialog}
          onClickOK={({ numerator, denominator }) => {
            addTimeSignature(rootStore)(player.position, numerator, denominator)
          }}
        />
      </>
    )
  }),
)
