import { observer } from "mobx-react-lite"
import React, { FC, useCallback, useState } from "react"
import { envString } from "../../../common/localize/envString"
import { Localized } from "../../../common/localize/useLocalization"
import {
  ContextMenu,
  ContextMenuProps,
  ContextMenuHotKey as HotKey,
} from "../../../components/ContextMenu"
import { MenuItem } from "../../../components/Menu"
import { addTimeSignature, setLoopBegin, setLoopEnd } from "../../actions"
import { useStores } from "../../hooks/useStores"
import { RulerStore } from "../../stores/RulerStore"
import { TimeSignatureDialog } from "./TimeSignatureDialog"

export interface RulerContextMenuProps extends ContextMenuProps {
  rulerStore: RulerStore
  tick: number
}

export const RulerContextMenu: FC<RulerContextMenuProps> = React.memo(
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
