import { useTheme } from "@emotion/react"
import ChevronRight from "mdi-react/ChevronRightIcon"
import { FC, useCallback, useState } from "react"
import { hasFSAccess } from "../../actions/file"
import { useExport } from "../../hooks/useExport"
import { Localized } from "../../localize/useLocalization"
import { Menu, MenuDivider, MenuItem, SubMenu } from "../ui/Menu"
import { FileMenu } from "./FileMenu"
import { LegacyFileMenu } from "./LegacyFileMenu"
import { NavButton } from "./Navigation"

export const FileMenuButton: FC = () => {
  const { exportSong } = useExport()
  const theme = useTheme()
  const [isOpen, setOpen] = useState(false)

  const handleClose = useCallback(() => setOpen(false), [])

  const onClickExportWav = useCallback(() => {
    handleClose()
    exportSong("WAV")
  }, [handleClose, exportSong])

  const onClickExportMp3 = useCallback(() => {
    handleClose()
    exportSong("MP3")
  }, [handleClose, exportSong])

  return (
    <Menu
      open={isOpen}
      onOpenChange={setOpen}
      trigger={
        <NavButton id="tab-file">
          <span>
            <Localized name="file" />
          </span>
        </NavButton>
      }
    >
      {hasFSAccess ? <FileMenu close={handleClose} /> : <LegacyFileMenu close={handleClose} />}

      <MenuDivider />

      <SubMenu
        trigger={
          <MenuItem>
            <Localized name="export" />
            <ChevronRight
              style={{ marginLeft: "auto", fill: theme.tertiaryTextColor }}
            />
          </MenuItem>
        }
      >
        <MenuItem onClick={onClickExportWav}>WAV</MenuItem>
        <MenuItem onClick={onClickExportMp3}>MP3</MenuItem>
      </SubMenu>
    </Menu>
  )
}
