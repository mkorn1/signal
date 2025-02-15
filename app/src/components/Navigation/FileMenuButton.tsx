import { useTheme } from "@emotion/react"
import ChevronRight from "mdi-react/ChevronRightIcon"
import CloudOutlined from "mdi-react/CloudOutlineIcon"
import KeyboardArrowDown from "mdi-react/KeyboardArrowDownIcon"
import { observer } from "mobx-react-lite"
import { FC, useCallback, useRef } from "react"
import { useExportSong } from "../../actions"
import { hasFSAccess } from "../../actions/file"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { Menu, MenuDivider, MenuItem, SubMenu } from "../ui/Menu"
import { CloudFileMenu } from "./CloudFileMenu"
import { FileMenu } from "./FileMenu"
import { LegacyFileMenu } from "./LegacyFileMenu"
import { Tab } from "./Navigation"

export const FileMenuButton: FC = observer(() => {
  const {
    rootViewStore,
    authStore: { authUser: user },
  } = useStores()
  const isOpen = rootViewStore.openFileDrawer
  const handleClose = () => (rootViewStore.openFileDrawer = false)
  const exportSong = useExportSong()
  const theme = useTheme()

  const onClickExportWav = () => {
    handleClose()
    exportSong("WAV")
  }

  const onClickExportMp3 = () => {
    handleClose()
    exportSong("MP3")
  }

  const ref = useRef<HTMLDivElement>(null)

  return (
    <Menu
      open={isOpen}
      onOpenChange={(open) => (rootViewStore.openFileDrawer = open)}
      trigger={
        <Tab
          ref={ref}
          onClick={useCallback(() => (rootViewStore.openFileDrawer = true), [])}
          id="tab-file"
        >
          <span style={{ marginLeft: "0.25rem" }}>
            <Localized name="file" />
          </span>
          <KeyboardArrowDown style={{ width: "1rem", marginLeft: "0.25rem" }} />
        </Tab>
      }
    >
      {user === null && hasFSAccess && <FileMenu close={handleClose} />}

      {user === null && !hasFSAccess && <LegacyFileMenu close={handleClose} />}

      {user && <CloudFileMenu close={handleClose} />}

      {user === null && (
        <>
          <MenuDivider />
          <MenuItem
            onClick={() => {
              handleClose()
              rootViewStore.openSignInDialog = true
            }}
          >
            <CloudOutlined style={{ marginRight: "0.5em" }} />
            <Localized name="please-sign-up" />
          </MenuItem>
        </>
      )}

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
})
