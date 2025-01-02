import CloudOutlined from "mdi-react/CloudOutlineIcon"
import KeyboardArrowDown from "mdi-react/KeyboardArrowDownIcon"
import { observer } from "mobx-react-lite"
import { FC, useCallback, useRef } from "react"
import { hasFSAccess } from "../../actions/file"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { Menu, MenuDivider, MenuItem } from "../ui/Menu"
import { CloudFileMenu } from "./CloudFileMenu"
import { FileMenu } from "./FileMenu"
import { LegacyFileMenu } from "./LegacyFileMenu"
import { Tab } from "./Navigation"

export const FileMenuButton: FC = observer(() => {
  const {
    rootViewStore,
    exportStore,
    authStore: { authUser: user },
  } = useStores()
  const isOpen = rootViewStore.openDrawer
  const handleClose = () => (rootViewStore.openDrawer = false)

  const onClickExport = () => {
    handleClose()
    exportStore.openExportDialog = true
  }

  const ref = useRef<HTMLDivElement>(null)

  return (
    <Menu
      open={isOpen}
      onOpenChange={(open) => (rootViewStore.openDrawer = open)}
      trigger={
        <Tab
          ref={ref}
          onClick={useCallback(() => (rootViewStore.openDrawer = true), [])}
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

      <MenuItem onClick={onClickExport}>
        <Localized name="export-audio" />
      </MenuItem>
    </Menu>
  )
})
