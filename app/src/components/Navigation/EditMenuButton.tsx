import CloudOutlined from "mdi-react/CloudOutlineIcon"
import KeyboardArrowDown from "mdi-react/KeyboardArrowDownIcon"
import { observer } from "mobx-react-lite"
import { FC, useCallback, useRef } from "react"
import { hasFSAccess } from "../../actions/file"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { Menu, MenuDivider, MenuItem } from "../ui/Menu"
import { EditMenu } from "./EditMenu"
import { Tab } from "./Navigation"

export const EditMenuButton: FC = observer(() => {
  const {
    rootViewStore,
    exportStore,
    authStore: { authUser: user },
  } = useStores()
  const isOpen = rootViewStore.openEditDrawer
  const handleClose = () => (rootViewStore.openEditDrawer = false)

  const ref = useRef<HTMLDivElement>(null)

  return (
    <Menu
      open={isOpen}
      onOpenChange={(open) => (rootViewStore.openEditDrawer = open)}
      trigger={
        <Tab
          ref={ref}
          onClick={useCallback(() => (rootViewStore.openEditDrawer = true), [])}
          id="tab-edit"
        >
          <span style={{ marginLeft: "0.25rem" }}>
            <Localized name="edit" />
          </span>
          <KeyboardArrowDown style={{ width: "1rem", marginLeft: "0.25rem" }} />
        </Tab>
      }
    >
      <EditMenu close={handleClose} />
    </Menu>
  )
})
