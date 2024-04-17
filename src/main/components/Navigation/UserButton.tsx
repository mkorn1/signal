import AccountCircle from "mdi-react/AccountCircleIcon"
import { observer } from "mobx-react-lite"
import { FC, useRef, useState } from "react"
import { Localized } from "../../../components/Localized"
import { Menu, MenuItem } from "../../../components/Menu"
import { auth } from "../../../firebase/firebase"
import { isRunningInElectron } from "../../helpers/platform"
import { useStores } from "../../hooks/useStores"
import { useTheme } from "../../hooks/useTheme"
import { IconStyle, Tab, TabTitle } from "./Navigation"

export const UserButton: FC = observer(() => {
  const {
    rootViewStore,
    authStore: { authUser: user },
  } = useStores()

  const [open, setOpen] = useState(false)

  const onClickSignIn = () => {
    if (isRunningInElectron()) {
      window.electronAPI.openAuthWindow()
    } else {
      rootViewStore.openSignInDialog = true
    }
    setOpen(false)
  }

  const onClickSignOut = async () => {
    await auth.signOut()
    setOpen(false)
  }

  const onClickProfile = () => {
    if (user !== null) {
      window.open(`https://signal.vercel.app/users/${user.uid}`)
    }
    setOpen(false)
  }

  const onClickFiles = () => {
    rootViewStore.openCloudFileDialog = true
    setOpen(false)
  }

  const theme = useTheme()
  const ref = useRef<HTMLDivElement>(null)

  if (user === null) {
    return (
      <Tab onClick={onClickSignIn}>
        <AccountCircle style={IconStyle} />
        <TabTitle>
          <Localized default="Sign in">sign-in</Localized>
        </TabTitle>
      </Tab>
    )
  }

  return (
    <Menu
      open={open}
      onOpenChange={setOpen}
      trigger={
        <Tab ref={ref}>
          <img
            style={{
              ...IconStyle,
              borderRadius: "0.65rem",
              border: `1px solid ${theme.dividerColor}`,
            }}
            src={user.photoURL ?? undefined}
          />
          <TabTitle>{user.displayName}</TabTitle>
        </Tab>
      }
    >
      <MenuItem onClick={onClickProfile}>
        <Localized default="Profile">profile</Localized>
      </MenuItem>

      <MenuItem onClick={onClickSignOut}>
        <Localized default="Sign out">sign-out</Localized>
      </MenuItem>
    </Menu>
  )
})
