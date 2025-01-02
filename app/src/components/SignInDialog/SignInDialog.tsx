import { useToast } from "dialog-hooks"
import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import { useLocalization } from "../../localize/useLocalization"
import { SignInDialogContent } from "./SignInDialogContent"

export const SignInDialog: FC = observer(() => {
  const {
    rootViewStore,
    rootViewStore: { openSignInDialog },
  } = useStores()
  const toast = useToast()
  const localized = useLocalization()

  const onClose = useCallback(
    () => (rootViewStore.openSignInDialog = false),
    [rootViewStore],
  )

  const signInSuccessWithAuthResult = async () => {
    rootViewStore.openSignInDialog = false
    toast.success(localized["success-sign-in"])
  }

  const signInFailure = (error: firebaseui.auth.AuthUIError) => {
    console.warn(error)
  }

  return (
    <SignInDialogContent
      open={openSignInDialog}
      onClose={onClose}
      onSuccess={signInSuccessWithAuthResult}
      onFailure={signInFailure}
    />
  )
})
