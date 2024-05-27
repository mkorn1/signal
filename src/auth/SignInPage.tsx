import { FC, useState } from "react"
import { DialogContent, DialogTitle } from "../main/components/Dialog/Dialog"

import styled from "@emotion/styled"
import "firebase/auth"
import { GithubAuthProvider, GoogleAuthProvider } from "firebase/auth"
import { auth } from "../firebase/firebase"
import { StyledFirebaseAuth } from "../main/components/FirebaseAuth/StyledFirebaseAuth"
import { Localized } from "../main/localize/useLocalization"
import { SignInSuccessPage } from "./SignInSuccessPage"

const Container = styled.div`
  padding: 2rem 3rem;
`

export const SignInPage: FC = () => {
  const [isSucceeded, setIsSucceeded] = useState(false)

  if (isSucceeded) {
    return <SignInSuccessPage />
  }

  return (
    <Container>
      <DialogTitle>
        <Localized name="sign-in" />
      </DialogTitle>
      <DialogContent>
        <StyledFirebaseAuth
          uiConfig={{
            signInOptions: [
              GoogleAuthProvider.PROVIDER_ID,
              GithubAuthProvider.PROVIDER_ID,
              "apple.com",
            ],
            callbacks: {
              signInSuccessWithAuthResult: ({ credential }) => {
                const redirectUrl = new URLSearchParams(location.search).get(
                  "redirect_uri",
                )
                if (
                  redirectUrl &&
                  (redirectUrl.startsWith("jp.codingcafe.signal://") ||
                    redirectUrl.startsWith("jp.codingcafe.signal.dev://"))
                ) {
                  const url =
                    redirectUrl + "?credential=" + JSON.stringify(credential)

                  try {
                    location.assign(url)
                    setIsSucceeded(true)
                  } catch {
                    alert("Failed to open the app. Please try again.")
                  }
                }
                return false
              },
              signInFailure(error) {
                console.error(error)
                alert("Failed to sign in. Please try again.")
              },
            },
            signInFlow: "popup",
          }}
          firebaseAuth={auth}
        />
      </DialogContent>
    </Container>
  )
}
