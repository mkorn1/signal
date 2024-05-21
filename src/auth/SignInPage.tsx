import { FC, useState } from "react"
import { DialogContent, DialogTitle } from "../components/Dialog"

import styled from "@emotion/styled"
import "firebase/auth"
import { GithubAuthProvider, GoogleAuthProvider } from "firebase/auth"
import { Localized } from "../components/Localized"
import { auth } from "../firebase/firebase"
import { StyledFirebaseAuth } from "../main/components/FirebaseAuth/StyledFirebaseAuth"
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
        <Localized default="Sign in">sign-in</Localized>
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
                  const handle = window.open(url)
                  if (handle) {
                    setIsSucceeded(true)
                  } else {
                    alert("Failed to open the app. Please try again.")
                  }
                }
                return false
              },
            },
            signInFlow: "redirect",
          }}
          firebaseAuth={auth}
        />
      </DialogContent>
    </Container>
  )
}
