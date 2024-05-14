import { FC } from "react"
import { DialogContent, DialogTitle } from "../components/Dialog"

import styled from "@emotion/styled"
import "firebase/auth"
import { GithubAuthProvider, GoogleAuthProvider } from "firebase/auth"
import { Localized } from "../components/Localized"
import { auth } from "../firebase/firebase"
import { StyledFirebaseAuth } from "../main/components/FirebaseAuth/StyledFirebaseAuth"

const Container = styled.div`
  padding: 2rem 3rem;
`

export const SignInPage: FC = () => {
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
                  location.href =
                    redirectUrl + "?credential=" + JSON.stringify(credential)
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
