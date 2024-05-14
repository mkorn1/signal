import { app, shell } from "electron"
import { authCallbackUrl } from "./scheme"

const authURL = (redirectUri: string) => {
  const parameter = `redirect_uri=${redirectUri}`

  return app.isPackaged
    ? `https://signal.vercel.app/auth?${parameter}`
    : `http://localhost:3000/auth?${parameter}`
}

export const signInWithBrowser = async () => {
  const url = authURL(authCallbackUrl)
  shell.openExternal(url)
}
