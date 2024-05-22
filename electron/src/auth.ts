import { execFile } from "child_process"
import { app, shell } from "electron"
import log from "electron-log"
import path from "path"
import { appScheme, authCallbackUrl } from "./scheme"

const authURL = (redirectUri: string) => {
  const parameter = `redirect_uri=${redirectUri}`

  return app.isPackaged
    ? `https://signal.vercel.app/auth?${parameter}`
    : `http://localhost:3000/auth?${parameter}`
}

// In the mas build, return callbackURL when authentication is completed, and do not return anything when the application is launched with a schema.
export const signInWithBrowser = async (): Promise<string | null> => {
  if (process.mas) {
    const callbackURL = await startAuthSession(
      authURL(authCallbackUrl),
      appScheme,
    )
    return callbackURL
  }
  const url = authURL(authCallbackUrl)
  shell.openExternal(url)
  return null
}

const startAuthSession = async (
  url: string,
  callbackURLScheme: string,
): Promise<string> => {
  return new Promise((resolve, reject) => {
    log.info("electron:auth:startAuthSession", url, callbackURLScheme)
    execFile(
      path.join(__dirname, "..", "resources", "AuthSession_mac"),
      [url, callbackURLScheme],
      (error, stdout, stderr) => {
        if (error) {
          log.error("electron:auth:startAuthSession", error, stderr)
          reject(error)
        } else {
          log.info("electron:auth:startAuthSession", stdout)
          const callbackURL = stdout.trim()
          resolve(callbackURL)
        }
      },
    )
  })
}
