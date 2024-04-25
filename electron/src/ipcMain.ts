import { IpcMainInvokeEvent, app, dialog, ipcMain, shell } from "electron"
import { readFile, readdir, writeFile } from "fs/promises"
import { getPort } from "get-port-please"
import { isAbsolute, join } from "path"
import { getArgument } from "./arguments"
import { launchAuthCallbackServer } from "./authCallback"
import { Ipc } from "./ipc"

interface Callbacks {
  onAuthStateChanged: (isLoggedIn: boolean) => void
}

const api = (ipc: Ipc, { onAuthStateChanged }: Callbacks) => ({
  showOpenDialog: async () => {
    const fileObj = await dialog.showOpenDialog({
      properties: ["openFile"],
      filters: [{ name: "MIDI File", extensions: ["mid", "midi"] }],
    })
    if (fileObj.canceled) {
      return null
    }
    const path = fileObj.filePaths[0]
    const content = await readFile(path)
    return { path, content: content.buffer }
  },
  showOpenDirectoryDialog: async () => {
    const fileObj = await dialog.showOpenDialog({
      properties: ["openDirectory"],
    })
    if (fileObj.canceled) {
      return null
    }
    const path = fileObj.filePaths[0]
    return path
  },
  saveFile: async (_e: IpcMainInvokeEvent, path: string, data: ArrayBuffer) => {
    await writeFile(path, Buffer.from(data))
  },
  readFile: async (_e: IpcMainInvokeEvent, path: string) => {
    const filePath = isAbsolute(path) ? path : join(app.getAppPath(), path)
    const content = await readFile(filePath)
    return content.buffer
  },
  searchSoundFonts: async (_e: IpcMainInvokeEvent, path: string) => {
    const files = await readdir(path, { withFileTypes: true })
    return files
      .filter((f) => f.isFile() && f.name.endsWith(".sf2"))
      .map((f) => join(f.path, f.name))
  },
  showSaveDialog: async () => {
    const fileObj = await dialog.showSaveDialog({
      filters: [{ name: "MIDI File", extensions: ["mid", "midi"] }],
    })
    if (fileObj.canceled) {
      return null
    }
    const path = fileObj.filePath
    if (!path) {
      return null
    }
    return { path }
  },
  addRecentDocument: (_e: IpcMainInvokeEvent, path: string) => {
    app.addRecentDocument(path)
  },
  getArgument: async () => getArgument(),
  openAuthWindow: async () => {
    const port = await getPort()
    let closeTimeout: NodeJS.Timeout

    const server = launchAuthCallbackServer({
      port,
      onComplete: (credential) => {
        server.close()
        clearTimeout(closeTimeout)
        ipc.send("onBrowserSignInCompleted", { credential })
      },
    })

    const parameter = `redirect_uri=http://localhost:${port}`

    shell.openExternal(
      app.isPackaged
        ? `https://signal.vercel.app/auth?${parameter}`
        : `http://localhost:3000/auth?${parameter}`,
    )

    // close server after 5 minutes
    closeTimeout = setTimeout(
      () => {
        if (server.listening) {
          server.close()
        }
      },
      1000 * 60 * 5,
    )
  },
  authStateChanged: (_e: IpcMainInvokeEvent, isLoggedIn: boolean) => {
    onAuthStateChanged(isLoggedIn)
  },
})

export type IpcMainAPI = ReturnType<typeof api>

export const registerIpcMain = (ipc: Ipc, callbacks: Callbacks) => {
  Object.entries(api(ipc, callbacks)).forEach(([name, func]) => {
    ipcMain.handle(name, func)
  })
}
