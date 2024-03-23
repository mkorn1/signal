import { IpcMainInvokeEvent, dialog, ipcMain } from "electron"
import { readFile, writeFile } from "fs/promises"

const api = {
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
  saveFile: async (_e: IpcMainInvokeEvent, path: string, data: ArrayBuffer) => {
    await writeFile(path, Buffer.from(data))
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
}

export type IpcMainAPI = typeof api

export const registerIpcMain = () => {
  Object.entries(api).forEach(([name, func]) => {
    ipcMain.handle(name, func)
  })
}
