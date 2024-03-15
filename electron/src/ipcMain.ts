import { dialog, ipcMain } from "electron"
import { readFile } from "fs/promises"

ipcMain.handle("showOpenDialog", async () => {
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
})
