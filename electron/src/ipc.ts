import { BrowserWindow } from "electron"

export type IpcEvent =
  | { name: "openSetting" }
  | { name: "openHelp" }
  | { name: "openFile" }
  | { name: "saveFile" }
  | { name: "saveFileAs" }

export class Ipc {
  constructor(private readonly mainWindow: BrowserWindow) {}

  invoke<T extends { name: string; params?: any }>(
    name: T["name"],
    params: T["params"] = undefined,
  ): void {
    this.mainWindow.webContents.send(name, params)
  }
}
