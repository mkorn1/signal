import { BrowserWindow } from "electron"

export type IpcEvent =
  | { name: "onOpenSetting" }
  | { name: "onOpenHelp" }
  | { name: "onNewFile"; params: void }
  | { name: "onOpenFile" }
  | { name: "onSaveFile" }
  | { name: "onSaveFileAs" }

type EventParams<T extends IpcEvent> = T extends { params: any }
  ? T["params"]
  : undefined

export class Ipc {
  constructor(private readonly mainWindow: BrowserWindow) {}

  send<T extends IpcEvent>(name: T["name"], params?: EventParams<T>): void {
    this.mainWindow.webContents.send(name, params)
  }
}
