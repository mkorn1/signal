import { BrowserWindow } from "electron"

interface IpcEventBase<T extends string, P = void> {
  name: T
  params?: P
}

export type IpcEvent =
  | IpcEventBase<"openSetting">
  | IpcEventBase<"openHelp">
  | IpcEventBase<"openFile">
  | IpcEventBase<"saveFile">
  | IpcEventBase<"saveFileAs">

export class Ipc {
  constructor(private readonly mainWindow: BrowserWindow) {}

  invoke<T extends IpcEvent>(
    name: T["name"],
    params: T["params"] = undefined,
  ): void {
    this.mainWindow.webContents.send(name, params)
  }
}
