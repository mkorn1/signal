import { BrowserWindow } from "electron"

export type IpcEvent =
  | {
      name: "openSetting"
      params: void
    }
  | {
      name: "openHelp"
      params: void
    }

export class Ipc {
  constructor(private readonly mainWindow: BrowserWindow) {}

  openSetting(): void {
    this.invoke("openSetting")
  }

  openHelp(): void {
    this.invoke("openHelp")
  }

  invoke<T extends IpcEvent>(
    name: T["name"],
    params: T["params"] = undefined,
  ): void {
    this.mainWindow.webContents.send(name, params)
  }
}
