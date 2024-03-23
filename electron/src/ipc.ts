import { BrowserWindow } from "electron"

export type IpcEvent =
  | { name: "onNewFile" }
  | { name: "onOpenFile" }
  | { name: "onSaveFile" }
  | { name: "onSaveFileAs" }
  | { name: "onExportWav" }
  | { name: "onUndo" }
  | { name: "onRedo" }
  | { name: "onCut" }
  | { name: "onCopy" }
  | { name: "onPaste" }
  | { name: "onOpenSetting" }
  | { name: "onOpenHelp" }

type EventParams<T extends IpcEvent> = T extends { params: any }
  ? T["params"]
  : undefined

export class Ipc {
  constructor(private readonly mainWindow: BrowserWindow) {}

  send<T extends IpcEvent>(name: T["name"], params?: EventParams<T>): void {
    this.mainWindow.webContents.send(name, params)
  }
}
