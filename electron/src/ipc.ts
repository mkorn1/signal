import { BrowserWindow } from "electron"

export type IpcEvent =
  | { name: "onNewFile" }
  | { name: "onClickOpenFile" }
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
  | { name: "onOpenFile"; params: { filePath: string } }
  | { name: "onIdTokenReceived"; params: { idToken: string } }

export type ParamsForEvent<T extends IpcEvent["name"]> =
  Extract<IpcEvent, { name: T }> extends { params: infer P } ? P : undefined

export class Ipc {
  constructor(private readonly mainWindow: BrowserWindow) {}

  send<T extends IpcEvent["name"]>(name: T, params?: ParamsForEvent<T>): void {
    this.mainWindow.webContents.send(name, params)
  }
}
