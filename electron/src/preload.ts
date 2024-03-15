import { contextBridge, ipcRenderer } from "electron"
import type { IpcEvent } from "./ipc"

function register<T extends IpcEvent>(
  name: T["name"],
  callback: (
    params: T extends { params: unknown } ? T["params"] : void,
  ) => void,
) {
  ipcRenderer.on(name, (_event, value) => callback(value))
}

const api = {
  onOpenFile: (callback: () => void) => register("openFile", callback),
  onOpenSetting: (callback: () => void) => register("openSetting", callback),
  onOpenHelp: (callback: () => void) => register("openHelp", callback),
  showOpenDialog: async () => await ipcRenderer.invoke("showOpenDialog"),
}

export type ElectronAPI = typeof api

contextBridge.exposeInMainWorld("electronAPI", api)
