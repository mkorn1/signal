import { contextBridge, ipcRenderer } from "electron"
import type { IpcEvent } from "./ipc"

function register<T extends IpcEvent>(
  name: T["name"],
  callback: (params: T["params"]) => void,
) {
  ipcRenderer.on(name, (_event, value) => callback(value))
}

const api = {
  onOpenSetting: (callback: () => void) => register("openSetting", callback),
  onOpenHelp: (callback: () => void) => register("openHelp", callback),
}

export type ElectronAPI = typeof api

contextBridge.exposeInMainWorld("electronAPI", api)
