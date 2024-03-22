import { contextBridge, ipcRenderer } from "electron"
import type { IpcEvent } from "./ipc"
import type { IpcMainAPI } from "./ipcMain"

function register<T extends IpcEvent>(
  name: T["name"],
  callback: (
    params: T extends { params: unknown } ? T["params"] : void,
  ) => void,
) {
  ipcRenderer.on(name, (_event, value) => callback(value))
}

function invoke<T extends keyof IpcMainAPI>(
  name: T,
  ...params: Parameters<IpcMainAPI[T]>
) {
  return ipcRenderer.invoke(name, ...params)
}

const api = {
  onOpenFile: (callback: () => void) => register("openFile", callback),
  onOpenSetting: (callback: () => void) => register("openSetting", callback),
  onOpenHelp: (callback: () => void) => register("openHelp", callback),
  showOpenDialog: async () => await invoke("showOpenDialog"),
}

export type ElectronAPI = typeof api

contextBridge.exposeInMainWorld("electronAPI", api)
