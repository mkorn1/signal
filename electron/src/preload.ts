import { contextBridge, ipcRenderer } from "electron"
import { ElectronAPI } from "./ElectronAPI"
import type { IpcEvent, ParamsForEvent } from "./ipc"
import type { IpcMainAPI } from "./ipcMain"

type Tail<T extends unknown[]> = T extends [any, ...infer Rest] ? Rest : []

function register<T extends IpcEvent["name"]>(
  name: T,
  callback: (params: ParamsForEvent<T>) => void,
) {
  const listener = (
    _event: Electron.IpcRendererEvent,
    value: ParamsForEvent<T>,
  ) => {
    callback(value)
  }
  ipcRenderer.on(name, listener)
  return () => {
    ipcRenderer.removeListener(name, listener)
  }
}

function invoke<T extends keyof IpcMainAPI>(
  name: T,
  ...params: Tail<Parameters<IpcMainAPI[T]>>
) {
  return ipcRenderer.invoke(name, ...params) as ReturnType<IpcMainAPI[T]>
}

const api: ElectronAPI = {
  onNewFile: (callback) => register("onNewFile", callback),
  onClickOpenFile: (callback) => register("onClickOpenFile", callback),
  onOpenFile: (callback) => register("onOpenFile", callback),
  onSaveFile: (callback) => register("onSaveFile", callback),
  onSaveFileAs: (callback) => register("onSaveFileAs", callback),
  onRename: (callback) => register("onRename", callback),
  onImport: (callback) => register("onImport", callback),
  onExportWav: (callback) => register("onExportWav", callback),
  onUndo: (callback) => register("onUndo", callback),
  onRedo: (callback) => register("onRedo", callback),
  onCut: (callback) => register("onCut", callback),
  onCopy: (callback) => register("onCopy", callback),
  onPaste: (callback) => register("onPaste", callback),
  onOpenSetting: (callback) => register("onOpenSetting", callback),
  onOpenHelp: (callback) => register("onOpenHelp", callback),
  onBrowserSignInCompleted: (callback) =>
    register("onBrowserSignInCompleted", callback),
  // tell to main process that the renderer process is ready
  ready: () => invoke("ready"),
  showOpenDialog: async () => await invoke("showOpenDialog"),
  showOpenDirectoryDialog: async () => await invoke("showOpenDirectoryDialog"),
  showSaveDialog: async () => await invoke("showSaveDialog"),
  saveFile: async (path, data) => await invoke("saveFile", path, data),
  readFile: async (path) => await invoke("readFile", path),
  searchSoundFonts: async (path) => await invoke("searchSoundFonts", path),
  addRecentDocument: (path) => invoke("addRecentDocument", path),
  getArgument: () => invoke("getArgument"),
  openAuthWindow: async () => await invoke("openAuthWindow"),
  authStateChanged: (isLoggedIn) => invoke("authStateChanged", isLoggedIn),
  closeMainWindow: () => invoke("closeMainWindow"),
}

contextBridge.exposeInMainWorld("electronAPI", api)
