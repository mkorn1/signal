import { app, BrowserWindow, Menu, shell } from "electron"
import path from "path"
import { getArgument } from "./arguments"
import { defaultMenuTemplate } from "./defaultMenu"
import { Ipc } from "./ipc"
import { registerIpcMain } from "./ipcMain"
import { menuTemplate } from "./menu"

let onOpenFile: (filePath: string) => void = () => {}
let onDropFileOnAppIcon: (filePath: string) => void = () => {}
let mainWindow: BrowserWindow
let mainMenu: Electron.Menu

const createWindow = (): void => {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 960,
    height: 720,
    title: `signal v${app.getVersion()}`,
    titleBarStyle: "hidden",
    trafficLightPosition: { x: 10, y: 17 },
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false,
      preload: path.join(__dirname, "..", "dist_preload", "preload.js"),
    },
  })

  // and load the index.html of the app.
  if (!app.isPackaged) {
    mainWindow.loadURL("http://localhost:3000/edit")
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(
      path.join(__dirname, "..", "dist_renderer", "edit.html"),
    )
  }

  const ipc = new Ipc(mainWindow)

  mainMenu = Menu.buildFromTemplate(
    menuTemplate({
      onClickNew: () => ipc.send("onNewFile"),
      onClickOpen: async () => ipc.send("onClickOpenFile"),
      onClickSave: () => ipc.send("onSaveFile"),
      onClickSaveAs: () => ipc.send("onSaveFileAs"),
      onClickExportWav: () => ipc.send("onExportWav"),
      onClickUndo: () => ipc.send("onUndo"),
      onClickRedo: () => ipc.send("onRedo"),
      onClickCut: () => ipc.send("onCut"),
      onClickCopy: () => ipc.send("onCopy"),
      onClickPaste: () => ipc.send("onPaste"),
      onClickSetting: () => ipc.send("onOpenSetting"),
      onClickHelp: () => ipc.send("onOpenHelp"),
    }),
  )
  Menu.setApplicationMenu(mainMenu)

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (
      url.startsWith("http://localhost:9099/emulator/auth/") ||
      url.startsWith("https://signal-9546d.firebaseapp.com/__/auth/")
    ) {
      return { action: "allow" }
    }
    if (url.startsWith("http")) {
      shell.openExternal(url)
    }
    return { action: "deny" }
  })

  onDropFileOnAppIcon = (filePath) => {
    ipc.send("onOpenFile", { filePath })
  }

  onOpenFile = (filePath) => {
    ipc.send("onOpenFile", { filePath })
  }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on("ready", createWindow)

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit()
  }
})

app.on("activate", () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

const additionalData = { filePath: getArgument() }
type AdditionalData = typeof additionalData
const gotTheLock = app.requestSingleInstanceLock(additionalData)

if (!gotTheLock) {
  app.quit()
} else {
  app.on("second-instance", (event, argv, workingDirectory, additionalData) => {
    const { filePath } = additionalData as AdditionalData
    if (filePath !== null) {
      onDropFileOnAppIcon(filePath)
    }
  })
}

app.on("open-file", (event, filePath) => {
  event.preventDefault()
  onOpenFile(filePath)
})

app.on("browser-window-focus", (event, window) => {
  const defaultMenu = Menu.buildFromTemplate(defaultMenuTemplate)
  Menu.setApplicationMenu(window === mainWindow ? mainMenu : defaultMenu)
})

registerIpcMain()
