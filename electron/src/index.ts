import { app, BrowserWindow, Menu, shell } from "electron"
import isDev from "electron-is-dev"
import path from "path"
import { fileURLToPath } from "url"
import { Ipc } from "./ipc.js"
import { menuTemplate } from "./menu.js"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const createWindow = (): void => {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 900,
    height: 700,
    title: `signal v${app.getVersion()}`,
    titleBarStyle: "hidden",
    trafficLightPosition: { x: 10, y: 17 },
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, "..", "dist_preload", "preload.js"),
    },
  })

  // and load the index.html of the app.
  if (isDev) {
    mainWindow.loadURL("http://localhost:3000/edit")
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(path.join(__dirname, "../build/index.html"))
  }

  const ipc = new Ipc(mainWindow)

  const menu = Menu.buildFromTemplate(
    menuTemplate({
      onClickSetting: () => ipc.openSetting(),
      onClickHelp: () => ipc.openHelp(),
    }),
  )
  Menu.setApplicationMenu(menu)

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (url.startsWith("http")) {
      shell.openExternal(url)
    }
    return { action: "deny" }
  })
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
