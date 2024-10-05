import { MenuItemConstructorOptions, app } from "electron"

const isMac = process.platform === "darwin"

export interface MenuTemplateProps {
  isLoggedIn: boolean
  onClickNew: () => void
  onClickOpen: () => void
  onClickSave: () => void
  onClickSaveAs: () => void
  onClickRename: () => void
  onClickImport: () => void
  onClickExportWav: () => void
  onClickUndo: () => void
  onClickRedo: () => void
  onClickCut: () => void
  onClickCopy: () => void
  onClickPaste: () => void
  onClickSetting: () => void
  onClickHelp: () => void
  onClickSupport: () => void
}

export const menuTemplate = ({
  isLoggedIn,
  onClickNew,
  onClickOpen,
  onClickSave,
  onClickSaveAs,
  onClickRename,
  onClickImport,
  onClickExportWav,
  onClickUndo,
  onClickRedo,
  onClickCut,
  onClickCopy,
  onClickPaste,
  onClickSetting,
  onClickHelp,
  onClickSupport,
}: MenuTemplateProps): MenuItemConstructorOptions[] => [
  // { role: 'appMenu' }
  ...((isMac
    ? [
        {
          label: app.name,
          submenu: [
            { role: "about" },
            { type: "separator" },
            {
              label: "Settings",
              click: onClickSetting,
            },
            { type: "separator" },
            { role: "services" },
            { type: "separator" },
            { role: "hide" },
            { role: "hideothers" },
            { role: "unhide" },
            { type: "separator" },
            { role: "quit" },
          ],
        },
      ]
    : []) as MenuItemConstructorOptions[]),
  // { role: 'fileMenu' }
  {
    label: "File",
    submenu: [
      { label: "New", accelerator: "CmdOrCtrl+N", click: onClickNew },
      { label: "Open", accelerator: "CmdOrCtrl+O", click: onClickOpen },
      { label: "Save", accelerator: "CmdOrCtrl+S", click: onClickSave },
      {
        label: "Save As",
        accelerator: "CmdOrCtrl+Shift+S",
        click: onClickSaveAs,
      },
      ...(isLoggedIn ? [{ label: "Rename", click: onClickRename }] : []),
      { type: "separator" },
      ...(isLoggedIn ? [{ label: "Import", click: onClickImport }] : []),
      { label: "Export", submenu: [{ label: "WAV", click: onClickExportWav }] },
      { type: "separator" },
      {
        label: "Settings",
        click: onClickSetting,
      },
      { type: "separator" },
      isMac ? { role: "close" } : { role: "quit" },
    ] as MenuItemConstructorOptions[],
  },
  // { role: 'editMenu' }
  {
    label: "Edit",
    submenu: [
      { label: "Undo", accelerator: "CmdOrCtrl+Z", click: onClickUndo },
      { label: "Redo", accelerator: "CmdOrCtrl+Y", click: onClickRedo },
      { type: "separator" },
      { label: "Cut", accelerator: "CmdOrCtrl+X", click: onClickCut },
      { label: "Copy", accelerator: "CmdOrCtrl+C", click: onClickCopy },
      { label: "Paste", accelerator: "CmdOrCtrl+V", click: onClickPaste },
    ],
  },
  // { role: 'viewMenu' }
  {
    label: "View",
    submenu: [
      { role: "reload" },
      { role: "forceReload" },
      { role: "toggleDevTools" },
      { type: "separator" },
      { role: "resetZoom" },
      { role: "zoomIn" },
      { role: "zoomOut" },
      { type: "separator" },
      { role: "togglefullscreen" },
    ],
  },
  // { role: 'windowMenu' }
  {
    label: "Window",
    submenu: [
      { role: "minimize" },
      { role: "zoom" },
      ...((isMac
        ? [
            { type: "separator" },
            { role: "front" },
            { type: "separator" },
            { role: "window" },
          ]
        : [{ role: "close" }]) as MenuItemConstructorOptions[]),
    ],
  },
  {
    role: "help",
    submenu: [
      {
        label: "Help",
        click: onClickHelp,
      },
      {
        label: "Support",
        click: onClickSupport,
      },
    ],
  },
]
