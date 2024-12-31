import { FC, useEffect, useState } from "react"
import { ElectronAPI } from "../../../../electron/src/ElectronAPI"
import { FirebaseCredential } from "../../../../electron/src/FirebaseCredential"

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}

export interface ElectronCallbackProps {
  onNewFile: () => void
  onClickOpenFile: () => void
  onOpenFile: (param: { filePath: string }) => void
  onSaveFile: () => void
  onSaveFileAs: () => void
  onRename: () => void
  onImport: () => void
  onExportWav: () => void
  onUndo: () => void
  onRedo: () => void
  onCut: () => void
  onCopy: () => void
  onPaste: () => void
  onOpenSetting: () => void
  onOpenHelp: () => void
  onBrowserSignInCompleted: (param: { credential: FirebaseCredential }) => void
}

export const ElectronCallback: FC<ElectronCallbackProps> = ({
  onNewFile,
  onClickOpenFile,
  onOpenFile,
  onSaveFile,
  onSaveFileAs,
  onRename,
  onImport,
  onExportWav,
  onUndo,
  onRedo,
  onCut,
  onCopy,
  onPaste,
  onOpenSetting,
  onOpenHelp,
  onBrowserSignInCompleted,
}) => {
  const [isInitialized, setIsInitialized] = useState(false)

  useEffect(() => window.electronAPI.onNewFile(onNewFile), [onNewFile])
  useEffect(
    () => window.electronAPI.onClickOpenFile(onClickOpenFile),
    [onClickOpenFile],
  )
  useEffect(() => window.electronAPI.onOpenFile(onOpenFile), [onOpenFile])
  useEffect(() => window.electronAPI.onSaveFile(onSaveFile), [onSaveFile])
  useEffect(() => window.electronAPI.onSaveFileAs(onSaveFileAs), [onSaveFileAs])
  useEffect(() => window.electronAPI.onRename(onRename), [onRename])
  useEffect(() => window.electronAPI.onImport(onImport), [onImport])
  useEffect(() => window.electronAPI.onExportWav(onExportWav), [onExportWav])
  useEffect(() => window.electronAPI.onUndo(onUndo), [onUndo])
  useEffect(() => window.electronAPI.onRedo(onRedo), [onRedo])
  useEffect(() => window.electronAPI.onCut(onCut), [onCut])
  useEffect(() => window.electronAPI.onCopy(onCopy), [onCopy])
  useEffect(() => window.electronAPI.onPaste(onPaste), [onPaste])
  useEffect(
    () => window.electronAPI.onOpenSetting(onOpenSetting),
    [onOpenSetting],
  )
  useEffect(() => window.electronAPI.onOpenHelp(onOpenHelp), [onOpenHelp])
  useEffect(
    () => window.electronAPI.onBrowserSignInCompleted(onBrowserSignInCompleted),
    [onBrowserSignInCompleted],
  )

  useEffect(() => {
    if (!isInitialized) {
      setIsInitialized(true)
      window.electronAPI.ready()
    }
  }, [])

  return <></>
}
