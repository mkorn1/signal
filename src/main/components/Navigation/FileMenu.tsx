import { observer } from "mobx-react-lite"
import { FC } from "react"
import { Localized } from "../../../components/Localized"
import { MenuDivider, MenuItem } from "../../../components/Menu"
import { useSongFile } from "../../hooks/useSongFile"
import { useStores } from "../../hooks/useStores"

export const FileMenu: FC<{ close: () => void }> = observer(({ close }) => {
  const rootStore = useStores()
  const { createNewSong, openSong, saveSong, saveAsSong, downloadSong } =
    useSongFile()

  const onClickNew = async () => {
    close()
    await createNewSong()
  }

  const onClickOpen = async () => {
    close()
    await openSong()
  }

  const onClickSave = async () => {
    close()
    await saveSong()
  }

  const onClickSaveAs = async () => {
    close()
    await saveAsSong()
  }

  const onClickDownload = async () => {
    close()
    await downloadSong()
  }

  return (
    <>
      <MenuItem onClick={onClickNew}>
        <Localized default="New">new-song</Localized>
      </MenuItem>

      <MenuDivider />

      <MenuItem onClick={onClickOpen}>
        <Localized default="Open">open-song</Localized>
      </MenuItem>

      <MenuItem
        onClick={onClickSave}
        disabled={rootStore.song.fileHandle === null}
      >
        <Localized default="Save">save-song</Localized>
      </MenuItem>

      <MenuItem onClick={onClickSaveAs}>
        <Localized default="Save As">save-as</Localized>
      </MenuItem>

      <MenuItem onClick={onClickDownload}>
        <Localized default="Download MIDI File">download-midi</Localized>
      </MenuItem>
    </>
  )
})
