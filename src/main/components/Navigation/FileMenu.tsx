import { observer } from "mobx-react-lite"
import { FC } from "react"
import { Localized } from "../../../common/localize/useLocalization"
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
        <Localized name="new-song" />
      </MenuItem>

      <MenuDivider />

      <MenuItem onClick={onClickOpen}>
        <Localized name="open-song" />
      </MenuItem>

      <MenuItem
        onClick={onClickSave}
        disabled={rootStore.song.fileHandle === null}
      >
        <Localized name="save-song" />
      </MenuItem>

      <MenuItem onClick={onClickSaveAs}>
        <Localized name="save-as" />
      </MenuItem>

      <MenuItem onClick={onClickDownload}>
        <Localized name="download-midi" />
      </MenuItem>
    </>
  )
})
