import { observer } from "mobx-react-lite"
import { ChangeEvent, FC } from "react"
import { Localized } from "../../../components/Localized"
import { MenuDivider, MenuItem } from "../../../components/Menu"
import { hasFSAccess } from "../../actions/file"
import { useCloudFile } from "../../hooks/useCloudFile"
import { useStores } from "../../hooks/useStores"
import { FileInput } from "./LegacyFileMenu"

export const CloudFileMenu: FC<{ close: () => void }> = observer(
  ({ close }) => {
    const rootStore = useStores()
    const { song } = rootStore
    const isCloudSaved = song.cloudSongId !== null
    const {
      createNewSong,
      openSong,
      saveSong,
      saveAsSong,
      renameSong,
      importSong,
      importSongLegacy,
      exportSong,
      publishSong,
    } = useCloudFile()

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

    const onClickRename = async () => {
      close()
      await renameSong()
    }

    const onClickImportLegacy = async (e: ChangeEvent<HTMLInputElement>) => {
      close()
      await importSongLegacy(e)
    }

    const onClickImport = async () => {
      close()
      await importSong()
    }

    const onClickExport = async () => {
      close()
      await exportSong()
    }

    const onClickPublish = async () => {
      close()
      await publishSong()
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

        <MenuItem onClick={onClickSave} disabled={rootStore.song.isSaved}>
          <Localized default="Save">save-song</Localized>
        </MenuItem>

        <MenuItem onClick={onClickSaveAs} disabled={!isCloudSaved}>
          <Localized default="Save As">save-as</Localized>
        </MenuItem>

        <MenuItem onClick={onClickRename} disabled={!isCloudSaved}>
          <Localized default="Rename">rename</Localized>
        </MenuItem>

        <MenuDivider />

        {!hasFSAccess && (
          <FileInput onChange={onClickImportLegacy}>
            <MenuItem>
              <Localized default="Import MIDI file">import-midi</Localized>
            </MenuItem>
          </FileInput>
        )}

        {hasFSAccess && (
          <MenuItem onClick={onClickImport}>
            <Localized default="Import MIDI file">import-midi</Localized>
          </MenuItem>
        )}

        <MenuItem onClick={onClickExport}>
          <Localized default="Export MIDI file">export-midi</Localized>
        </MenuItem>

        <MenuDivider />

        <MenuItem onClick={onClickPublish} disabled={!isCloudSaved}>
          <Localized default="Publish">publish</Localized>
        </MenuItem>
      </>
    )
  },
)
