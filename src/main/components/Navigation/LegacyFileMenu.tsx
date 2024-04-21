import { observer } from "mobx-react-lite"
import { ChangeEvent, FC } from "react"
import { Localized } from "../../../components/Localized"
import { MenuDivider, MenuItem } from "../../../components/Menu"
import { useSongFile } from "../../hooks/useSongFile"

const fileInputID = "OpenButtonInputFile"

export const FileInput: FC<
  React.PropsWithChildren<{
    onChange: (e: ChangeEvent<HTMLInputElement>) => void
    accept?: string
  }>
> = ({ onChange, children, accept }) => (
  <>
    <input
      accept={accept}
      style={{ display: "none" }}
      id={fileInputID}
      type="file"
      onChange={onChange}
    />
    <label htmlFor={fileInputID}>{children}</label>
  </>
)

export const LegacyFileMenu: FC<{ close: () => void }> = observer(
  ({ close }) => {
    const { createNewSong, openSongLegacy, downloadSong } = useSongFile()

    const onClickNew = async () => {
      close()
      await createNewSong()
    }

    const onClickOpen = async (e: ChangeEvent<HTMLInputElement>) => {
      close()
      await openSongLegacy(e)
    }

    const onClickSave = async () => {
      close()
      await downloadSong()
    }

    return (
      <>
        <MenuItem onClick={onClickNew}>
          <Localized default="New">new-song</Localized>
        </MenuItem>

        <MenuDivider />

        <FileInput onChange={onClickOpen} accept="audio/midi">
          <MenuItem>
            <Localized default="Open">open-song</Localized>
          </MenuItem>
        </FileInput>

        <MenuItem onClick={onClickSave}>
          <Localized default="Save">save-song</Localized>
        </MenuItem>
      </>
    )
  },
)
