import styled from "@emotion/styled"
import { observer } from "mobx-react-lite"
import { ChangeEvent, FC } from "react"
import { Localized } from "../../../common/localize/useLocalization"
import { Alert } from "../../../components/Alert"
import { Button } from "../../../components/Button"
import { DialogContent, DialogTitle } from "../../../components/Dialog"
import { isRunningInElectron } from "../../helpers/platform"
import { useStores } from "../../hooks/useStores"
import { FileInput } from "../Navigation/LegacyFileMenu"
import { SoundFontList } from "./SoundFontList"
import { SoundFontScanPathList } from "./SoundFontScanPathList"

const OpenFileButton = styled(Button)`
  display: inline-flex;
  align-items: center;
`

const SectionTitle = styled.div`
  font-weight: bold;
  margin: 1rem 0;
`

export const SoundFontSettingsView: FC = observer(() => {
  const { soundFontStore } = useStores()

  // TODO: add open local file dialog and put it to SoundFontStore
  const onOpenSoundFont = async (e: ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    const file = e.currentTarget.files?.item(0)
    if (file) {
      const arrayBuffer = await file.arrayBuffer()
      soundFontStore.addSoundFont(
        { type: "local", data: arrayBuffer },
        { name: file.name },
      )
    }
  }

  return (
    <>
      <DialogTitle>
        <Localized name="soundfont" />
      </DialogTitle>
      <DialogContent>
        <SoundFontList />
        {!isRunningInElectron() && (
          <>
            <FileInput onChange={onOpenSoundFont} accept=".sf2">
              <OpenFileButton as="div">
                <Localized name="add" />
              </OpenFileButton>
            </FileInput>
            <Alert severity="info" style={{ marginTop: "1rem" }}>
              <Localized name="soundfont-save-notice" />
            </Alert>
          </>
        )}
        {isRunningInElectron() && (
          <>
            <SectionTitle>
              <Localized name="soundfont-location" />
            </SectionTitle>
            <SoundFontScanPathList />
          </>
        )}
      </DialogContent>
    </>
  )
})
