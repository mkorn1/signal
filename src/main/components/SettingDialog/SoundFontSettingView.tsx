import styled from "@emotion/styled"
import { observer } from "mobx-react-lite"
import { ChangeEvent, FC } from "react"
import { Alert } from "../../../components/Alert"
import { Button } from "../../../components/Button"
import { DialogContent, DialogTitle } from "../../../components/Dialog"
import { Localized } from "../../../components/Localized"
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
        <Localized default="SoundFont">soundfont</Localized>
      </DialogTitle>
      <DialogContent>
        <SoundFontList />
        {!isRunningInElectron() && (
          <>
            <FileInput onChange={onOpenSoundFont} accept=".sf2">
              <OpenFileButton as="div">
                <Localized default="Add">add</Localized>
              </OpenFileButton>
            </FileInput>
            <Alert severity="info" style={{ marginTop: "1rem" }}>
              <Localized default="SoundFont will be saved in the browser.">
                soundfont-save-notice
              </Localized>
            </Alert>
          </>
        )}
        {isRunningInElectron() && (
          <>
            <SectionTitle>
              <Localized default="Location">soundfont-location</Localized>
            </SectionTitle>
            <SoundFontScanPathList />
          </>
        )}
      </DialogContent>
    </>
  )
})
