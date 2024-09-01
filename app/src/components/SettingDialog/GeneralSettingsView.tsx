import { observer } from "mobx-react-lite"
import { FC } from "react"
import { useStores } from "../../hooks/useStores"
import {
  Language,
  Localized,
  useCurrentLanguage,
} from "../../localize/useLocalization"
import { DialogContent, DialogTitle } from "../Dialog/Dialog"
import { Label } from "../ui/Label"
import { Select } from "../ui/Select"

interface LanguageItem {
  label: string
  language: Language
}

const LanguageSelect: FC = observer(() => {
  const { settingStore } = useStores()
  const language = useCurrentLanguage()
  const items: LanguageItem[] = [
    { label: "English", language: "en" },
    { label: "Japanese", language: "ja" },
    { label: "Chinese (Simplified)", language: "zh-Hans" },
    { label: "Chinese (Traditional)", language: "zh-Hant" },
  ]
  return (
    <Label>
      <Localized name="language" />
      <Select
        value={settingStore.language ?? language}
        onChange={(e) => (settingStore.language = e.target.value as Language)}
        style={{ marginTop: "0.5rem" }}
      >
        {items.map((item) => (
          <option key={item.language} value={item.language}>
            {item.label}
          </option>
        ))}
      </Select>
    </Label>
  )
})

export const GeneralSettingsView: FC = observer(() => {
  return (
    <>
      <DialogTitle>
        <Localized name="general" />
      </DialogTitle>
      <DialogContent>
        <LanguageSelect />
      </DialogContent>
    </>
  )
})
