import { observer } from "mobx-react-lite"
import { FC, PropsWithChildren } from "react"
import { useStores } from "../../hooks/useStores"
import { LocalizationContext } from "../../localize/useLocalization"

export const LocalizationProvider: FC<PropsWithChildren<{}>> = observer(
  ({ children }) => {
    const { settingStore } = useStores()
    return (
      <LocalizationContext.Provider value={{ language: settingStore.language }}>
        {children}
      </LocalizationContext.Provider>
    )
  },
)
