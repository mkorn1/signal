import { observer } from "mobx-react-lite"
import { FC, PropsWithChildren } from "react"
import { LocalizationContext } from "../../../common/localize/useLocalization"
import { useStores } from "../../hooks/useStores"

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
