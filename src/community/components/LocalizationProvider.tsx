import { observer } from "mobx-react-lite"
import { FC, PropsWithChildren } from "react"
import { LocalizationContext } from "../../common/localize/useLocalization"

export const LocalizationProvider: FC<PropsWithChildren<{}>> = observer(
  ({ children }) => {
    return (
      <LocalizationContext.Provider
        value={{
          language: null, // Use the browser's language
        }}
      >
        {children}
      </LocalizationContext.Provider>
    )
  },
)
