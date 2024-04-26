import { Context, createContext, useContext } from "react"
import { LocalizationTable } from "./LocalizationTable"
import { getBrowserLanguage } from "./getBrowserLanguage"

type LocalizationContextType<Languages extends string> = {
  language: Languages | null
}

const createLocalizationContext = <Languages extends string>() =>
  createContext<LocalizationContextType<Languages>>({
    language: null,
  })

const createUseLocalization = <Languages extends string, Keys extends string>(
  LocalizationContext: Context<LocalizationContextType<Languages>>,
  localizationTable: LocalizationTable<Languages, Keys>,
) => {
  return () => {
    const context = useContext(LocalizationContext)

    if (!context) {
      throw new Error("useLocalized must be used within a LocalizationContext")
    }

    return (key: Keys, defaultValue: string) => {
      const language = context.language ?? getBrowserLanguage()
      const supportedLanguage = localizationTable.getLanguage(language)
      if (supportedLanguage === null) {
        return defaultValue
      }
      return localizationTable.getString(supportedLanguage, key) ?? defaultValue
    }
  }
}

export const createLocalization = <
  Languages extends string,
  Keys extends string,
>(
  localizationTable: LocalizationTable<Languages, Keys>,
) => {
  const LocalizationContext = createLocalizationContext<Languages>()
  const useLocalization = createUseLocalization(
    LocalizationContext,
    localizationTable,
  )
  return { LocalizationContext, useLocalization }
}
