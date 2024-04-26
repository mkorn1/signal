"use client"

import { observer } from "mobx-react-lite"
import { FC } from "react"
import { LocalizationKey, useLocalization } from "./useLocalization"

export interface LocalizedProps {
  children: string
  localizationKey: LocalizationKey
}

export const Localized: FC<LocalizedProps> = observer(
  ({ children, localizationKey }) => {
    const localized = useLocalization()
    return <>{localized(localizationKey, children)}</>
  },
)
