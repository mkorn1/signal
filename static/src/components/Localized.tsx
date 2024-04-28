"use client"

import { LocalizationKey, LocalizedInternal } from "../l18n/useLocalization"

export default function Localized({ name }: { name: LocalizationKey }) {
  return <LocalizedInternal name={name} />
}
