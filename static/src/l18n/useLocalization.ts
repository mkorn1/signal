"use client"

import { createLocalization } from "use-l10n"
import localization from "./localization"

export const {
  LocalizationContext,
  useLocalization,
  Localized: LocalizedInternal,
} = createLocalization(localization, "en", [
  [/^zh-Hans/, "zh-Hans"],
  [/^zh-Hant/, "zh-Hant"],
  [/^zh$/, "zh-Hans"],
  [/^zh-TW$/, "zh-Hant"],
  [/^zh-HK$/, "zh-Hant"],
  [/^zh-MO$/, "zh-Hant"],
  [/^zh-CN$/, "zh-Hans"],
  [/^zh-SG$/, "zh-Hans"],
])

export type Language = keyof typeof localization
export type LocalizationKey = keyof (typeof localization)[Language]
