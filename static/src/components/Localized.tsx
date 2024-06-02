"use client"

import { useEffect, useState } from "react"
import localization from "../l18n/localization"
import type { LocalizationKey } from "../l18n/useLocalization"

export default function Localized({ name }: { name: LocalizationKey }) {
  const [text, setText] = useState<React.ReactNode>(localization.en[name])

  useEffect(() => {
    import("../l18n/useLocalization").then(({ LocalizedInternal }) => {
      setText(<LocalizedInternal name={name} />)
    })
  }, [name])

  return <>{text}</>
}
