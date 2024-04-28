"use client"

import { FC, PropsWithChildren } from "react"
import { LocalizationContext } from "../l18n/useLocalization"

export const Provider: FC<PropsWithChildren> = ({ children }) => {
  return (
    <>
      <LocalizationContext.Provider value={{ language: null }}>
        {children}
      </LocalizationContext.Provider>
    </>
  )
}
