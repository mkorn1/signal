"use client"

export const LocaleDate = ({ date }: { date: Date }) => {
  return <>{date.toLocaleDateString()}</>
}
