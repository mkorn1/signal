import styled from "@emotion/styled"
import { FC } from "react"
import { Localized } from "../../localize/useLocalization"
import { DialogContent, DialogTitle } from "../Dialog/Dialog"

const ShortcutList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`

const ShortcutItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--color-divider);

  &:last-child {
    border-bottom: none;
  }
`

const ShortcutLabel = styled.span`
  color: var(--color-text);
`

const ShortcutKeys = styled.span`
  font-family: var(--font-mono);
  font-size: 0.85rem;
  color: var(--color-text-secondary);
  background: var(--color-background);
  padding: 0.25rem 0.5rem;
  border-radius: 0.25rem;
  border: 1px solid var(--color-divider);
`

const SectionTitle = styled.div`
  font-weight: bold;
  margin: 1rem 0 0.5rem;
  color: var(--color-text);

  &:first-of-type {
    margin-top: 0;
  }
`

const isMac = navigator.platform.toUpperCase().indexOf("MAC") >= 0
const cmdKey = isMac ? "⌘" : "Ctrl"

const shortcuts = {
  playback: [
    { label: "Play / Pause", keys: "Space" },
    { label: "Stop", keys: "Enter" },
    { label: "Rewind one bar", keys: "A" },
    { label: "Fast forward one bar", keys: "D" },
    { label: "Toggle recording", keys: "R" },
  ],
  file: [
    { label: "New", keys: `${cmdKey}+N` },
    { label: "Open", keys: `${cmdKey}+O` },
    { label: "Save", keys: `${cmdKey}+S` },
    { label: "Save As", keys: `⇧+${cmdKey}+S` },
  ],
  edit: [
    { label: "Undo", keys: `${cmdKey}+Z` },
    { label: "Redo", keys: `⇧+${cmdKey}+Z` },
  ],
  navigation: [
    { label: "Piano Roll", keys: `${cmdKey}+1` },
    { label: "Arrangement View", keys: `${cmdKey}+2` },
  ],
}

export const ShortcutsSettingsView: FC = () => {
  return (
    <>
      <DialogTitle>
        <Localized name="keyboard-shortcut" />
      </DialogTitle>
      <DialogContent>
        <ShortcutList>
          <SectionTitle>Playback</SectionTitle>
          {shortcuts.playback.map((s) => (
            <ShortcutItem key={s.label}>
              <ShortcutLabel>{s.label}</ShortcutLabel>
              <ShortcutKeys>{s.keys}</ShortcutKeys>
            </ShortcutItem>
          ))}

          <SectionTitle>File</SectionTitle>
          {shortcuts.file.map((s) => (
            <ShortcutItem key={s.label}>
              <ShortcutLabel>{s.label}</ShortcutLabel>
              <ShortcutKeys>{s.keys}</ShortcutKeys>
            </ShortcutItem>
          ))}

          <SectionTitle>Edit</SectionTitle>
          {shortcuts.edit.map((s) => (
            <ShortcutItem key={s.label}>
              <ShortcutLabel>{s.label}</ShortcutLabel>
              <ShortcutKeys>{s.keys}</ShortcutKeys>
            </ShortcutItem>
          ))}

          <SectionTitle>Navigation</SectionTitle>
          {shortcuts.navigation.map((s) => (
            <ShortcutItem key={s.label}>
              <ShortcutLabel>{s.label}</ShortcutLabel>
              <ShortcutKeys>{s.keys}</ShortcutKeys>
            </ShortcutItem>
          ))}
        </ShortcutList>
      </DialogContent>
    </>
  )
}
