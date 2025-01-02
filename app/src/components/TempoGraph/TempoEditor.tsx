import styled from "@emotion/styled"
import { FC } from "react"
import { TempoEditorKeyboardShortcut } from "../KeyboardShortcut/TempoEditorKeyboardShortcut"
import { TempoGraphToolbar } from "../TempoGraphToolbar/TempoGraphToolbar"
import { TempoGraph } from "./TempoGraph"

const Container = styled.div`
  display: flex;
  flex-direction: column;
  flex-grow: 1;
  overflow: hidden;
`

export const TempoEditor: FC = () => {
  return (
    <Container>
      <TempoEditorKeyboardShortcut />
      <TempoGraphToolbar />
      <TempoGraph />
    </Container>
  )
}
