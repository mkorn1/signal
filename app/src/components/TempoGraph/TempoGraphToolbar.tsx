import styled from "@emotion/styled"
import { observer } from "mobx-react-lite"
import { FC } from "react"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { AutoScrollButton } from "../Toolbar/AutoScrollButton"
import { Toolbar } from "../Toolbar/Toolbar"
import { TempoGraphQuantizeSelector } from "./TempoGraphQuantizeSelector"
import { TempoGraphToolSelector } from "./TempoGraphToolSelector"

const Title = styled.span`
  font-weight: bold;
  margin-right: 2em;
  font-size: 1rem;
  margin-left: 0.5rem;
`

const FlexibleSpacer = styled.div`
  flex-grow: 1;
`

export const TempoGraphToolbar: FC = observer(() => {
  const { tempoEditorStore } = useStores()
  const { autoScroll } = tempoEditorStore

  return (
    <Toolbar>
      <Title>
        <Localized name="tempo" />
      </Title>

      <FlexibleSpacer />

      <TempoGraphToolSelector />

      <TempoGraphQuantizeSelector />

      <AutoScrollButton
        onClick={() => (tempoEditorStore.autoScroll = !autoScroll)}
        selected={autoScroll}
      />
    </Toolbar>
  )
})
