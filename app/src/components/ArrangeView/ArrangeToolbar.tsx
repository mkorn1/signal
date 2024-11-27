import styled from "@emotion/styled"
import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import { Localized } from "../../localize/useLocalization"
import { AutoScrollButton } from "../Toolbar/AutoScrollButton"
import { Toolbar } from "../Toolbar/Toolbar"
import { ArrangeQuantizeSelector } from "./ArrangeQuantizeSelector"

const Title = styled.div`
  font-weight: bold;
  font-size: 1rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 14rem;
  min-width: 3em;
  margin-left: 0.5rem;
`

const FlexibleSpacer = styled.div`
  flex-grow: 1;
`

export const ArrangeToolbar: FC = observer(() => {
  const { arrangeViewStore } = useStores()
  const { autoScroll } = arrangeViewStore

  const onClickAutoScroll = useCallback(
    () => (arrangeViewStore.autoScroll = !arrangeViewStore.autoScroll),
    [arrangeViewStore],
  )

  return (
    <Toolbar>
      <Title>
        <Localized name="arrangement-view" />
      </Title>

      <FlexibleSpacer />

      <ArrangeQuantizeSelector />

      <AutoScrollButton onClick={onClickAutoScroll} selected={autoScroll} />
    </Toolbar>
  )
})
