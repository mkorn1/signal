import styled from "@emotion/styled"
import { FC } from "react"
import { AutoScrollButton } from "../Toolbar/AutoScrollButton"
import { Toolbar } from "../Toolbar/Toolbar"

const FlexibleSpacer = styled.div`
  flex-grow: 1;
`

export const ArrangeToolbar: FC = () => {
  return (
    <Toolbar>
      <FlexibleSpacer />
      <AutoScrollButton />
    </Toolbar>
  )
}
