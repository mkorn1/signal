import styled from "@emotion/styled"
import { FC } from "react"
import { Localized } from "../../localize/useLocalization"
import { Button } from "../ui/Button"

export type SettingRoute = "general" | "midi" | "soundfont"
const routes: SettingRoute[] = ["general", "midi", "soundfont"]

const RouteItem = styled(Button)<{ selected: boolean }>`
  display: flex;
  font-size: 0.875rem;
  align-items: center;
  margin-bottom: 0.5rem;
  background: ${({ theme, selected }) =>
    selected ? theme.highlightColor : "inherit"};
`

const Container = styled.div`
  display: flex;
  flex-direction: column;
  min-width: 12em;
  margin-right: 2rem;
`

const RouteName: FC<{ route: SettingRoute }> = ({ route }) => {
  switch (route) {
    case "general":
      return <Localized name="general" />
    case "midi":
      return <Localized name="midi" />
    case "soundfont":
      return <Localized name="soundfont" />
  }
}

export const SettingNavigation: FC<{
  route: SettingRoute
  onChange: (route: SettingRoute) => void
}> = ({ route, onChange }) => {
  return (
    <Container>
      {routes.map((r) => (
        <RouteItem key={r} selected={route === r} onClick={() => onChange(r)}>
          <RouteName key={r} route={r} />
        </RouteItem>
      ))}
    </Container>
  )
}
