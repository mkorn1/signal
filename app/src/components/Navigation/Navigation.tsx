import styled from "@emotion/styled"
import Headphones from "mdi-react/HeadphonesIcon"
import Settings from "mdi-react/SettingsIcon"
import { CSSProperties, FC, MouseEvent, useCallback } from "react"
import { getPlatform, isRunningInElectron } from "../../helpers/platform"
import { useEditorMode } from "../../hooks/useEditorMode"
import { useHQRender } from "../../hooks/useHQRender"
import { useRootView } from "../../hooks/useRootView"
import { useRouter } from "../../hooks/useRouter"
import ArrangeIcon from "../../images/icons/arrange.svg"
import PianoIcon from "../../images/icons/piano.svg"
import { envString } from "../../localize/envString"
import { Localized } from "../../localize/useLocalization"
import { CircularProgress } from "../ui/CircularProgress"
import { Tooltip } from "../ui/Tooltip"
import { EditMenuButton } from "./EditMenuButton"
import { FileMenuButton } from "./FileMenuButton"

const Container = styled.div`
  display: flex;
  flex-direction: row;
  background: var(--color-background-dark);
  height: 3.25rem;
  flex-shrink: 0;
  -webkit-app-region: drag;
  border-bottom: 1px solid var(--color-divider);
  padding: ${() => {
    if (isRunningInElectron()) {
      const platform = getPlatform()
      switch (platform) {
        case "Windows":
          return "0 0 0 0"
        case "macOS":
          return "0 0 0 76px"
      }
    }
  }};
`

export const Tab = styled.div<{ isActive?: boolean }>`
  display: flex;
  flex-direction: row;
  align-items: center;
  align-self: center;
  padding: 0.375rem 0.75rem;
  margin: 0 0.25rem;
  font-size: 0.6875rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  border-radius: 0.375rem;
  border: 1px solid ${({ isActive }) =>
    isActive ? "var(--color-theme)" : "rgba(255, 255, 255, 0.1)"};
  color: ${({ isActive }) =>
    isActive ? "var(--color-on-surface)" : "var(--color-text-secondary)"};
  background: ${({ isActive }) =>
    isActive ? "var(--color-theme)" : "rgba(255, 255, 255, 0.04)"};
  cursor: pointer;
  -webkit-app-region: none;
  transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);

  &.active {
    color: var(--color-on-surface);
    background: var(--color-theme);
    border-color: var(--color-theme);
  }

  &:hover:not(.active) {
    color: var(--color-text);
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(255, 255, 255, 0.15);
  }

  &:active {
    transform: scale(0.98);
  }

  a {
    color: inherit;
    text-decoration: none;
  }

  svg {
    width: 0.875rem;
    height: 0.875rem;
    fill: currentColor;
  }
`

export const TabTitle = styled.span`
  margin-left: 0.375rem;

  @media (max-width: 700px) {
    display: none;
  }
`

const FlexibleSpacer = styled.div`
  flex-grow: 1;
`

export const NavButton = styled.button<{ isActive?: boolean; isLoading?: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  align-self: center;
  gap: 0.375rem;
  padding: 0.375rem 0.75rem;
  border: 1px solid ${({ isActive }) => isActive ? "var(--color-theme)" : "rgba(255, 255, 255, 0.1)"};
  border-radius: 0.375rem;
  background: ${({ isActive }) => isActive ? "var(--color-theme)" : "rgba(255, 255, 255, 0.04)"};
  color: ${({ isActive, isLoading }) => 
    isActive ? "var(--color-on-surface)" : 
    isLoading ? "var(--color-theme)" : 
    "var(--color-text-secondary)"};
  font-size: 0.6875rem;
  font-weight: 600;
  font-family: inherit;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  cursor: ${({ isLoading }) => isLoading ? "wait" : "pointer"};
  transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
  -webkit-app-region: none;
  opacity: ${({ isLoading }) => isLoading ? 0.8 : 1};

  &:hover:not(:disabled) {
    background: ${({ isActive }) => isActive ? "var(--color-theme)" : "rgba(255, 255, 255, 0.08)"};
    border-color: ${({ isActive }) => isActive ? "var(--color-theme)" : "rgba(255, 255, 255, 0.15)"};
    color: ${({ isActive }) => isActive ? "var(--color-on-surface)" : "var(--color-text)"};
  }

  &:active:not(:disabled) {
    transform: scale(0.98);
  }

  svg {
    width: 0.875rem;
    height: 0.875rem;
    fill: currentColor;
  }
  
  @media (max-width: 700px) {
    span {
      display: none;
    }
    padding: 0.375rem 0.5rem;
  }
`

const NavButtonGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0 0.5rem;
  -webkit-app-region: none;
`

const ModeToggle = styled.div`
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 0.375rem;
  padding: 0.125rem;
  -webkit-app-region: none;
`

const ModeToggleOption = styled.button<{ isActive: boolean }>`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
  padding: 0.25rem 0.625rem;
  border: none;
  border-radius: 0.25rem;
  background: ${({ isActive }) => isActive ? "var(--color-theme)" : "transparent"};
  color: ${({ isActive }) => isActive ? "var(--color-on-surface)" : "var(--color-text-secondary)"};
  font-size: 0.6875rem;
  font-weight: 600;
  font-family: inherit;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  cursor: pointer;
  transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
  -webkit-app-region: none;

  &:hover {
    color: ${({ isActive }) => isActive ? "var(--color-on-surface)" : "var(--color-text)"};
    background: ${({ isActive }) => isActive ? "var(--color-theme)" : "rgba(255, 255, 255, 0.06)"};
  }

  svg {
    width: 0.75rem;
    height: 0.75rem;
    fill: currentColor;
  }
  
  @media (max-width: 700px) {
    span {
      display: none;
    }
    padding: 0.25rem 0.375rem;
  }
`

export const IconStyle: CSSProperties = {
  width: "1.3rem",
  height: "1.3rem",
  fill: "currentColor",
}

export const Navigation: FC = () => {
  const { openSettingDialog, setOpenSettingDialog } = useRootView()
  const { path, setPath } = useRouter()
  const { toggle: toggleEditorMode, isAdvanced } = useEditorMode()
  const { render: renderHQ, isLoading: isRenderLoading } = useHQRender()

  const onClickPianoRollTab = useCallback(
    (e: MouseEvent) => {
      e.preventDefault()
      setPath("/track")
    },
    [setPath],
  )

  const onClickArrangeTab = useCallback(
    (e: MouseEvent) => {
      e.preventDefault()
      setPath("/arrange")
    },
    [setPath],
  )

  const onClickSettings = useCallback(
    (e: MouseEvent) => {
      e.preventDefault()
      setOpenSettingDialog(true)
    },
    [setOpenSettingDialog],
  )

  const onClickRenderHQ = useCallback(
    (e: MouseEvent) => {
      e.preventDefault()
      if (!isRenderLoading) {
        renderHQ()
      }
    },
    [renderHQ, isRenderLoading],
  )

  return (
    <Container>
      {!isRunningInElectron() && (
        <NavButtonGroup style={{ marginLeft: "0.5rem" }}>
          <FileMenuButton />
          <EditMenuButton />
        </NavButtonGroup>
      )}

      <Tooltip
        title={`Edit notes for a single track [${envString.cmdOrCtrl}+1]`}
        delayDuration={500}
      >
        <Tab
          className={path === "/track" ? "active" : undefined}
          onMouseDown={onClickPianoRollTab}
        >
          <PianoIcon style={IconStyle} viewBox="0 0 128 128" />
          <TabTitle>
            <Localized name="piano-roll" />
          </TabTitle>
        </Tab>
      </Tooltip>

      <Tooltip
        title={`View and arrange all tracks [${envString.cmdOrCtrl}+2]`}
        delayDuration={500}
      >
        <Tab
          className={path === "/arrange" ? "active" : undefined}
          onMouseDown={onClickArrangeTab}
        >
          <ArrangeIcon style={IconStyle} viewBox="0 0 128 128" />
          <TabTitle>
            <Localized name="arrange" />
          </TabTitle>
        </Tab>
      </Tooltip>

      <FlexibleSpacer />

      <NavButtonGroup>
        <Tooltip title="Render high-quality audio with FluidSynth" delayDuration={500}>
          <NavButton isLoading={isRenderLoading} onClick={onClickRenderHQ}>
            {isRenderLoading ? (
              <CircularProgress size="0.875rem" />
            ) : (
              <Headphones />
            )}
            <span>Render</span>
          </NavButton>
        </Tooltip>

        <Tooltip title="Toggle between simple and advanced editing modes" delayDuration={500}>
          <ModeToggle>
            <ModeToggleOption
              isActive={!isAdvanced}
              onClick={() => isAdvanced && toggleEditorMode()}
            >
              <span>Simple</span>
            </ModeToggleOption>
            <ModeToggleOption
              isActive={isAdvanced}
              onClick={() => !isAdvanced && toggleEditorMode()}
            >
              <span>Advanced</span>
            </ModeToggleOption>
          </ModeToggle>
        </Tooltip>

        {!isRunningInElectron() && (
          <Tooltip title="App preferences and audio settings" delayDuration={500}>
            <NavButton isActive={openSettingDialog} onClick={onClickSettings}>
              <Settings />
              <span>Settings</span>
            </NavButton>
          </Tooltip>
        )}
      </NavButtonGroup>
    </Container>
  )
}
