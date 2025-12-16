# UX Update: Poolsuite.net-Style Draggable Window Modals

## Overview

Update the app's modal system to use poolsuite.net-inspired draggable windows with:
- Draggable title bars for repositioning windows
- macOS-style window chrome (traffic light buttons)
- Z-index stacking (click to bring to front)
- Smooth spring-based animations
- AI Chat as a persistent, non-closable overlay modal

**Affected Components:**
- `InstrumentBrowser` - Instrument selection modal
- `SettingDialog` - Settings modal
- `AIChat` - Transform from sidebar to persistent modal overlay

**No changes to existing styles/themes** - only interaction patterns and modal presentation.

---

## Z-Index Hierarchy (Reference)

Establish clear z-index layers to prevent conflicts:

| Layer | Z-Index Range | Usage |
|-------|---------------|-------|
| Base content | 0-99 | Main app content |
| Floating panels | 100-199 | Transport panel, toolbars |
| Draggable windows | 1000-1099 | InstrumentBrowser, SettingDialog, AIChatModal |
| Dropdown/Select menus | 1100-1199 | Radix Select, custom dropdowns |
| Tooltips | 1200-1299 | All tooltips |
| Context menus | 1300-1399 | Right-click menus |
| Critical dialogs | 1400-1499 | ActionDialog, PromptDialog (blocking) |
| Toast notifications | 1500+ | Error/success toasts |

---

## Phase 0: Fix Existing Issues (Pre-requisite)

### 0.1 Fix useInstrumentBrowser Hook Violation

**File:** `app/src/hooks/useInstrumentBrowser.ts`

**Problem:** The hook uses getters that call hooks inside them, violating Rules of Hooks.

**Before:**
```ts
return {
  get isOpen() {
    return useAtomValue(isOpenAtom)  // ❌ Hook inside getter
  },
  get categoryFirstProgramEvents() {
    return useMemo(() => { ... }, [...])  // ❌ Hook inside getter
  },
  get categoryInstruments() {
    return useMemo(() => { ... }, [...])  // ❌ Hook inside getter
  },
}
```

**After:**
```ts
export function useInstrumentBrowser() {
  // Move all hook calls to top level
  const isOpen = useAtomValue(isOpenAtom)
  const [setting, setSetting] = useAtom(settingAtom)
  // ... other existing hooks ...

  const categoryFirstProgramEvents = useMemo(() => {
    if (setting.isRhythmTrack) {
      return [0]
    }
    return range(0, 127, 8)
  }, [setting.isRhythmTrack])

  const categoryInstruments = useMemo(() => {
    if (setting.isRhythmTrack) {
      return [0, 8, 16, 24, 25, 32, 40, 48, 56]
    }
    const offset = selectedCategoryIndex * 8
    return range(offset, offset + 8)
  }, [selectedCategoryIndex, setting.isRhythmTrack])

  return {
    isOpen,  // Now a plain value, not a getter
    categoryFirstProgramEvents,  // Now a plain value
    categoryInstruments,  // Now a plain value
    // ... rest unchanged ...
  }
}
```

### 0.2 Audit AIChat Toggle Call Sites

**Problem:** The plan removes all toggle/hide functionality. Need to find and remove all call sites.

**Search for:** All uses of `setOpen`, `setAIChatOpen`, `toggle()`, `isOpen` from useAIChat

**Files to audit and update:**
- `app/src/components/AIChat/AIChat.tsx` - Remove close button
- `app/src/hooks/useGlobalKeyboardShortcut.tsx` - Remove any toggle shortcut
- `app/src/components/Navigation/Navigation.tsx` - Remove toggle button
- Any toolbar components using useAIChat

**Action:** 
- Remove close button from AIChat component
- Remove toggle keyboard shortcut
- Remove any toggle buttons in Navigation/toolbars
- Remove `isOpen`, `setOpen`, `toggle`, `show`, `hide` from useAIChat hook

---

## Phase 1: Create Core Infrastructure

### 1.1 Create Window Position Store

**File:** `app/src/stores/windowPositions.ts`

Persist window positions across open/close cycles and page reloads.

```ts
import { atom } from "jotai"
import { atomWithStorage } from "jotai/utils"

export interface WindowPosition {
  x: number
  y: number
}

export interface WindowPositions {
  [windowId: string]: WindowPosition
}

// Persisted to localStorage
export const windowPositionsAtom = atomWithStorage<WindowPositions>(
  "signal-window-positions",
  {}
)

// Runtime window stack for z-index (not persisted)
export const windowStackAtom = atom<string[]>([])
```

### 1.2 Create Window Manager Hook

**File:** `app/src/hooks/useWindowManager.ts`

Manages z-index stacking and position persistence.

```ts
import { useAtom } from "jotai"
import { useCallback, useMemo, useEffect } from "react"
import { windowStackAtom, windowPositionsAtom, WindowPosition } from "../stores/windowPositions"

const BASE_Z_INDEX = 1000

export interface UseWindowManagerOptions {
  windowId: string
  defaultPosition: WindowPosition
}

export const useWindowManager = ({ windowId, defaultPosition }: UseWindowManagerOptions) => {
  const [stack, setStack] = useAtom(windowStackAtom)
  const [positions, setPositions] = useAtom(windowPositionsAtom)

  // Get position from store or use default
  const position = useMemo(() => {
    return positions[windowId] ?? defaultPosition
  }, [positions, windowId, defaultPosition])

  // Save position to store
  const setPosition = useCallback((newPosition: WindowPosition) => {
    setPositions((prev) => ({
      ...prev,
      [windowId]: newPosition,
    }))
  }, [windowId, setPositions])

  // Bring window to front of stack
  const bringToFront = useCallback(() => {
    setStack((prev) => {
      if (prev[prev.length - 1] === windowId) {
        return prev // Already at front
      }
      const filtered = prev.filter((id) => id !== windowId)
      return [...filtered, windowId]
    })
  }, [windowId, setStack])

  // Remove from stack when window closes
  const removeFromStack = useCallback(() => {
    setStack((prev) => prev.filter((id) => id !== windowId))
  }, [windowId, setStack])

  // Calculate z-index based on stack position
  const zIndex = useMemo(() => {
    const index = stack.indexOf(windowId)
    return index >= 0 ? BASE_Z_INDEX + index + 1 : BASE_Z_INDEX
  }, [stack, windowId])

  const isFrontmost = stack[stack.length - 1] === windowId

  return {
    position,
    setPosition,
    zIndex,
    bringToFront,
    removeFromStack,
    isFrontmost,
  }
}
```

### 1.3 Create Drag Hook with Proper Cleanup

**File:** `app/src/hooks/useWindowDrag.ts`

Safe drag handling with cleanup and touch support.

```ts
import { useCallback, useRef, useEffect } from "react"
import { WindowPosition } from "../stores/windowPositions"

interface UseWindowDragOptions {
  position: WindowPosition
  setPosition: (pos: WindowPosition) => void
  minWidth: number
  onDragStart?: () => void
  onDragEnd?: () => void
}

export const useWindowDrag = ({
  position,
  setPosition,
  minWidth,
  onDragStart,
  onDragEnd,
}: UseWindowDragOptions) => {
  const isDraggingRef = useRef(false)
  const startPosRef = useRef({ x: 0, y: 0 })
  const startWindowPosRef = useRef({ x: 0, y: 0 })
  
  // Store refs to callbacks for cleanup
  const handlersRef = useRef<{
    move: (e: MouseEvent | TouchEvent) => void
    end: (e: MouseEvent | TouchEvent) => void
  } | null>(null)

  // Constrain position to viewport
  const constrainPosition = useCallback((x: number, y: number): WindowPosition => {
    const padding = 50
    return {
      x: Math.max(-minWidth + padding, Math.min(window.innerWidth - padding, x)),
      y: Math.max(0, Math.min(window.innerHeight - padding, y)),
    }
  }, [minWidth])

  // Handle window resize - keep windows in bounds
  useEffect(() => {
    const handleResize = () => {
      const constrained = constrainPosition(position.x, position.y)
      if (constrained.x !== position.x || constrained.y !== position.y) {
        setPosition(constrained)
      }
    }

    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [position, setPosition, constrainPosition])

  // Cleanup function for drag events
  const cleanup = useCallback(() => {
    if (handlersRef.current) {
      document.removeEventListener("mousemove", handlersRef.current.move)
      document.removeEventListener("mouseup", handlersRef.current.end)
      document.removeEventListener("touchmove", handlersRef.current.move)
      document.removeEventListener("touchend", handlersRef.current.end)
      handlersRef.current = null
    }
    isDraggingRef.current = false
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return cleanup
  }, [cleanup])

  const getClientPos = (e: MouseEvent | TouchEvent): { x: number; y: number } => {
    if ("touches" in e) {
      const touch = e.touches[0] || e.changedTouches[0]
      return { x: touch.clientX, y: touch.clientY }
    }
    return { x: e.clientX, y: e.clientY }
  }

  const handleDragStart = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    // Only handle left click for mouse
    if ("button" in e && e.button !== 0) return
    // Don't drag if clicking a button
    if ((e.target as HTMLElement).closest("button")) return

    e.preventDefault()
    isDraggingRef.current = true
    onDragStart?.()

    const clientPos = getClientPos(e.nativeEvent as MouseEvent | TouchEvent)
    startPosRef.current = clientPos
    startWindowPosRef.current = { ...position }

    const handleMove = (moveEvent: MouseEvent | TouchEvent) => {
      if (!isDraggingRef.current) return
      
      const currentPos = getClientPos(moveEvent)
      const delta = {
        x: currentPos.x - startPosRef.current.x,
        y: currentPos.y - startPosRef.current.y,
      }

      const newPos = constrainPosition(
        startWindowPosRef.current.x + delta.x,
        startWindowPosRef.current.y + delta.y
      )
      setPosition(newPos)
    }

    const handleEnd = () => {
      cleanup()
      onDragEnd?.()
    }

    // Store handlers for cleanup
    handlersRef.current = { move: handleMove, end: handleEnd }

    document.addEventListener("mousemove", handleMove)
    document.addEventListener("mouseup", handleEnd)
    document.addEventListener("touchmove", handleMove, { passive: false })
    document.addEventListener("touchend", handleEnd)
  }, [position, setPosition, constrainPosition, cleanup, onDragStart, onDragEnd])

  return {
    handleDragStart,
    isDragging: isDraggingRef.current,
  }
}
```

### 1.4 Create DraggableWindow Component

**File:** `app/src/components/ui/DraggableWindow.tsx`

Core draggable window with focus management and accessibility.

```tsx
import styled from "@emotion/styled"
import { FC, ReactNode, useCallback, useEffect, useRef } from "react"
import { FocusScope } from "@radix-ui/react-focus-scope"
import { useWindowManager } from "../../hooks/useWindowManager"
import { useWindowDrag } from "../../hooks/useWindowDrag"
import { WindowPosition } from "../../stores/windowPositions"

interface DraggableWindowProps {
  windowId: string
  title: string
  children: ReactNode
  isOpen: boolean
  onClose?: () => void
  canClose?: boolean
  defaultPosition?: WindowPosition
  minWidth?: number
  minHeight?: number
  maxWidth?: number
  maxHeight?: number
}

const WindowContainer = styled.div<{ x: number; y: number; zIndex: number; isActive: boolean }>`
  position: fixed;
  left: ${({ x }) => x}px;
  top: ${({ y }) => y}px;
  z-index: ${({ zIndex }) => zIndex};
  display: flex;
  flex-direction: column;
  background: ${({ theme }) => theme.backgroundColor};
  border: 1px solid ${({ isActive }) => 
    isActive ? "rgba(255, 255, 255, 0.15)" : "rgba(255, 255, 255, 0.08)"};
  border-radius: 10px;
  box-shadow: ${({ isActive }) => isActive 
    ? "0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 0 1px rgba(255, 255, 255, 0.08) inset"
    : "0 15px 35px -10px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05) inset"};
  overflow: hidden;
  transition: box-shadow 150ms ease, border-color 150ms ease;
  animation: windowAppear 200ms cubic-bezier(0.34, 1.56, 0.64, 1);

  @keyframes windowAppear {
    from {
      opacity: 0;
      transform: scale(0.95) translateY(10px);
    }
    to {
      opacity: 1;
      transform: scale(1) translateY(0);
    }
  }

  &:focus {
    outline: none;
  }
`

const TitleBar = styled.div<{ isDragging?: boolean }>`
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: linear-gradient(180deg, 
    rgba(255, 255, 255, 0.06) 0%, 
    rgba(255, 255, 255, 0.02) 100%
  );
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  cursor: ${({ isDragging }) => isDragging ? "grabbing" : "grab"};
  user-select: none;
  touch-action: none;
`

const WindowControls = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`

const WindowButton = styled.button<{ variant: "close" | "minimize" | "maximize"; disabled?: boolean }>`
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: none;
  padding: 0;
  cursor: pointer;
  transition: all 120ms ease;
  
  background: ${({ variant, disabled, theme }) => {
    if (disabled) return theme.secondaryBackgroundColor;
    switch (variant) {
      case "close": return "#ff5f57";
      case "minimize": return "#febc2e";
      case "maximize": return "#28c840";
    }
  }};
  
  &:hover:not(:disabled) {
    filter: brightness(1.15);
    transform: scale(1.15);
  }

  &:active:not(:disabled) {
    transform: scale(0.95);
  }

  &:disabled {
    cursor: not-allowed;
    opacity: 0.3;
  }

  &:focus {
    outline: none;
    box-shadow: 0 0 0 2px ${({ theme }) => theme.themeColor};
  }
`

const TitleText = styled.span`
  flex: 1;
  font-size: 13px;
  font-weight: 500;
  color: ${({ theme }) => theme.textColor};
  letter-spacing: -0.01em;
  text-align: center;
  opacity: 0.9;
`

const Content = styled.div`
  flex: 1;
  overflow: auto;
  display: flex;
  flex-direction: column;
`

// Calculate center position
const getCenterPosition = (minWidth: number, minHeight: number): WindowPosition => ({
  x: Math.max(50, (window.innerWidth - minWidth) / 2),
  y: Math.max(50, (window.innerHeight - minHeight) / 3),
})

export const DraggableWindow: FC<DraggableWindowProps> = ({
  windowId,
  title,
  children,
  isOpen,
  onClose,
  canClose = true,
  defaultPosition,
  minWidth = 300,
  minHeight = 200,
  maxWidth,
  maxHeight,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  
  const {
    position,
    setPosition,
    zIndex,
    bringToFront,
    removeFromStack,
    isFrontmost,
  } = useWindowManager({
    windowId,
    defaultPosition: defaultPosition ?? getCenterPosition(minWidth, minHeight),
  })

  const { handleDragStart } = useWindowDrag({
    position,
    setPosition,
    minWidth,
    onDragStart: bringToFront,
  })

  // Handle open/close stack management
  useEffect(() => {
    if (isOpen) {
      bringToFront()
    } else {
      removeFromStack()
    }
  }, [isOpen, bringToFront, removeFromStack])

  // Handle Escape key to close (when closable)
  useEffect(() => {
    if (!isOpen || !canClose) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isFrontmost) {
        e.preventDefault()
        onClose?.()
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [isOpen, canClose, isFrontmost, onClose])

  // Focus container when opened
  useEffect(() => {
    if (isOpen && containerRef.current) {
      containerRef.current.focus()
    }
  }, [isOpen])

  const handleContainerMouseDown = useCallback(() => {
    bringToFront()
  }, [bringToFront])

  const handleClose = useCallback(() => {
    if (canClose) {
      onClose?.()
    }
  }, [canClose, onClose])

  if (!isOpen) return null

  return (
    <FocusScope trapped={isFrontmost} loop>
      <WindowContainer
        ref={containerRef}
        x={position.x}
        y={position.y}
        zIndex={zIndex}
        isActive={isFrontmost}
        style={{
          minWidth,
          minHeight,
          maxWidth: maxWidth ?? "90vw",
          maxHeight: maxHeight ?? "85vh",
        }}
        onMouseDown={handleContainerMouseDown}
        onTouchStart={handleContainerMouseDown}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${windowId}-title`}
      >
        <TitleBar
          onMouseDown={handleDragStart}
          onTouchStart={handleDragStart}
        >
          <WindowControls>
            <WindowButton
              variant="close"
              onClick={handleClose}
              disabled={!canClose}
              aria-label={canClose ? "Close window" : "Cannot close"}
              tabIndex={canClose ? 0 : -1}
            />
            <WindowButton
              variant="minimize"
              aria-label="Minimize window"
              tabIndex={0}
            />
            <WindowButton
              variant="maximize"
              aria-label="Maximize window"
              tabIndex={0}
            />
          </WindowControls>
          <TitleText id={`${windowId}-title`}>{title}</TitleText>
          <div style={{ width: 52 }} aria-hidden="true" />
        </TitleBar>
        <Content>{children}</Content>
      </WindowContainer>
    </FocusScope>
  )
}
```

### 1.5 Create Window Content Styled Components

**File:** `app/src/components/ui/WindowContent.tsx`

```tsx
import styled from "@emotion/styled"

export const WindowBody = styled.div`
  padding: 1rem;
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
`

export const WindowFooter = styled.div`
  display: flex;
  justify-content: flex-end;
  gap: 0.75rem;
  padding: 1rem;
  border-top: 1px solid ${({ theme }) => theme.dividerColor};
  background: ${({ theme }) => theme.secondaryBackgroundColor};
`
```

---

## Phase 2: Update Existing Modals

### 2.1 Update InstrumentBrowser

**File:** `app/src/components/InstrumentBrowser/InstrumentBrowser.tsx`

**Changes:**
1. Replace `Dialog` with `DraggableWindow`
2. Use fixed hook pattern (from Phase 0.1)
3. Add proper `windowId`

```tsx
import styled from "@emotion/styled"
import { FC } from "react"
import { useInstrumentBrowser } from "../../hooks/useInstrumentBrowser"
import { Localized } from "../../localize/useLocalization"
import { DraggableWindow } from "../ui/DraggableWindow"
import { WindowBody, WindowFooter } from "../ui/WindowContent"
import { InstrumentName } from "../TrackList/InstrumentName"
import { Button, PrimaryButton } from "../ui/Button"
import { Checkbox } from "../ui/Checkbox"
import { Label } from "../ui/Label"
import { DrumKitCategoryName, FancyCategoryName } from "./CategoryName"
import { SelectBox } from "./SelectBox"

// Keep existing styled components: Finder, Left, Right, Footer

export const InstrumentBrowser: FC = () => {
  const {
    isOpen,
    setOpen,
    setting: { programNumber, isRhythmTrack },
    selectedCategoryIndex,
    categoryFirstProgramEvents,
    categoryInstruments,
    onChangeInstrument: onChange,
    onClickOK,
    onChangeRhythmTrack,
  } = useInstrumentBrowser()

  const categoryOptions = categoryFirstProgramEvents.map((preset, i) => ({
    value: i,
    label: isRhythmTrack ? (
      <DrumKitCategoryName />
    ) : (
      <FancyCategoryName programNumber={preset} />
    ),
  }))

  const instrumentOptions = categoryInstruments.map((p) => ({
    value: p,
    label: <InstrumentName programNumber={p} isRhythmTrack={isRhythmTrack} />,
  }))

  return (
    <DraggableWindow
      windowId="instrument-browser"
      title="Instruments"
      isOpen={isOpen}
      onClose={() => setOpen(false)}
      canClose={true}
      minWidth={580}
      minHeight={400}
    >
      <WindowBody>
        <Finder>
          <Left>
            <Label style={{ marginBottom: "0.5rem" }}>
              <Localized name="categories" />
            </Label>
            <SelectBox
              items={categoryOptions}
              selectedValue={selectedCategoryIndex}
              onChange={(i) => onChange(i * 8)}
            />
          </Left>
          <Right>
            <Label style={{ marginBottom: "0.5rem" }}>
              <Localized name="instruments" />
            </Label>
            <SelectBox
              items={instrumentOptions}
              selectedValue={programNumber}
              onChange={onChange}
            />
          </Right>
        </Finder>
        <Footer>
          <Checkbox
            checked={isRhythmTrack}
            onCheckedChange={(state) => onChangeRhythmTrack(state === true)}
            label={<Localized name="rhythm-track" />}
          />
        </Footer>
      </WindowBody>
      <WindowFooter>
        <Button onClick={() => setOpen(false)}>
          <Localized name="cancel" />
        </Button>
        <PrimaryButton onClick={onClickOK}>
          <Localized name="ok" />
        </PrimaryButton>
      </WindowFooter>
    </DraggableWindow>
  )
}
```

### 2.2 Update SettingDialog

**File:** `app/src/components/SettingDialog/SettingDialog.tsx`

```tsx
import styled from "@emotion/styled"
import { FC, useCallback, useState } from "react"
import { useRootView } from "../../hooks/useRootView"
import { Localized } from "../../localize/useLocalization"
import { DraggableWindow } from "../ui/DraggableWindow"
import { WindowBody, WindowFooter } from "../ui/WindowContent"
import { Button } from "../ui/Button"
import { GeneralSettingsView } from "./GeneralSettingsView"
import { MIDIDeviceView } from "./MIDIDeviceView/MIDIDeviceView"
import { SettingNavigation, SettingRoute } from "./SettingNavigation"
import { ShortcutsSettingsView } from "./ShortcutsSettingsView"
import { SoundFontSettingsView } from "./SoundFontSettingView"

const RouteContent: FC<{ route: SettingRoute }> = ({ route }) => {
  switch (route) {
    case "general":
      return <GeneralSettingsView />
    case "midi":
      return <MIDIDeviceView />
    case "soundfont":
      return <SoundFontSettingsView />
    case "shortcuts":
      return <ShortcutsSettingsView />
  }
}

const SettingsLayout = styled.div`
  display: flex;
  flex-direction: row;
  flex: 1;
  min-height: 0;
`

const Content = styled.div`
  flex-grow: 1;
  padding: 1rem;
  overflow-y: auto;
`

export const SettingDialog: FC = () => {
  const { openSettingDialog: open, setOpenSettingDialog } = useRootView()
  const [route, setRoute] = useState<SettingRoute>("general")

  const onClose = useCallback(
    () => setOpenSettingDialog(false),
    [setOpenSettingDialog],
  )

  return (
    <DraggableWindow
      windowId="settings"
      title="Settings"
      isOpen={open}
      onClose={onClose}
      canClose={true}
      minWidth={600}
      minHeight={480}
    >
      <WindowBody style={{ padding: 0, display: "flex", flex: 1 }}>
        <SettingsLayout>
          <SettingNavigation route={route} onChange={setRoute} />
          <Content>
            <RouteContent route={route} />
          </Content>
        </SettingsLayout>
      </WindowBody>
      <WindowFooter>
        <Button onClick={onClose}>
          <Localized name="close" />
        </Button>
      </WindowFooter>
    </DraggableWindow>
  )
}
```

---

## Phase 3: Transform AIChat to Persistent Modal

### 3.1 Update useAIChat Hook

**File:** `app/src/hooks/useAIChat.ts`

Remove toggle/visibility controls - the modal is always visible (except InitialView).

```ts
import { atom, useAtomValue, useSetAtom } from "jotai"
import { GenerationStage } from "../services/aiBackend"

// Remove: aiChatOpenAtom is no longer needed for visibility
// The modal is always visible when not in InitialView

// ... existing atoms for messages, loading state, etc. unchanged ...

export function useAIChat() {
  // Remove: isOpen, setOpen, toggle, show, hide
  // Keep only the chat state management
  
  const messages = useAtomValue(aiChatMessagesAtom)
  const setMessages = useSetAtom(aiChatMessagesAtom)
  const isLoading = useAtomValue(aiChatIsLoadingAtom)
  const setIsLoading = useSetAtom(aiChatIsLoadingAtom)
  // ... other state ...

  return {
    // Remove: isOpen, setOpen, toggle, show, hide
    messages,
    setMessages,
    isLoading,
    setIsLoading,
    // ... rest of chat state ...
  }
}
```

**Note:** The `isOpen` state is removed entirely. The modal visibility is now controlled by RootView based on whether we're in InitialView or not.

### 3.2 Create AIChatModal Wrapper

**File:** `app/src/components/AIChat/AIChatModal.tsx`

The modal is always open - visibility is controlled by RootView's conditional render.

```tsx
import { FC } from "react"
import styled from "@emotion/styled"
import { DraggableWindow } from "../ui/DraggableWindow"
import { AIChat } from "./AIChat"

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 400px;
`

export const AIChatModal: FC = () => {
  // Calculate default position (right side of screen)
  const getDefaultPosition = () => ({
    x: typeof window !== "undefined" ? Math.max(50, window.innerWidth - 450) : 100,
    y: 80,
  })

  return (
    <DraggableWindow
      windowId="ai-chat"
      title="AI Composer"
      isOpen={true}  // Always open - RootView controls when this component renders
      canClose={false}  // Cannot be closed
      defaultPosition={getDefaultPosition()}
      minWidth={400}
      minHeight={500}
      maxHeight={700}
    >
      <ChatContainer>
        <AIChat standalone />
      </ChatContainer>
    </DraggableWindow>
  )
}
```

### 3.3 Update AIChat Component

**File:** `app/src/components/AIChat/AIChat.tsx`

**Changes:**
1. Remove header close button (window chrome handles this)
2. Simplify container styling
3. Keep all functionality intact

```tsx
// Update Container - remove border-left, simpler styling
const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
  background: ${({ theme }) => theme.backgroundColor};
  overflow: hidden;
`

// Update Header - simplified, no close button needed in modal
const Header = styled.div`
  padding: 0.75rem 1rem;
  border-bottom: 1px solid ${({ theme }) => theme.dividerColor};
  font-weight: 600;
  font-size: 0.8125rem;
  color: ${({ theme }) => theme.textColor};
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  flex-shrink: 0;
`

// In the component JSX, the Header section becomes:
<Header>
  <HeaderLeft>
    <Tooltip title="Choose generation method">
      <Select
        value={agentType}
        onChange={handleAgentTypeChange}
        disabled={isLoading}
      >
        <option value="hybrid">Hybrid Agent</option>
        <option value="hybrid_legacy">Hybrid (Legacy)</option>
        <option value="composition_agent">Deep Agent</option>
        <option value="llm">LLM Direct</option>
      </Select>
    </Tooltip>
  </HeaderLeft>
  {activeThreadId && (
    <Tooltip title="Start fresh conversation">
      <NewChatButton onClick={handleNewChat} disabled={isLoading}>
        New Chat
      </NewChatButton>
    </Tooltip>
  )}
  <HeaderRight>
    <Tooltip
      title={
        backendStatus === "connected"
          ? "Backend connected"
          : backendStatus === "disconnected"
            ? "Backend not available"
            : "Checking connection..."
      }
    >
      <StatusDot status={backendStatus} />
    </Tooltip>
    {/* Close button removed - window chrome handles this */}
  </HeaderRight>
</Header>
```

---

## Phase 4: Update App Layout

### 4.1 Update RootView

**File:** `app/src/components/RootView/RootView.tsx`

**Changes:**
1. Add `AIChatModal` import and render
2. Keep it outside conditional rendering so it's always available

```tsx
import { AIChatModal } from "../AIChat/AIChatModal"

export const RootView: FC = () => {
  // ... existing code ...

  return (
    <>
      {shouldShowInitialView ? (
        // ... InitialView rendering ...
      ) : (
        // ... Main app rendering ...
      )}
      
      {/* Draggable window modals - always available */}
      <AIChatModal />
      
      {/* Existing dialogs - these still use Radix for blocking behavior */}
      <ExportProgressDialog />
      <Head />
      <SettingDialog />
      <ControlSettingDialog />
      <HQAudioPlayer />
      <OnInit />
      <OnBeforeUnload />
    </>
  )
}
```

### 4.2 Update ArrangeEditor

**File:** `app/src/components/ArrangeView/ArrangeEditor.tsx`

**Changes:**
Remove split pane AIChat - it's now a floating modal.

```tsx
// Remove these imports:
// import { AIChat } from "../AIChat/AIChat"
// import { useAIChat } from "../../hooks/useAIChat"

// Remove StyledSplitPane import if no longer needed elsewhere

export const ArrangeEditor: FC = () => {
  // Remove: const { isOpen: isAIChatOpen } = useAIChat()

  return (
    <ArrangeViewScope>
      <MainContainer>
        <Content />
      </MainContainer>
    </ArrangeViewScope>
  )
}
```

### 4.3 Update PianoRollEditor

**File:** `app/src/components/PianoRoll/PianoRollEditor.tsx`

**Changes:**
Same as ArrangeEditor - remove split pane.

```tsx
export const PianoRollEditor: FC = () => {
  return (
    <PianoRollScope>
      <MainContainer>
        <EditorContent />
      </MainContainer>
      <PianoRollTransposeDialog />
      <PianoRollVelocityDialog />
    </PianoRollScope>
  )
}
```

### 4.4 Update InitialView

**File:** `app/src/components/InitialView/InitialView.tsx`

No changes needed - InitialView continues to render the embedded AIChat. The floating AIChatModal is simply not rendered when in InitialView (controlled by RootView).

```tsx
import styled from "@emotion/styled"
import { FC } from "react"
import { AIChat } from "../AIChat/AIChat"

const Container = styled.div`
  height: 100vh;
  width: 100vw;
  display: flex;
  align-items: center;
  justify-content: center;
  background: ${({ theme }) => theme.backgroundColor};
`

const ChatWrapper = styled.div`
  width: 100%;
  max-width: 800px;
  height: 100%;
  max-height: 90vh;
  display: flex;
  flex-direction: column;
`

export const InitialView: FC = () => {
  return (
    <Container>
      <ChatWrapper>
        <AIChat standalone={true} />
      </ChatWrapper>
    </Container>
  )
}
```

**RootView controls AIChatModal visibility:**

```tsx
export const RootView: FC = () => {
  // ... existing code ...
  
  return (
    <>
      {/* ... existing layout ... */}
      
      {/* Floating modal only rendered when not in InitialView */}
      {!shouldShowInitialView && <AIChatModal />}
      
      {/* ... existing dialogs ... */}
    </>
  )
}
```

### 4.5 Remove Navigation Toggle Button (if exists)

**File:** `app/src/components/Navigation/Navigation.tsx` (or wherever toggle exists)

**Remove any AI chat toggle button** - the modal is always visible and cannot be hidden.

Search for and remove:
- Any button that calls `toggle()` or `setOpen()` on useAIChat
- Any UI element for showing/hiding the AI chat panel

---

## Phase 5: Testing & Verification

### 5.1 Test Checklist

- [ ] Dragging works on all windows (mouse and touch)
- [ ] Windows stay in viewport on resize
- [ ] Z-index stacking works (click to bring forward)
- [ ] Escape closes closable windows (InstrumentBrowser, Settings)
- [ ] AI Chat cannot be closed or hidden (always visible)
- [ ] Focus trapping works in frontmost window
- [ ] Tab navigation stays within window
- [ ] Window positions persist across open/close
- [ ] Window positions persist across page reload
- [ ] No duplicate AIChat renders
- [ ] InitialView shows embedded chat (no floating modal)
- [ ] Transitioning from InitialView shows floating modal
- [ ] No toggle button/shortcut for AI chat exists
- [ ] Screen reader announces window open/close

### 5.2 Known Limitations

1. **Minimize not implemented** - Buttons are visual only for now
2. **Maximize not implemented** - Would need additional state
3. **No window resize handles** - Fixed size windows
4. **Mobile UX** - Works but not optimized (consider fullscreen on small screens)

---

## Files Summary

### Files to Create

| File | Description |
|------|-------------|
| `app/src/stores/windowPositions.ts` | Jotai atoms for window state |
| `app/src/hooks/useWindowManager.ts` | Z-index and position management |
| `app/src/hooks/useWindowDrag.ts` | Drag handling with cleanup |
| `app/src/components/ui/DraggableWindow.tsx` | Core window component |
| `app/src/components/ui/WindowContent.tsx` | Body/footer styled components |
| `app/src/components/AIChat/AIChatModal.tsx` | AI chat modal wrapper |

### Files to Modify

| File | Changes |
|------|---------|
| `app/src/hooks/useInstrumentBrowser.ts` | Fix hook violations (Phase 0) |
| `app/src/hooks/useAIChat.ts` | Remove isOpen/toggle/show/hide (no longer needed) |
| `app/src/components/InstrumentBrowser/InstrumentBrowser.tsx` | Use DraggableWindow |
| `app/src/components/SettingDialog/SettingDialog.tsx` | Use DraggableWindow |
| `app/src/components/AIChat/AIChat.tsx` | Remove close button, simplify |
| `app/src/components/RootView/RootView.tsx` | Add AIChatModal (conditional on InitialView) |
| `app/src/components/ArrangeView/ArrangeEditor.tsx` | Remove split pane |
| `app/src/components/PianoRoll/PianoRollEditor.tsx` | Remove split pane |
| `app/src/components/InitialView/InitialView.tsx` | Remove useAIChat import (no longer needed) |
| `app/src/components/Navigation/Navigation.tsx` | Remove AI chat toggle button |
| `app/src/hooks/useGlobalKeyboardShortcut.tsx` | Remove AI chat toggle shortcut |

### Dependencies

**Existing (no changes):**
- `@emotion/styled`
- `jotai` (+ `jotai/utils` for `atomWithStorage`)
- `@signal-app/core`

**New Required:**
- `@radix-ui/react-focus-scope` - For focus trapping (likely already installed with other Radix packages)

---

## Implementation Order

1. **Phase 0.1** - Fix useInstrumentBrowser hook violation
2. **Phase 0.2** - Audit AIChat toggle call sites
3. **Phase 1.1** - Create window position store
4. **Phase 1.2** - Create useWindowManager hook
5. **Phase 1.3** - Create useWindowDrag hook
6. **Phase 1.4** - Create DraggableWindow component
7. **Phase 1.5** - Create WindowContent components
8. **Phase 2.1** - Update InstrumentBrowser
9. **Phase 2.2** - Update SettingDialog
10. **Phase 3.1** - Update useAIChat hook
11. **Phase 3.2** - Create AIChatModal
12. **Phase 3.3** - Update AIChat component
13. **Phase 4.1** - Update RootView
14. **Phase 4.2** - Update ArrangeEditor
15. **Phase 4.3** - Update PianoRollEditor
16. **Phase 4.4** - Update InitialView
17. **Phase 5** - Test all functionality
