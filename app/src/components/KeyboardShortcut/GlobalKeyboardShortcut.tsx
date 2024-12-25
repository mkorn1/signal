import { observer } from "mobx-react-lite"
import { FC, useEffect } from "react"
import {
  toggleRecording,
  useFastForwardOneBar,
  useNextTrack,
  usePreviousTrack,
  useRewindOneBar,
  useStop,
  useToggleGhost,
  useToggleMute,
  useToggleSolo,
} from "../../actions"
import { redo, undo } from "../../actions/history"
import { useStores } from "../../hooks/useStores"
import { KeyboardShortcut } from "./KeyboardShortcut"

export const GlobalKeyboardShortcut: FC = observer(() => {
  const rootStore = useStores()
  const { rootViewStore, router, player } = rootStore
  const rewindOneBar = useRewindOneBar()
  const fastForwardOneBar = useFastForwardOneBar()
  const stop = useStop()
  const nextTrack = useNextTrack()
  const previousTrack = usePreviousTrack()
  const toggleSolo = useToggleSolo()
  const toggleMute = useToggleMute()
  const toggleGhost = useToggleGhost()

  useEffect(() => {
    // prevent zooming
    const onWheel = (e: WheelEvent) => {
      // Touchpad pinches are translated into wheel with ctrl event
      if (e.ctrlKey) {
        e.preventDefault()
      }
    }

    document.addEventListener("wheel", onWheel, { passive: false })

    // disable bounce scroll (Safari does not support overscroll-behavior CSS)
    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault
    }

    document.addEventListener("touchmove", onTouchMove, { passive: false })

    // do not allow to open the default context menu
    document.oncontextmenu = (e) => e.preventDefault()

    return () => {
      document.removeEventListener("wheel", onWheel)
      document.removeEventListener("touchmove", onTouchMove)
      document.oncontextmenu = null
    }
  }, [])

  return (
    <KeyboardShortcut
      actions={[
        { code: "Space", run: () => player.playOrPause() },
        {
          code: "KeyZ",
          metaKey: true,
          shiftKey: true,
          run: () => redo(rootStore)(),
        },
        {
          code: "KeyZ",
          metaKey: true,
          shiftKey: false,
          run: () => undo(rootStore)(),
        },
        { code: "KeyY", metaKey: true, run: () => redo(rootStore)() },
        {
          // Press ?
          code: "Slash",
          shiftKey: true,
          run: () => (rootViewStore.openHelp = true),
        },
        { code: "Enter", run: stop },
        { code: "KeyA", run: rewindOneBar },
        { code: "KeyD", run: fastForwardOneBar },
        { code: "KeyS", run: () => nextTrack() },
        { code: "KeyW", run: () => previousTrack() },
        { code: "KeyN", run: () => toggleSolo() },
        { code: "KeyM", run: () => toggleMute() },
        { code: "KeyR", run: () => toggleRecording(rootStore)() },
        { code: "Comma", run: () => toggleGhost() },
        {
          code: "Digit1",
          metaKey: true,
          run: () => (router.path = "/track"),
        },
        {
          code: "Digit2",
          metaKey: true,
          run: () => (router.path = "/arrange"),
        },
        {
          code: "Digit3",
          metaKey: true,
          run: () => (router.path = "/tempo"),
        },
      ]}
    />
  )
})
