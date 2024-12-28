import { useStores } from "../hooks/useStores"

export const useToggleRecording = () => {
  const { midiRecorder, player } = useStores()
  return () => {
    if (midiRecorder.isRecording) {
      midiRecorder.isRecording = false
      player.stop()
    } else {
      midiRecorder.isRecording = true
      player.play()
    }
  }
}
