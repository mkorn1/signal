import { useTheme } from "@emotion/react"
import { Rectangles } from "@ryohey/webgl-react"
import Color from "color"
import { FC, useMemo } from "react"
import { colorToVec4 } from "../../../gl/color"
import { useArrangeView } from "../../../hooks/useArrangeView"
import { useSong } from "../../../hooks/useSong"
import { useTickScroll } from "../../../hooks/useTickScroll"
import { useTrackScroll } from "../../../hooks/useTrackScroll"

const AUDIO_CLIP_HEIGHT = 20 // Height in pixels for audio clips

export const AudioClips: FC<{ zIndex: number }> = ({ zIndex }) => {
  const { trackTransform } = useArrangeView()
  const { transform: tickTransform } = useTickScroll()
  const { trackHeight } = useTrackScroll()
  const { tracks } = useSong()
  const theme = useTheme()

  const audioClips = useMemo(() => {
    const clips: Array<{
      x: number
      y: number
      width: number
      height: number
    }> = []

    tracks.forEach((track, trackIndex) => {
      // Check if this is an audio track
      const isAudioTrack = (track as any).isAudioTrack
      const audioData = (track as any).audioData

      if (isAudioTrack && audioData && track.endOfTrack > 0) {
        // Create a rectangle for the audio clip
        // Position it at the start of the track (tick 0)
        const x = tickTransform.getX(0)
        const width = tickTransform.getX(track.endOfTrack) - x
        const y = trackTransform.getY(trackIndex) + (trackHeight - AUDIO_CLIP_HEIGHT) / 2

        clips.push({
          x,
          y,
          width: Math.max(width, 10), // Minimum width for visibility
          height: AUDIO_CLIP_HEIGHT,
        })
      }
    })

    return clips
  }, [tracks, tickTransform, trackTransform, trackHeight])

  if (audioClips.length === 0) {
    return null
  }

  // Use a different color for audio clips (e.g., a lighter/different shade)
  const audioColor = Color(theme.themeColor).lighten(0.3)

  return (
    <Rectangles
      rects={audioClips}
      color={colorToVec4(audioColor)}
      zIndex={zIndex}
    />
  )
}
