import { useStores } from "../hooks/useStores"
import { TrackId } from "../track"

export const useToggleMuteTrack = () => {
  const { song, trackMute, player } = useStores()

  return (trackId: TrackId) => {
    const channel = song.getTrack(trackId)?.channel
    if (channel === undefined) {
      return
    }

    if (trackMute.isMuted(trackId)) {
      trackMute.unmute(trackId)
    } else {
      trackMute.mute(trackId)
      player.allSoundsOffChannel(channel)
    }
  }
}

export const useToggleSoloTrack = () => {
  const { song, trackMute, player } = useStores()

  return (trackId: TrackId) => {
    const channel = song.getTrack(trackId)?.channel
    if (channel === undefined) {
      return
    }

    if (trackMute.isSolo(trackId)) {
      trackMute.unsolo(trackId)
      player.allSoundsOffChannel(channel)
    } else {
      trackMute.solo(trackId)
      player.allSoundsOffExclude(channel)
    }
  }
}
