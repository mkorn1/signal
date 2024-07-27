import RootStore from "../stores/RootStore"
import { TrackId } from "../track"

export const toggleMuteTrack = (rootStore: RootStore) => (trackId: TrackId) => {
  const { song, trackMute, player } = rootStore
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

export const toggleSoloTrack = (rootStore: RootStore) => (trackId: TrackId) => {
  const { song, trackMute, player } = rootStore
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
