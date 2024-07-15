import RootStore from "../stores/RootStore"

export const toggleMuteTrack =
  (rootStore: RootStore) => (trackIndex: number) => {
    const {
      song: { tracks },
      trackMute,
      player,
    } = rootStore
    const channel = tracks[trackIndex].channel
    if (channel === undefined) {
      return
    }

    if (trackMute.isMuted(trackIndex)) {
      trackMute.unmute(trackIndex)
    } else {
      trackMute.mute(trackIndex)
      player.allSoundsOffChannel(channel)
    }
  }

export const toggleSoloTrack =
  (rootStore: RootStore) => (trackIndex: number) => {
    const {
      song: { tracks },
      trackMute,
      player,
    } = rootStore
    const channel = tracks[trackIndex].channel
    if (channel === undefined) {
      return
    }

    if (trackMute.isSolo(trackIndex)) {
      trackMute.unsolo(trackIndex)
      player.allSoundsOffChannel(channel)
    } else {
      trackMute.solo(trackIndex)
      player.allSoundsOffExclude(channel)
    }
  }
