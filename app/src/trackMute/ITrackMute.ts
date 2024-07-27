import { TrackId } from "../track/Track"

export interface ITrackMute {
  shouldPlayTrack(trackId: TrackId): boolean
}
