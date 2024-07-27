import { TrackId } from "../track/Track"
import TrackMute from "./TrackMute"

function getTrackId(value: number): TrackId {
  return value as TrackId
}

describe("TrackMute", () => {
  it("not muted by default", () => {
    const t = new TrackMute()
    expect(t.isMuted(getTrackId(getTrackId(0)))).toBeFalsy()
    expect(t.isMuted(getTrackId(100))).toBeFalsy()
  })

  it("mute", () => {
    const t = new TrackMute()
    expect(t.isMuted(getTrackId(getTrackId(0)))).toBeFalsy()
    t.mute(getTrackId(getTrackId(0)))
    expect(t.isMuted(getTrackId(getTrackId(0)))).toBeTruthy()
    expect(t.shouldPlayTrack(getTrackId(getTrackId(0)))).toBeFalsy()
    t.unmute(getTrackId(getTrackId(0)))
    expect(t.isMuted(getTrackId(getTrackId(0)))).toBeFalsy()
  })

  it("solo", () => {
    const t = new TrackMute()
    expect(t.isSolo(getTrackId(getTrackId(0)))).toBeFalsy()
    t.solo(getTrackId(0))
    expect(t.isSolo(getTrackId(0))).toBeTruthy()
    expect(t.isSoloMode()).toBeTruthy()
    expect(t.isMuted(getTrackId(1))).toBeTruthy()
    expect(t.shouldPlayTrack(getTrackId(0))).toBeTruthy()
    expect(t.shouldPlayTrack(getTrackId(1))).toBeFalsy()
    t.solo(getTrackId(1))
    expect(t.isSolo(getTrackId(0))).toBeTruthy()
    expect(t.isSolo(getTrackId(1))).toBeTruthy()
    expect(t.isSoloMode()).toBeTruthy()
    expect(t.isMuted(getTrackId(0))).toBeFalsy()
    expect(t.isMuted(getTrackId(1))).toBeFalsy()
    expect(t.isMuted(getTrackId(2))).toBeTruthy()
    expect(t.shouldPlayTrack(getTrackId(0))).toBeTruthy()
    expect(t.shouldPlayTrack(getTrackId(1))).toBeTruthy()
    expect(t.shouldPlayTrack(getTrackId(2))).toBeFalsy()
    t.unsolo(getTrackId(0))
    expect(t.isSolo(getTrackId(0))).toBeFalsy()
    expect(t.isSoloMode()).toBeTruthy()
    t.unsolo(getTrackId(1))
    expect(t.isSolo(getTrackId(1))).toBeFalsy()
    expect(t.isSoloMode()).toBeFalsy()
  })
})
