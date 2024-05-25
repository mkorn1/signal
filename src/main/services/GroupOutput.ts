import { SendableEvent, SynthOutput } from "@signal-app/player"
import { ITrackMute } from "../../common/trackMute/ITrackMute"

export interface SynthEntry {
  synth: SynthOutput
  isEnabled: boolean
}

// Routing of MIDI events to multiple SynthOutputs and muting of tracks
export class GroupOutput implements SynthOutput {
  outputs: SynthEntry[] = []

  constructor(private readonly trackMute: ITrackMute) {}

  activate() {
    this.outputs.filter((o) => o.isEnabled).forEach((o) => o.synth.activate())
  }

  sendEvent(
    event: SendableEvent,
    delayTime: number,
    timestampNow: number,
    trackId?: number,
  ): void {
    if (
      trackId !== undefined &&
      this.trackMute !== null &&
      !this.trackMute.shouldPlayTrack(trackId)
    ) {
      return
    }

    this.outputs
      .filter((o) => o.isEnabled)
      .forEach((o) =>
        o.synth.sendEvent(event, delayTime, timestampNow, trackId),
      )
  }
}
