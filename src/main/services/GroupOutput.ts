import { SendableEvent, SynthOutput } from "@signal-app/player"
import { makeObservable, observable } from "mobx"
import { METRONOME_TRACK_ID } from "../../common/player/EventSource"
import { ITrackMute } from "../../common/trackMute/ITrackMute"

export interface SynthEntry {
  synth: SynthOutput
  isEnabled: boolean
}

// Routing of MIDI events to multiple SynthOutputs and muting of tracks
export class GroupOutput implements SynthOutput {
  outputs: SynthEntry[] = []
  isMetronomeEnabled: boolean = false

  constructor(
    private readonly trackMute: ITrackMute,
    private readonly metronomeOutput: SynthOutput,
  ) {
    makeObservable(this, {
      isMetronomeEnabled: observable,
    })
  }

  activate() {
    this.outputs.filter((o) => o.isEnabled).forEach((o) => o.synth.activate())
  }

  private getOutputs(trackId: number | undefined): SynthOutput[] {
    if (trackId === METRONOME_TRACK_ID) {
      return this.isMetronomeEnabled ? [this.metronomeOutput] : []
    } else if (
      trackId !== undefined &&
      this.trackMute !== null &&
      !this.trackMute.shouldPlayTrack(trackId)
    ) {
      return []
    } else {
      return this.outputs.filter((o) => o.isEnabled).map((o) => o.synth)
    }
  }

  sendEvent(
    event: SendableEvent,
    delayTime: number,
    timestampNow: number,
    trackId?: number,
  ): void {
    this.getOutputs(trackId).forEach((synth) =>
      synth.sendEvent(event, delayTime, timestampNow, trackId),
    )
  }
}
