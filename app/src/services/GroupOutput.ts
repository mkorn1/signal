import { SendableEvent, SynthOutput } from "@signal-app/player"
import { makeObservable, observable } from "mobx"
import { METRONOME_TRACK_ID } from "../player/EventSource"
import { ITrackMute } from "../trackMute/ITrackMute"

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

  private getOutputs(trackIndex: number | undefined): SynthOutput[] {
    if (trackIndex === METRONOME_TRACK_ID) {
      return this.isMetronomeEnabled ? [this.metronomeOutput] : []
    } else if (
      trackIndex !== undefined &&
      this.trackMute !== null &&
      !this.trackMute.shouldPlayTrack(trackIndex)
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
    trackIndex?: number,
  ): void {
    this.getOutputs(trackIndex).forEach((synth) =>
      synth.sendEvent(event, delayTime, timestampNow, trackIndex),
    )
  }
}
