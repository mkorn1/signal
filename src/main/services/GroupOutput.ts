import { SendableEvent, SynthOutput } from "@signal-app/player"

export interface SynthEntry {
  synth: SynthOutput
  isEnabled: boolean
}

export class GroupOutput implements SynthOutput {
  outputs: SynthEntry[] = []

  activate() {
    this.outputs.filter((o) => o.isEnabled).forEach((o) => o.synth.activate())
  }

  sendEvent(
    event: SendableEvent,
    delayTime: number,
    timestampNow: number,
  ): void {
    this.outputs
      .filter((o) => o.isEnabled)
      .forEach((o) => o.synth.sendEvent(event, delayTime, timestampNow))
  }
}
