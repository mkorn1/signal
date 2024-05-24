import { ControllerEvent, NoteOffEvent, NoteOnEvent } from "midifile-ts"

export function controllerMidiEvent(
  deltaTime: number,
  channel: number,
  controllerType: number,
  value: number,
): ControllerEvent {
  return {
    deltaTime,
    type: "channel",
    subtype: "controller",
    channel,
    controllerType,
    value,
  }
}

export function noteOffMidiEvent(
  deltaTime: number,
  channel: number,
  noteNumber: number,
  velocity: number = 0,
): NoteOffEvent {
  return {
    deltaTime,
    type: "channel",
    subtype: "noteOff",
    channel,
    noteNumber,
    velocity,
  }
}

export function noteOnMidiEvent(
  deltaTime: number,
  channel: number,
  noteNumber: number,
  velocity: number,
): NoteOnEvent {
  return {
    deltaTime,
    type: "channel",
    subtype: "noteOn",
    channel,
    noteNumber,
    velocity,
  }
}
