import { clamp, flow } from "lodash"
import { bpmToUSecPerBeat, uSecPerBeatToBPM } from "../../helpers/bpm"
import { controllerTypeString } from "../../helpers/noteNumberString"
import { TrackEvent } from "../../track"

interface NumberEventInputProp {
  type: "number"
  value: number
}

interface TextEventInputProp {
  type: "text"
  value: string
}

export type EventInputProp = NumberEventInputProp | TextEventInputProp

export type EventValueUpdator = {
  // null means no update
  update: (value: string) => any | null
}

// Abstraction Layer for manipulating TrackEvent on EventList
export type EventController = {
  name: string
  gate?: EventInputProp & EventValueUpdator
  value?: EventInputProp & EventValueUpdator
}

export function getEventController<T extends TrackEvent>(
  e: T,
): EventController {
  switch (e.type) {
    case "channel":
      switch (e.subtype) {
        case "controller":
          return {
            name:
              controllerTypeString(e.controllerType) ?? `CC${e.controllerType}`,
            value: {
              type: "number",
              value: e.value,
              update: intConverter(0, 127, (value) => ({ value })),
            },
          }
        case "note":
          return {
            name: e.subtype,
            value: {
              type: "number",
              value: e.velocity,
              update: intConverter(0, 127, (velocity) => ({ velocity })),
            },
            gate: {
              type: "number",
              value: e.duration,
              update: intConverter(0, Infinity, (duration) => ({ duration })),
            },
          }
        case "programChange":
          return {
            name: e.subtype,
            value: {
              type: "number",
              value: e.value,
              update: intConverter(0, 127, (value) => ({ value })),
            },
          }
        case "pitchBend":
          return {
            name: e.subtype,
            value: {
              type: "number",
              value: e.value,
              update: intConverter(0, 16384, (value) => ({ value })),
            },
          }
        default:
          return { name: e.subtype }
      }
    case "meta":
      switch (e.subtype) {
        case "trackName":
          return {
            name: e.subtype,
            value: {
              type: "text",
              value: e.text,
              update: (text) => ({ text }),
            },
          }
        case "midiChannelPrefix":
          return {
            name: e.subtype,
            value: {
              type: "number",
              value: e.value,
              update: intConverter(0, 127, (channel) => ({ channel })),
            },
          }
        default:
          return { name: e.subtype }
      }
    case "dividedSysEx":
    case "sysEx":
      return { name: e.type }
  }
}

const nanToNull = (value: number) => {
  if (Number.isNaN(value)) {
    return null
  }
  return value
}

const createClamp = (min: number, max: number) => (value: number) =>
  clamp(value, min, max)

const optional =
  <T, S>(fn: (value: T) => S) =>
  (value: T | null) => {
    if (value === null) {
      return null
    }
    return fn(value)
  }

const intConverter = <T>(
  minValue: number,
  maxValue: number,
  fn: (value: number) => T,
) => flow(parseInt, createClamp(minValue, maxValue), nanToNull, optional(fn))
