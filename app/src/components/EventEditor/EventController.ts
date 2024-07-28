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
              update: flow(
                createParseClampedInt(0, 127),
                optional((value) => ({ value })),
              ),
            },
          }
        case "note":
          return {
            name: e.subtype,
            value: {
              type: "number",
              value: e.velocity,
              update: flow(
                createParseClampedInt(0, 127),
                optional((velocity) => ({ velocity })),
              ),
            },
            gate: {
              type: "number",
              value: e.duration,
              update: flow(
                createParseClampedInt(0, Infinity),
                optional((duration) => ({ duration })),
              ),
            },
          }
        case "programChange":
          return {
            name: e.subtype,
            value: {
              type: "number",
              value: e.value,
              update: flow(
                createParseClampedInt(0, 127),
                optional((value) => ({ value })),
              ),
            },
          }
        case "pitchBend":
          return {
            name: e.subtype,
            value: {
              type: "number",
              value: e.value,
              update: flow(
                createParseClampedInt(0, 16384),
                optional((value) => ({ value })),
              ),
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
              update: flow(
                createParseClampedInt(0, 127),
                optional((channel) => ({ channel })),
              ),
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

const createParseClampedInt = (min: number, max: number) => (value: string) => {
  const num = parseInt(value)
  if (Number.isNaN(num)) {
    return null
  }
  return clamp(num, min, max)
}

const optional =
  <T, S>(fn: (value: T) => S) =>
  (value: T | null) => {
    if (value === null) {
      return null
    }
    return fn(value)
  }
