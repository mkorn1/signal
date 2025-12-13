import {
  createGeneraterObject,
  getInstrumentGenerators,
  ParseResult,
} from "@signal-app/sf2parser"

export function getInstrumentZones(parsed: ParseResult, instrumentID: number) {
  const instrumentGenerators = getInstrumentGenerators(parsed, instrumentID)
  const zones = instrumentGenerators.map(createGeneraterObject)

  // Handle empty zones array
  if (zones.length === 0) {
    return {
      zones: [],
      globalZone: undefined,
    }
  }

  // If the first zone does not have sampleID, it is a global instrument zone.
  let globalZone: any | undefined
  const firstInstrumentZone = zones[0]
  if (firstInstrumentZone && firstInstrumentZone.sampleID === undefined) {
    globalZone = zones[0]
  }

  return {
    zones: zones.filter((zone) => zone.sampleID !== undefined),
    globalZone,
  }
}
