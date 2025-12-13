import {
  parse,
  getPresetGenerators,
  getInstrumentGenerators,
  createGeneraterObject,
} from "@signal-app/sf2parser"
import { getPresetZones } from "../soundfont/getPresetZones"

export interface ZoneDiagnostic {
  presetIndex: number
  presetName: string
  bank: number
  preset: number
  instrumentID: number | undefined
  instrumentName: string | undefined
  zoneIndex: number
  sampleID: number | undefined
  isGlobalZone: boolean
  generators: Record<string, unknown>
}

export interface SF2Diagnostic {
  filename: string
  totalPresets: number
  totalInstruments: number
  totalSamples: number
  zones: ZoneDiagnostic[]
  zonesWithUndefinedSampleID: ZoneDiagnostic[]
  summary: {
    totalZones: number
    globalZones: number
    validZones: number
    invalidZones: number // Non-global zones with undefined sampleID
  }
}

function getInstrumentZonesRaw(parsed: ReturnType<typeof parse>, instrumentID: number) {
  const instrumentGenerators = getInstrumentGenerators(parsed, instrumentID)
  const zones = instrumentGenerators.map(createGeneraterObject)

  if (zones.length === 0) {
    return { allZones: [], filteredZones: [], globalZone: undefined }
  }

  // If the first zone does not have sampleID, it is a global instrument zone.
  let globalZone: ReturnType<typeof createGeneraterObject> | undefined
  const firstInstrumentZone = zones[0]
  if (firstInstrumentZone && firstInstrumentZone.sampleID === undefined) {
    globalZone = zones[0]
  }

  // Return ALL zones, not filtered - we want to see what's happening
  return {
    allZones: zones,
    filteredZones: zones.filter((zone) => zone.sampleID !== undefined),
    globalZone,
  }
}

export function analyzeSF2(data: Uint8Array, filename: string): SF2Diagnostic {
  const parsed = parse(data)
  const zones: ZoneDiagnostic[] = []

  for (let i = 0; i < parsed.presetHeaders.length; i++) {
    const presetHeader = parsed.presetHeaders[i]

    // Skip the terminal preset (EOP)
    if (presetHeader.presetName === "EOP") {
      continue
    }

    const presetGenerators = getPresetGenerators(parsed, i)
    const presetZones = getPresetZones(presetGenerators)

    for (const presetZone of presetZones.zones) {
      const instrumentID = presetZone.instrument

      if (
        instrumentID === undefined ||
        instrumentID < 0 ||
        instrumentID >= parsed.instruments.length
      ) {
        zones.push({
          presetIndex: i,
          presetName: presetHeader.presetName,
          bank: presetHeader.bank,
          preset: presetHeader.preset,
          instrumentID: instrumentID,
          instrumentName: undefined,
          zoneIndex: -1,
          sampleID: undefined,
          isGlobalZone: false,
          generators: presetZone as Record<string, unknown>,
        })
        continue
      }

      const instrumentHeader = parsed.instruments[instrumentID]
      const instrumentZones = getInstrumentZonesRaw(parsed, instrumentID)

      // Log the global zone if it exists
      if (instrumentZones.globalZone) {
        zones.push({
          presetIndex: i,
          presetName: presetHeader.presetName,
          bank: presetHeader.bank,
          preset: presetHeader.preset,
          instrumentID: instrumentID,
          instrumentName: instrumentHeader.instrumentName,
          zoneIndex: -1,
          sampleID: undefined,
          isGlobalZone: true,
          generators: instrumentZones.globalZone as Record<string, unknown>,
        })
      }

      // Log each instrument zone (from allZones to see everything)
      instrumentZones.allZones.forEach((zone, zoneIdx) => {
        // Skip the global zone (already logged)
        if (zoneIdx === 0 && instrumentZones.globalZone) {
          return
        }

        zones.push({
          presetIndex: i,
          presetName: presetHeader.presetName,
          bank: presetHeader.bank,
          preset: presetHeader.preset,
          instrumentID: instrumentID,
          instrumentName: instrumentHeader.instrumentName,
          zoneIndex: zoneIdx,
          sampleID: zone.sampleID,
          isGlobalZone: false,
          generators: zone as Record<string, unknown>,
        })
      })
    }
  }

  const zonesWithUndefinedSampleID = zones.filter(
    (z) => !z.isGlobalZone && z.sampleID === undefined,
  )
  const globalZones = zones.filter((z) => z.isGlobalZone)
  const validZones = zones.filter(
    (z) => !z.isGlobalZone && z.sampleID !== undefined,
  )

  return {
    filename,
    totalPresets: parsed.presetHeaders.length - 1, // Exclude EOP
    totalInstruments: parsed.instruments.length - 1, // Exclude EOI
    totalSamples: parsed.samples.length,
    zones,
    zonesWithUndefinedSampleID,
    summary: {
      totalZones: zones.length,
      globalZones: globalZones.length,
      validZones: validZones.length,
      invalidZones: zonesWithUndefinedSampleID.length,
    },
  }
}

export function compareSF2s(sf2a: SF2Diagnostic, sf2b: SF2Diagnostic): void {
  console.log("\n=== SF2 Comparison ===\n")
  console.log(`${sf2a.filename}:`)
  console.log(
    `  Presets: ${sf2a.totalPresets}, Instruments: ${sf2a.totalInstruments}, Samples: ${sf2a.totalSamples}`,
  )
  console.log(
    `  Zones: ${sf2a.summary.totalZones} (${sf2a.summary.globalZones} global, ${sf2a.summary.validZones} valid, ${sf2a.summary.invalidZones} invalid)`,
  )

  console.log(`\n${sf2b.filename}:`)
  console.log(
    `  Presets: ${sf2b.totalPresets}, Instruments: ${sf2b.totalInstruments}, Samples: ${sf2b.totalSamples}`,
  )
  console.log(
    `  Zones: ${sf2b.summary.totalZones} (${sf2b.summary.globalZones} global, ${sf2b.summary.validZones} valid, ${sf2b.summary.invalidZones} invalid)`,
  )

  if (sf2a.summary.invalidZones > 0) {
    console.log(
      `\n${sf2a.filename} - Invalid zones (non-global with undefined sampleID):`,
    )
    sf2a.zonesWithUndefinedSampleID.forEach((z) => {
      console.log(
        `  Preset ${z.preset} (${z.presetName}) Bank ${z.bank} -> Instrument ${z.instrumentID} (${z.instrumentName})`,
      )
      console.log(`    Generators:`, JSON.stringify(z.generators, null, 2))
    })
  }

  if (sf2b.summary.invalidZones > 0) {
    console.log(
      `\n${sf2b.filename} - Invalid zones (non-global with undefined sampleID):`,
    )
    sf2b.zonesWithUndefinedSampleID.forEach((z) => {
      console.log(
        `  Preset ${z.preset} (${z.presetName}) Bank ${z.bank} -> Instrument ${z.instrumentID} (${z.instrumentName})`,
      )
      console.log(`    Generators:`, JSON.stringify(z.generators, null, 2))
    })
  }
}
