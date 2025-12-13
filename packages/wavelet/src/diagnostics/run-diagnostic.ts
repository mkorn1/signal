import * as fs from "fs"
import * as path from "path"
import { analyzeSF2, compareSF2s } from "./sf2-analyzer"

async function main() {
  const args = process.argv.slice(2)

  if (args.length < 1) {
    console.log(
      "Usage: npx tsx run-diagnostic.ts <sf2-file> [comparison-sf2-file]",
    )
    process.exit(1)
  }

  const file1 = args[0]
  const data1 = new Uint8Array(fs.readFileSync(file1))
  const diag1 = analyzeSF2(data1, path.basename(file1))

  console.log("\n=== SF2 Analysis ===\n")
  console.log(`File: ${diag1.filename}`)
  console.log(`Presets: ${diag1.totalPresets}`)
  console.log(`Instruments: ${diag1.totalInstruments}`)
  console.log(`Samples: ${diag1.totalSamples}`)
  console.log(`\nZone Summary:`)
  console.log(`  Total: ${diag1.summary.totalZones}`)
  console.log(`  Global: ${diag1.summary.globalZones}`)
  console.log(`  Valid: ${diag1.summary.validZones}`)
  console.log(`  Invalid: ${diag1.summary.invalidZones}`)

  if (diag1.summary.invalidZones > 0) {
    console.log(`\nInvalid zones (non-global with undefined sampleID):`)
    diag1.zonesWithUndefinedSampleID.slice(0, 20).forEach((z) => {
      console.log(`  Preset ${z.preset} "${z.presetName}" Bank ${z.bank}`)
      console.log(`    Instrument: ${z.instrumentID} "${z.instrumentName}"`)
      console.log(`    Zone ${z.zoneIndex}: sampleID=${z.sampleID}`)
    })
    if (diag1.zonesWithUndefinedSampleID.length > 20) {
      console.log(
        `  ... and ${diag1.zonesWithUndefinedSampleID.length - 20} more`,
      )
    }
  }

  if (args.length >= 2) {
    const file2 = args[1]
    const data2 = new Uint8Array(fs.readFileSync(file2))
    const diag2 = analyzeSF2(data2, path.basename(file2))
    compareSF2s(diag1, diag2)
  }

  // Write full diagnostic to JSON file
  const outputPath = file1.replace(".sf2", "-diagnostic.json")
  fs.writeFileSync(outputPath, JSON.stringify(diag1, null, 2))
  console.log(`\nFull diagnostic written to: ${outputPath}`)
}

main().catch(console.error)
