import {
  defaultInstrumentZone,
  GeneratorParams,
  getPresetGenerators,
  parse,
} from "@signal-app/sf2parser"
import { AmplitudeEnvelopeParameter } from "../types/AmplitudeEnvelopeParameter"
import {
  LoadSampleEvent,
  SampleLoop,
  SampleParameter,
  SampleParameterEvent,
  SampleRange,
} from "../SynthEvent"
import { getInstrumentZones } from "./getInstrumentZones"
import { getPresetZones } from "./getPresetZones"

export interface BufferCreator {
  createBuffer(
    numberOfChannels: number,
    length: number,
    sampleRate: number,
  ): AudioBuffer
}

const parseSamplesFromSoundFont = (data: Uint8Array) => {
  const parsed = parse(data)
  const result: { parameter: SampleParameter; range: SampleRange }[] = []
  const convertedSampleBuffers: { [key: number]: Float32Array } = {}

  function addSampleIfNeeded(sampleID: number) {
    const cached = convertedSampleBuffers[sampleID]
    if (cached) {
      return cached
    }

    // Bounds check for sampleID
    if (sampleID < 0 || sampleID >= parsed.samples.length) {
      console.warn(
        `Cannot load sample ${sampleID}: index out of bounds (valid range: 0-${parsed.samples.length - 1})`,
      )
      return new Float32Array(0)
    }

    const sample = parsed.samples[sampleID]
    const audioData = new Float32Array(sample.length)
    for (let i = 0; i < sample.length; i++) {
      audioData[i] = sample[i] / 32767
    }

    convertedSampleBuffers[sampleID] = audioData
    return audioData
  }

  for (let i = 0; i < parsed.presetHeaders.length; i++) {
    const presetHeader = parsed.presetHeaders[i]
    const presetGenerators = getPresetGenerators(parsed, i)

    const presetZones = getPresetZones(presetGenerators)

    for (const presetZone of presetZones.zones) {
      const presetGen = {
        ...removeUndefined(presetZones.globalZone ?? {}),
        ...removeUndefined(presetZone),
      }

      const instrumentID = presetZone.instrument

      // Bounds check for instrumentID before accessing instruments array
      if (
        instrumentID === undefined ||
        instrumentID < 0 ||
        instrumentID >= parsed.instruments.length
      ) {
        console.warn(
          `Skipping preset zone with invalid instrumentID ${instrumentID} (valid range: 0-${parsed.instruments.length - 1})`,
        )
        continue
      }

      const instrumentZones = getInstrumentZones(parsed, instrumentID)

      for (const zone of instrumentZones.zones) {
        const sampleID = zone.sampleID!

        // Bounds check for sampleID before accessing sampleHeaders and samples
        if (
          sampleID < 0 ||
          sampleID >= parsed.sampleHeaders.length ||
          sampleID >= parsed.samples.length
        ) {
          console.warn(
            `Skipping zone with invalid sampleID ${sampleID} (valid range: 0-${Math.min(parsed.sampleHeaders.length, parsed.samples.length) - 1})`,
          )
          continue
        }

        const sampleHeader = parsed.sampleHeaders[sampleID]

        const { velRange: defaultVelRange, ...generatorDefault } =
          defaultInstrumentZone

        const gen = {
          ...generatorDefault,
          ...removeUndefined(instrumentZones.globalZone ?? {}),
          ...removeUndefined(zone),
        }

        // inherit preset's velRange
        gen.velRange = gen.velRange ?? presetGen.velRange ?? defaultVelRange

        // add presetGenerator value
        for (const key of Object.keys(gen)) {
          if (
            key in presetGen &&
            typeof (gen as any)[key] === "number" &&
            typeof (presetGen as any)[key] === "number"
          ) {
            ;(gen as any)[key] += (presetGen as any)[key]
          }
        }

        const tune = gen.coarseTune + gen.fineTune / 100

        const basePitch =
          tune +
          sampleHeader.pitchCorrection / 100 -
          (gen.overridingRootKey ?? sampleHeader.originalPitch)

        const sampleStart =
          gen.startAddrsCoarseOffset * 32768 + gen.startAddrsOffset

        const sampleEnd = gen.endAddrsCoarseOffset * 32768 + gen.endAddrsOffset

        const loopStart =
          sampleHeader.loopStart +
          gen.startloopAddrsCoarseOffset * 32768 +
          gen.startloopAddrsOffset

        const loopEnd =
          sampleHeader.loopEnd +
          gen.endloopAddrsCoarseOffset * 32768 +
          gen.endloopAddrsOffset

        const audioData = addSampleIfNeeded(sampleID)

        const amplitudeEnvelope: AmplitudeEnvelopeParameter = {
          attackTime: timeCentToSec(gen.attackVolEnv),
          holdTime: timeCentToSec(gen.holdVolEnv),
          decayTime: timeCentToSec(gen.decayVolEnv),
          sustainLevel: 1 / centibelToLinear(gen.sustainVolEnv),
          releaseTime: timeCentToSec(gen.releaseVolEnv),
        }

        const loop: SampleLoop = (() => {
          switch (gen.sampleModes) {
            case 0:
              // no_loop
              break
            case 1:
              if (loopEnd > 0) {
                return {
                  type: "loop_continuous",
                  start: loopStart,
                  end: loopEnd,
                }
              }
            case 3:
              if (loopEnd > 0) {
                return {
                  type: "loop_sustain",
                  start: loopStart,
                  end: loopEnd,
                }
              }
              break
          }
          // fallback as no_loop
          return { type: "no_loop" }
        })()

        const parameter: SampleParameter = {
          sampleID: sampleID,
          pitch: -basePitch,
          name: sampleHeader.sampleName,
          sampleStart,
          sampleEnd: sampleEnd === 0 ? audioData.length : sampleEnd,
          loop,
          sampleRate: sampleHeader.sampleRate,
          amplitudeEnvelope,
          scaleTuning: gen.scaleTuning / 100,
          pan: (gen.pan ?? 0) / 500,
          exclusiveClass: gen.exclusiveClass,
          volume: centibelToLinear(-gen.initialAttenuation),
        }

        const range: SampleRange = {
          instrument: presetHeader.preset,
          bank: presetHeader.bank,
          keyRange: [gen.keyRange.lo, gen.keyRange.hi],
          velRange: [gen.velRange.lo, gen.velRange.hi],
        }

        result.push({ parameter, range })
      }
    }
  }

  return {
    parameters: result,
    samples: convertedSampleBuffers,
  }
}

export const getSampleEventsFromSoundFont = (
  data: Uint8Array,
): {
  event: LoadSampleEvent | SampleParameterEvent
  transfer?: Transferable[]
}[] => {
  const { samples, parameters } = parseSamplesFromSoundFont(data)

  const loadSampleEvents: LoadSampleEvent[] = Object.entries(samples).map(
    ([key, value]) => ({
      type: "loadSample",
      sampleID: Number(key),
      data: value.buffer,
    }),
  )

  const sampleParameterEvents: SampleParameterEvent[] = parameters.map(
    ({ parameter, range }) => ({ type: "sampleParameter", parameter, range }),
  )

  return [
    ...loadSampleEvents.map((event) => ({ event, transfer: [event.data] })),
    ...sampleParameterEvents.map((event) => ({ event })),
  ]
}

function convertTime(value: number) {
  return Math.pow(2, value / 1200)
}

function timeCentToSec(value: number) {
  if (value <= -32768) {
    return 0
  }

  if (value < -12000) {
    value = -12000
  }

  if (value > 8000) {
    value = 8000
  }

  return convertTime(value)
}

function centibelToLinear(value: number) {
  return Math.pow(10, value / 200)
}

function removeUndefined<T>(obj: T) {
  const result: Partial<T> = {}
  for (let key in obj) {
    if (obj[key] !== undefined) {
      result[key] = obj[key]
    }
  }
  return result
}
