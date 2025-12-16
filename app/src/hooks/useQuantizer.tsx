import { Measure } from "@signal-app/core"
import { atom, useAtomValue, useSetAtom } from "jotai"
import { useCallback, useMemo } from "react"
import { useStores } from "./useStores"

// Global quantizer atoms - shared across all views
const quantizeAtom = atom(8)
const isEnabledAtom = atom(true)

function useQuantizeCalc(fn: (tick: number) => number) {
  const { songStore } = useStores()
  const quantize = useAtomValue(quantizeAtom)

  return useCallback(
    (tick: number) => {
      const measureStart = Measure.getMeasureStart(
        songStore.song.measures,
        tick,
        songStore.song.timebase,
      )
      const beats = quantize === 1 ? (measureStart.numerator ?? 4) : 4
      const u = (songStore.song.timebase * beats) / quantize
      const offset = measureStart?.tick ?? 0
      return fn((tick - offset) / u) * u + offset
    },
    [songStore, quantize, fn],
  )
}

export function useQuantizer() {
  return {
    get quantize() {
      return useAtomValue(quantizeAtom)
    },
    get quantizeUnit() {
      const { songStore } = useStores()
      const quantize = useAtomValue(quantizeAtom)
      return useMemo(
        () => (songStore.song.timebase * 4) / quantize,
        [songStore, quantize],
      )
    },
    get isQuantizeEnabled() {
      return useAtomValue(isEnabledAtom)
    },
    get quantizeRound() {
      const isEnabled = useAtomValue(isEnabledAtom)
      const calc = useQuantizeCalc(Math.round)
      return isEnabled ? calc : Math.round
    },
    get quantizeFloor() {
      const isEnabled = useAtomValue(isEnabledAtom)
      const calc = useQuantizeCalc(Math.floor)
      return isEnabled ? calc : Math.floor
    },
    get quantizeCeil() {
      const isEnabled = useAtomValue(isEnabledAtom)
      const calc = useQuantizeCalc(Math.ceil)
      return isEnabled ? calc : Math.ceil
    },
    get forceQuantizeRound() {
      return useQuantizeCalc(Math.round)
    },
    setQuantize: useSetAtom(quantizeAtom),
    setIsQuantizeEnabled: useSetAtom(isEnabledAtom),
  }
}
