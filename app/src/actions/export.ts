import { renderAudio } from "@signal-app/player"
import { encode } from "wav-encoder"
import { downloadBlob } from "../helpers/Downloader"
import { useStores } from "../hooks/useStores"
import Song from "../song"

const waitForAnimationFrame = () =>
  new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()))

export const useExportSongAsWav = () => {
  const { song, synth, exportStore } = useStores()

  return async () => {
    const soundFontData = synth.loadedSoundFontData
    if (soundFontData === null) {
      return
    }

    const sampleRate = 44100

    exportStore.isCanceled = false
    exportStore.openExportProgressDialog = true
    exportStore.progress = 0

    try {
      const audioBuffer = await renderAudio(
        soundFontData,
        song.allEvents,
        song.timebase,
        sampleRate,
        {
          bufferSize: 128,
          cancel: () => exportStore.isCanceled,
          waitForEventLoop: waitForAnimationFrame,
          onProgress: (numFrames, totalFrames) =>
            (exportStore.progress = numFrames / totalFrames),
        },
      )

      exportStore.progress = 1

      const wavData = await encode({
        sampleRate: audioBuffer.sampleRate,
        channelData: [
          audioBuffer.getChannelData(0),
          audioBuffer.getChannelData(1),
        ],
      })

      const blob = new Blob([wavData], { type: "audio/wav" })
      exportStore.openExportProgressDialog = false
      downloadBlob(blob, "song.wav")
    } catch (e) {
      console.warn(e)
    }
  }
}

export const useCancelExport = () => {
  const { exportStore } = useStores()

  return () => {
    exportStore.isCanceled = true
  }
}

export const canExport = (song: Song) =>
  song.allEvents.some((e) => e.tick >= 120)

