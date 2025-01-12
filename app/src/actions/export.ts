import { renderAudio } from "@signal-app/player"
import { downloadBlob } from "../helpers/Downloader"
import { encodeMp3, encodeWAV } from "../helpers/encodeAudio"
import { useStores } from "../hooks/useStores"
import Song from "../song"

const waitForAnimationFrame = () =>
  new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()))

export const useExportSong = () => {
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

      const encoder = getEncoder(exportStore.exportMode)
      const audioData = await encoder.encode(audioBuffer)

      const blob = new Blob([audioData], { type: encoder.mimeType })
      exportStore.openExportProgressDialog = false
      downloadBlob(blob, "song." + encoder.ext)
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

const getEncoder = (mode: "WAV" | "MP3") => {
  switch (mode) {
    case "WAV":
      return {
        encode: encodeWAV,
        ext: "wav",
        mimeType: "audio/wav",
      }
    case "MP3":
      return {
        encode: encodeMp3,
        ext: "mp3",
        mimeType: "audio/mp3",
      }
  }
}
