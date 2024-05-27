import { renderAudio } from "@signal-app/player"
import { encode } from "wav-encoder"
import { downloadBlob } from "../../main/common/helpers/Downloader"
import Song from "../../main/common/song"
import RootStore from "../stores/RootStore"

const waitForAnimationFrame = () =>
  new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()))

export const exportSongAsWav =
  ({ song, synth, exportStore }: RootStore) =>
  async () => {
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

export const cancelExport =
  ({ exportStore }: RootStore) =>
  () => {
    exportStore.isCanceled = true
  }

export const canExport = (song: Song) =>
  song.allEvents.some((e) => e.tick >= 120)
