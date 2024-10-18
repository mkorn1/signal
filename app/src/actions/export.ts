import { renderAudio } from "@signal-app/player"
import { encode } from "wav-encoder"
import { downloadBlob } from "../helpers/Downloader"
import lamejs from "lamejs"
import Song from "../song"
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

export const exportSongAsMp3 =
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

      const mp3Encoder = new lamejs.Mp3Encoder(
        audioBuffer.numberOfChannels,
        audioBuffer.sampleRate,
        128
      )
      const mp3Data: Uint8Array[] = []

      for (let i = 0; i < audioBuffer.length; i += 1152) {
        const left = audioBuffer.getChannelData(0).subarray(i, i + 1152)
        const right = audioBuffer.getChannelData(1).subarray(i, i + 1152)
        const mp3buf = mp3Encoder.encodeBuffer(left, right)
        if (mp3buf.length > 0) {
          mp3Data.push(new Uint8Array(mp3buf))
        }
      }
      const mp3buf = mp3Encoder.flush()
      if (mp3buf.length > 0) {
        mp3Data.push(new Uint8Array(mp3buf))
      }

      const blob = new Blob(mp3Data, { type: "audio/mp3" })
      exportStore.openExportProgressDialog = false
      downloadBlob(blob, "song.mp3")
    } catch (e) {
      console.warn(e)
    }
  }
  ({ exportStore }: RootStore) =>
  () => {
    exportStore.isCanceled = true
  }

export const exportSong =
  (rootStore: RootStore) =>
  () => {
    const { exportMode } = rootStore.exportStore
    if (exportMode === "WAV") {
      exportSongAsWav(rootStore)()
    } else if (exportMode === "MP3") {
      exportSongAsMp3(rootStore)()
    }
  }
  song.allEvents.some((e) => e.tick >= 120)
