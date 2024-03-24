import { makeObservable, observable } from "mobx"
import { makePersistable } from "mobx-persist-store"
import { isRunningInElectron } from "../helpers/platform"
import { IndexedDBStorage } from "../services/IndexedDBStorage"
import { SoundFontSynth } from "../services/SoundFontSynth"

const storeName = "soundfonts"

interface LocalSoundFont {
  type: "local"
  data: ArrayBuffer
}

interface RemoteSoundFont {
  type: "remote"
  url: string
}

// electron only feature
interface FileSoundFont {
  type: "file"
  path: string
}

interface Metadata {
  name: string
}

export type SoundFontFile = Metadata & { id: number }

type SoundFontItem = LocalSoundFont | RemoteSoundFont | FileSoundFont

const defaultSoundFonts: (SoundFontItem & Metadata & { id: number })[] =
  isRunningInElectron()
    ? [
        {
          id: -999, // Use negative number to avoid conflict with user saved soundfonts
          type: "file",
          path: "./assets/soundfonts/A320U.sf2",
          name: "A320U.sf2 (Signal Factory Sound)",
        },
      ]
    : [
        {
          id: -999, // Use negative number to avoid conflict with user saved soundfonts
          type: "remote",
          name: "A320U.sf2 (Signal Factory Sound)",
          url: "https://cdn.jsdelivr.net/gh/ryohey/signal@4569a31/public/A320U.sf2",
        },
      ]

export class SoundFontStore {
  private readonly storage: IndexedDBStorage<SoundFontItem, Metadata>
  files: SoundFontFile[] = []
  selectedSoundFontId: number | null = null

  constructor(private readonly synth: SoundFontSynth) {
    makeObservable(this, {
      files: observable,
      selectedSoundFontId: observable,
    })

    this.storage = new IndexedDBStorage("soundfonts", 1)
  }

  async init() {
    await makePersistable(this, {
      name: "SoundFontStore",
      properties: ["selectedSoundFontId"],
      storage: window.localStorage,
    })

    await this.storage.init()
    await this.updateFileList()

    // load last selected soundfont on startup
    await this.load(this.selectedSoundFontId ?? defaultSoundFonts[0].id)
  }

  private async updateFileList() {
    const list = await this.storage.list()
    const savedFiles = Object.keys(list).map((id) => ({
      ...list[Number(id)],
      id: Number(id),
    }))
    this.files = [...defaultSoundFonts, ...savedFiles]
  }

  private async getSoundFont(id: number): Promise<SoundFontItem | null> {
    const defaultSoundFont = defaultSoundFonts.find((f) => f.id === id)
    if (defaultSoundFont !== undefined) {
      return defaultSoundFont
    }
    return await this.storage.load(id)
  }

  async load(id: number) {
    const soundfont = await this.getSoundFont(id)

    if (soundfont === null) {
      throw new Error("SoundFont not found")
    }

    switch (soundfont.type) {
      case "local":
        await this.synth.loadSoundFont(soundfont.data)
        break
      case "remote":
        await this.synth.loadSoundFontFromURL(soundfont.url)
        break
      case "file":
        const data = await window.electronAPI.readFile(soundfont.path)
        await this.synth.loadSoundFont(data)
        break
    }

    this.selectedSoundFontId = id
  }

  async addSoundFont(item: SoundFontItem, metadata: Metadata) {
    await this.storage.save(item, metadata)
    await this.updateFileList()
  }

  async removeSoundFont(id: number) {
    await this.storage.delete(id)
    await this.updateFileList()
  }
}
