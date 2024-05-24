import {
  createCloudSongDataRepository,
  createCloudSongRepository,
  createUserRepository,
} from "@signal-app/api"
import Player from "../../common/player"
import { auth, firestore } from "../../firebase/firebase"
import { SoundFontSynth } from "../../main/services/SoundFontSynth"
import { AuthStore } from "../../main/stores/AuthStore"
import { EventSource } from "../services/EventSource"
import { CommunitySongStore } from "./CommunitySongStore"
import RootViewStore from "./RootViewStore"
import { SongStore } from "./SongStore"

export default class RootStore {
  readonly userRepository = createUserRepository(firestore, auth)
  readonly cloudSongRepository = createCloudSongRepository(firestore, auth)
  readonly cloudSongDataRepository = createCloudSongDataRepository(
    firestore,
    auth,
  )
  readonly songStore = new SongStore(this.cloudSongDataRepository)
  readonly authStore = new AuthStore(this.userRepository)
  readonly communitySongStore = new CommunitySongStore()
  readonly rootViewStore = new RootViewStore()
  readonly player: Player
  readonly synth: SoundFontSynth

  constructor() {
    const context = new (window.AudioContext || window.webkitAudioContext)()

    this.synth = new SoundFontSynth(context)

    const dummySynth = {
      activate() {},
      sendEvent() {},
    }

    const dummyTrackMute = {
      shouldPlayTrack: () => true,
    }

    const eventSource = new EventSource(this.songStore)

    this.player = new Player(
      this.synth,
      dummySynth,
      dummyTrackMute,
      eventSource,
    )
  }
}
