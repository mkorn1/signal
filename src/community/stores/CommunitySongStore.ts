import { makeObservable, observable } from "mobx"
import { CloudSong } from "signal-api"

export class CommunitySongStore {
  songs: CloudSong[] = []

  constructor() {
    makeObservable(this, {
      songs: observable,
    })
  }
}
