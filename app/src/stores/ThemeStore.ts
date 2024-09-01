import { makeObservable, observable } from "mobx"
import { makePersistable } from "mobx-persist-store"
import { ThemeType } from "../theme/Theme"

export class ThemeStore {
  themeType: ThemeType = "dark"

  constructor() {
    makeObservable(this, {
      themeType: observable,
    })

    makePersistable(this, {
      name: "ThemeStore",
      properties: ["themeType"],
      storage: window.localStorage,
    })
  }
}
