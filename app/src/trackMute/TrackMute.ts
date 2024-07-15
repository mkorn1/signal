import { makeObservable, observable } from "mobx"
import { ITrackMute } from "./ITrackMute"

function updated<T>(obj: T, key: keyof T, value: any) {
  return { ...obj, [key]: value }
}

type BoolMap = { [index: number]: boolean }

/**

  操作によって二つのモードが切り替わる

  ## mute モード

  単に mute/unmute でトラックの出力を OFF/ON にする
  solo とは独立してミュート設定を保持する

  ## solo モード

  何かのトラックを solo にした時にこのモードに遷移する
  指定トラック以外の全てのトラックを mute するが、
  追加で他のトラックを solo にしたときは
  そのトラックの mute を解除する (mute モードのミュート設定とは独立)

  すべてのトラックの solo が解除された時に
  mute モードに遷移する

*/
export default class TrackMute implements ITrackMute {
  private mutes: BoolMap = {}

  private solos: BoolMap = {}

  constructor() {
    makeObservable<TrackMute, "mutes" | "solos">(this, {
      mutes: observable,
      solos: observable,
    })
  }

  reset() {
    this.mutes = {}
    this.solos = {}
  }

  private _setMute(trackIndex: number, isMute: boolean) {
    if (this.isSoloMode()) {
      return
    }
    this.mutes = updated(this.mutes, trackIndex, isMute)
  }

  private _getMute(trackIndex: number) {
    return this.mutes[trackIndex] || false
  }

  private _setSolo(trackIndex: number, isSolo: boolean) {
    this.solos = updated(this.solos, trackIndex, isSolo)
  }

  private _getSolo(trackIndex: number) {
    return this.solos[trackIndex] || false
  }

  mute(trackIndex: number) {
    this._setMute(trackIndex, true)
  }

  unmute(trackIndex: number) {
    this._setMute(trackIndex, false)
  }

  solo(trackIndex: number) {
    this._setSolo(trackIndex, true)
  }

  unsolo(trackIndex: number) {
    this._setSolo(trackIndex, false)
  }

  isSoloMode(): boolean {
    // どれかひとつでも solo なら solo モード
    // Any one or Solo mode Solo mode
    return Object.values(this.solos).some((s) => s)
  }

  shouldPlayTrack(trackIndex: number) {
    if (this.isSoloMode()) {
      return this._getSolo(trackIndex)
    } else {
      return !this._getMute(trackIndex)
    }
  }

  // 表示用のメソッド
  // Method for display

  isSolo(trackIndex: number) {
    return this.isSoloMode() && this.solos[trackIndex]
  }

  isMuted(trackIndex: number) {
    return !this.shouldPlayTrack(trackIndex)
  }
}
