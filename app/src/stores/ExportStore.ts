import { makeObservable, observable } from "mobx"

export class ExportStore {
  openExportProgressDialog = false
  progress = 0
  isCanceled = false

  constructor() {
    makeObservable(this, {
      openExportProgressDialog: observable,
      progress: observable,
    })
  }
}
