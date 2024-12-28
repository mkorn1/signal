interface Serializable<T> {
  serialize(): T
  restore(serialized: T): void
}

export default class HistoryStore<State> {
  undoHistory: State[] = []
  redoHistory: State[] = []

  constructor(private readonly serializable: Serializable<State>) {}

  push() {
    const currentState = this.serializable.serialize()
    this.undoHistory.push(currentState)
    this.redoHistory = []
  }

  undo() {
    const currentState = this.serializable.serialize()
    const state = this.undoHistory.pop()
    if (state) {
      this.redoHistory.push(currentState)
      this.serializable.restore(state)
    }
  }

  redo() {
    const currentState = this.serializable.serialize()
    const state = this.redoHistory.pop()
    if (state) {
      this.undoHistory.push(currentState)
      this.serializable.restore(state)
    }
  }

  clear() {
    this.undoHistory = []
    this.redoHistory = []
  }
}
