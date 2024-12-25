import { useStores } from "../hooks/useStores"

export const usePushHistory = () => {
  const rootStore = useStores()
  return rootStore.pushHistory
}

export const useUndo = () => {
  const rootStore = useStores()

  return () => {
    const currentState = rootStore.serialize()
    const nextState = rootStore.historyStore.undo(currentState)
    if (nextState !== undefined) {
      rootStore.restore(nextState)
    }
  }
}

export const useRedo = () => {
  const rootStore = useStores()

  return () => {
    const currentState = rootStore.serialize()
    const nextState = rootStore.historyStore.redo(currentState)
    if (nextState !== undefined) {
      rootStore.restore(nextState)
    }
  }
}
