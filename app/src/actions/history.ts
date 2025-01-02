import { useStores } from "../hooks/useStores"

export const usePushHistory = () => {
  const { historyStore } = useStores()
  return () => historyStore.push()
}

export const useUndo = () => {
  const { historyStore } = useStores()
  return () => historyStore.undo()
}

export const useRedo = () => {
  const { historyStore } = useStores()
  return () => historyStore.redo()
}
