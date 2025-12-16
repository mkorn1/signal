import { atom, useAtomValue, useSetAtom } from "jotai"

export function useRootView() {
  return {
    get openSettingDialog() {
      return useAtomValue(openSettingDialogAtom)
    },
    get openControlSettingDialog() {
      return useAtomValue(openControlSettingDialogAtom)
    },
    get openHelpDialog() {
      return useAtomValue(openHelpDialogAtom)
    },
    get initializeError() {
      return useAtomValue(initializeErrorAtom)
    },
    get openInitializeErrorDialog() {
      return useAtomValue(openInitializeErrorDialogAtom)
    },
    setOpenSettingDialog: useSetAtom(openSettingDialogAtom),
    setOpenControlSettingDialog: useSetAtom(openControlSettingDialogAtom),
    setOpenHelpDialog: useSetAtom(openHelpDialogAtom),
    setInitializeError: useSetAtom(initializeErrorAtom),
    setOpenInitializeErrorDialog: useSetAtom(openInitializeErrorDialogAtom),
  }
}

// atoms
const openSettingDialogAtom = atom<boolean>(false)
const openControlSettingDialogAtom = atom<boolean>(false)
const openHelpDialogAtom = atom<boolean>(false)
const initializeErrorAtom = atom<Error | null>(null)
const openInitializeErrorDialogAtom = atom<boolean>(false)
