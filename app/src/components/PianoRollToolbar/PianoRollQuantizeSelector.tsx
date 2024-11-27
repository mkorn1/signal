import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import QuantizeSelector from "../Toolbar/QuantizeSelector/QuantizeSelector"

export const PianoRollQuantizeSelector: FC = observer(() => {
  const { pianoRollStore } = useStores()
  const { quantize, isQuantizeEnabled } = pianoRollStore

  const onSelectQuantize = useCallback(
    (denominator: number) => {
      pianoRollStore.quantize = denominator
    },
    [pianoRollStore],
  )

  const onClickQuantizeSwitch = useCallback(() => {
    pianoRollStore.isQuantizeEnabled = !pianoRollStore.isQuantizeEnabled
  }, [pianoRollStore])

  return (
    <QuantizeSelector
      value={quantize}
      enabled={isQuantizeEnabled}
      onSelect={onSelectQuantize}
      onClickSwitch={onClickQuantizeSwitch}
    />
  )
})
