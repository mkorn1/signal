import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import QuantizeSelector from "../Toolbar/QuantizeSelector/QuantizeSelector"

export const TempoGraphQuantizeSelector: FC = observer(() => {
  const { tempoEditorStore } = useStores()
  const { quantize } = tempoEditorStore

  const onSelectQuantize = useCallback(
    (denominator: number) => (tempoEditorStore.quantize = denominator),
    [tempoEditorStore],
  )

  const onClickQuantizeSwitch = useCallback(() => {
    tempoEditorStore.isQuantizeEnabled = !tempoEditorStore.isQuantizeEnabled
  }, [tempoEditorStore])

  return (
    <QuantizeSelector
      value={quantize}
      enabled={true}
      onSelect={onSelectQuantize}
      onClickSwitch={onClickQuantizeSwitch}
    />
  )
})
