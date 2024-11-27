import { observer } from "mobx-react-lite"
import { FC, useCallback } from "react"
import { useStores } from "../../hooks/useStores"
import QuantizeSelector from "../Toolbar/QuantizeSelector/QuantizeSelector"

export const ArrangeQuantizeSelector: FC = observer(() => {
  const { arrangeViewStore } = useStores()
  const { quantize } = arrangeViewStore

  const onSelectQuantize = useCallback(
    (denominator: number) => (arrangeViewStore.quantize = denominator),
    [arrangeViewStore],
  )

  return (
    <QuantizeSelector
      value={quantize}
      enabled={true}
      onSelect={onSelectQuantize}
      onClickSwitch={() => {}}
    />
  )
})
