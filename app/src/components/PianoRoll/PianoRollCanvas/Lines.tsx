import Color from "color"
import { observer } from "mobx-react-lite"
import { FC, useMemo } from "react"
import { Layout } from "../../../Constants"
import { colorToVec4 } from "../../../gl/color"
import { useStores } from "../../../hooks/useStores"
import { useTheme } from "../../../hooks/useTheme"
import { HorizontalGrid } from "./HorizontalGrid"

export const Lines: FC<{ zIndex: number }> = observer(({ zIndex }) => {
  const theme = useTheme()
  const rootStore = useStores()
  const {
    pianoRollStore: { scrollTop, canvasWidth, canvasHeight, scaleY },
  } = rootStore
  const whiteLaneColor = [0, 0, 0, 0]
  const blackLaneColor = colorToVec4(
    Color(theme.pianoBlackKeyLaneColor),
  ) as number[]
  const laneColors = useMemo(
    () =>
      Float32Array.from(
        [
          whiteLaneColor,
          blackLaneColor,
          whiteLaneColor,
          blackLaneColor,
          whiteLaneColor,
          whiteLaneColor,
          blackLaneColor,
          whiteLaneColor,
          blackLaneColor,
          whiteLaneColor,
          blackLaneColor,
          whiteLaneColor,
        ].flat(),
      ),
    [blackLaneColor, whiteLaneColor],
  )

  return (
    <HorizontalGrid
      rect={{
        x: 0,
        y: scrollTop,
        width: canvasWidth,
        height: canvasHeight,
      }}
      color={colorToVec4(Color(theme.dividerColor).alpha(0.2))}
      highlightedColor={colorToVec4(Color(theme.dividerColor).alpha(0.5))}
      laneColors={laneColors}
      height={scaleY * Layout.keyHeight}
      zIndex={zIndex}
    />
  )
})
