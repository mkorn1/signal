import Color from "color"
import { observer } from "mobx-react-lite"
import { FC, useMemo } from "react"
import { Layout } from "../../../Constants"
import { colorToVec4 } from "../../../gl/color"
import { useStores } from "../../../hooks/useStores"
import { useTheme } from "../../../hooks/useTheme"
import { majorScale } from "../../../scale/Scale"
import { HorizontalGrid } from "./HorizontalGrid"

function scaleToConditions(scale: number[], key: number = 0) {
  return new Array(12).fill(false).map((_, i) => scale.includes((i - key) % 12))
}

export const Lines: FC<{ zIndex: number }> = observer(({ zIndex }) => {
  const theme = useTheme()
  const rootStore = useStores()
  const {
    pianoRollStore: {
      scrollTop,
      canvasWidth,
      canvasHeight,
      scaleY,
      keySignature,
    },
  } = rootStore

  const laneColors = useMemo(() => {
    const whiteLaneColor = colorToVec4(
      Color(theme.pianoWhiteKeyLaneColor),
    ) as number[]

    const blackLaneColor = colorToVec4(
      Color(theme.pianoBlackKeyLaneColor),
    ) as number[]

    if (keySignature === null) {
      return Float32Array.from(
        Array(12)
          .fill(0)
          .map((_, i) =>
            majorScale.includes(i) ? whiteLaneColor : blackLaneColor,
          )
          .flat(),
      )
    }

    const highlightedKeys = scaleToConditions(
      keySignature.scale,
      keySignature.key,
    )

    const highlightedLaneColor = colorToVec4(
      Color(theme.pianoHighlightedLaneColor),
    ) as number[]

    const laneColors = highlightedKeys.map((isHighlighted) =>
      isHighlighted ? highlightedLaneColor : whiteLaneColor,
    )

    return Float32Array.from(laneColors.flat())
  }, [theme, keySignature])

  return (
    <HorizontalGrid
      rect={{
        x: 0,
        y: scrollTop,
        width: canvasWidth,
        height: canvasHeight,
      }}
      color={colorToVec4(Color(theme.dividerColor).alpha(0.1))}
      highlightedColor={colorToVec4(Color(theme.dividerColor).alpha(0.6))}
      laneColors={laneColors}
      height={scaleY * Layout.keyHeight}
      zIndex={zIndex}
    />
  )
})
