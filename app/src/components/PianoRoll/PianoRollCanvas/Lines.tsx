import Color from "color"
import { observer } from "mobx-react-lite"
import { FC, useMemo } from "react"
import { Layout } from "../../../Constants"
import { colorToVec4 } from "../../../gl/color"
import { useStores } from "../../../hooks/useStores"
import { useTheme } from "../../../hooks/useTheme"
import { HorizontalGrid } from "./HorizontalGrid"

type Scale = number[]

const majorScale = [0, 2, 4, 5, 7, 9, 11]
const minorScale = [0, 2, 3, 5, 7, 8, 10]
const harmonicMinorScale = [0, 2, 3, 5, 7, 8, 11]
const melodicMinorScale = [0, 2, 3, 5, 7, 9, 11]
const pentatonicMajorScale = [0, 2, 4, 7, 9]
const pentatonicMinorScale = [0, 3, 5, 7, 10]
const bluesScale = [0, 3, 5, 6, 7, 10]
const wholeToneScale = [0, 2, 4, 6, 8, 10]
const chromaticScale = Array.from({ length: 12 }, (_, i) => i)
const diminishedScale = [0, 1, 3, 4, 6, 7, 9, 10]
const augmentedScale = [0, 3, 4, 7, 8, 11]
const majorPentatonicScale = [0, 2, 4, 7, 9]
const minorPentatonicScale = [0, 3, 5, 7, 10]
const majorBluesScale = [0, 2, 3, 4, 7, 9]

function scaleToConditions(scale: number[], key: number = 0) {
  return new Array(12).fill(false).map((_, i) => scale.includes((i - key) % 12))
}

export const Lines: FC<{ zIndex: number }> = observer(({ zIndex }) => {
  const theme = useTheme()
  const rootStore = useStores()
  const {
    pianoRollStore: { scrollTop, canvasWidth, canvasHeight, scaleY },
  } = rootStore

  const scale: Scale | null = majorScale

  const laneColors = useMemo(() => {
    const whiteLaneColor = colorToVec4(
      Color(theme.pianoWhiteKeyLaneColor),
    ) as number[]

    const blackLaneColor = colorToVec4(
      Color(theme.pianoBlackKeyLaneColor),
    ) as number[]

    if (scale === null) {
      return Float32Array.from(
        Array(12)
          .fill(0)
          .map((_, i) =>
            majorScale.includes(i) ? whiteLaneColor : blackLaneColor,
          )
          .flat(),
      )
    }

    const highlightedKeys = scaleToConditions(scale, 0)

    const highlightedLaneColor = colorToVec4(
      Color(theme.pianoHighlightedLaneColor),
    ) as number[]

    const laneColors = highlightedKeys.map((isHighlighted) =>
      isHighlighted ? highlightedLaneColor : whiteLaneColor,
    )

    return Float32Array.from(laneColors.flat())
  }, [theme, scale])

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
