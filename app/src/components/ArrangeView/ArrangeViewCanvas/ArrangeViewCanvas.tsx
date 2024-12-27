import { useTheme } from "@emotion/react"
import { GLCanvas, Transform } from "@ryohey/webgl-react"
import { observer } from "mobx-react-lite"
import { FC, useCallback, useMemo } from "react"
import { Point } from "../../../entities/geometry/Point"
import { matrixFromTranslation } from "../../../helpers/matrix"
import { observeDrag } from "../../../helpers/observeDrag"
import { AbstractMouseEvent } from "../../../hooks/useContextMenu"
import { useStores } from "../../../hooks/useStores"
import { Beats } from "../../GLNodes/Beats"
import { Cursor } from "../../GLNodes/Cursor"
import { Selection } from "../../GLNodes/Selection"
import { Lines } from "./Lines"
import { Notes } from "./Notes"
import { useSelectionGesture } from "./gestures/useSelectionGesture"

export interface ArrangeViewCanvasProps {
  width: number
  onContextMenu: (e: AbstractMouseEvent) => void
}

export const ArrangeViewCanvas: FC<ArrangeViewCanvasProps> = observer(
  ({ width, onContextMenu }) => {
    const { arrangeViewStore } = useStores()
    const theme = useTheme()
    const {
      scrollLeft,
      scrollTop,
      contentHeight: height,
      rulerStore: { beats },
      cursorX,
      selectionRect,
    } = arrangeViewStore

    const selectionGesture = useSelectionGesture()

    const scrollXMatrix = useMemo(
      () => matrixFromTranslation(-scrollLeft, 0),
      [scrollLeft],
    )

    const scrollYMatrix = useMemo(
      () => matrixFromTranslation(0, -scrollTop),
      [scrollLeft, scrollTop],
    )

    const scrollXYMatrix = useMemo(
      () => matrixFromTranslation(-scrollLeft, -scrollTop),
      [scrollLeft, scrollTop],
    )

    const setScrollLeft = useCallback((v: number) => {
      arrangeViewStore.setScrollLeftInPixels(v)
      arrangeViewStore.autoScroll = false
    }, [])

    const setScrollTop = useCallback(
      (v: number) => arrangeViewStore.setScrollTop(v),
      [],
    )
    const handleMiddleClick = useCallback(
      (e: React.MouseEvent) => {
        function createPoint(e: MouseEvent) {
          return { x: e.clientX, y: e.clientY }
        }
        const startPos = createPoint(e.nativeEvent)

        observeDrag({
          onMouseMove(e) {
            const pos = createPoint(e)
            const delta = Point.sub(pos, startPos)
            setScrollLeft(Math.max(0, scrollLeft - delta.x))
            setScrollTop(Math.max(0, scrollTop - delta.y))
          },
        })
      },
      [scrollLeft, scrollTop],
    )

    const onMouseDown = useCallback(
      (e: React.MouseEvent) => {
        switch (e.button) {
          case 0:
            selectionGesture.onMouseDown(e)
            break
          case 1:
            handleMiddleClick(e)
            break
          case 2:
            onContextMenu(e)
            break
          default:
            break
        }
      },
      [selectionGesture, handleMiddleClick, onContextMenu],
    )

    return (
      <GLCanvas
        width={width}
        height={height}
        onMouseDown={onMouseDown}
        onContextMenu={useCallback((e: any) => e.preventDefault(), [])}
        style={{ backgroundColor: theme.editorBackgroundColor }}
      >
        <Transform matrix={scrollYMatrix}>
          <Lines width={width} zIndex={0} />
        </Transform>
        <Transform matrix={scrollXMatrix}>
          <Beats height={height} beats={beats} zIndex={1} />
          <Cursor x={cursorX} height={height} zIndex={4} />
        </Transform>
        <Transform matrix={scrollXYMatrix}>
          <Notes zIndex={2} />
          <Selection rect={selectionRect} zIndex={3} />
        </Transform>
      </GLCanvas>
    )
  },
)
