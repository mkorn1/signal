import Color from "color"
import React, { FC, useCallback, useState } from "react"
import { noteNameWithOctString } from "../../../main/common/helpers/noteNumberString"
import {
  noteOffMidiEvent,
  noteOnMidiEvent,
} from "../../../main/common/midi/MidiEvent"
import { Theme } from "../../../main/common/theme/Theme"
import { Layout } from "../../Constants"
import { observeDrag } from "../../helpers/observeDrag"
import { useStores } from "../../hooks/useStores"
import { useTheme } from "../../hooks/useTheme"
import DrawCanvas from "../DrawCanvas"

// 0: white, 1: black
const Colors = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]

function makeBlackKeyFillStyle(
  ctx: CanvasRenderingContext2D,
  width: number,
): CanvasFillStrokeStyles["fillStyle"] {
  const grd = ctx.createLinearGradient(0, 0, width, 0)
  grd.addColorStop(0.0, "rgba(33, 33, 33, 1.000)")
  grd.addColorStop(0.895, "rgba(96, 93, 93, 1.000)")
  grd.addColorStop(0.924, "rgba(48, 48, 48, 1.000)")
  grd.addColorStop(1.0, "rgba(0, 0, 0, 1.000)")
  return grd
}

function drawBorder(
  ctx: CanvasRenderingContext2D,
  width: number,
  dividerColor: string,
): void {
  ctx.lineWidth = 1
  ctx.strokeStyle = dividerColor
  ctx.beginPath()
  ctx.moveTo(0, 0)
  ctx.lineTo(width, 0)
  ctx.closePath()
  ctx.stroke()
}

function drawWhiteKey(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  theme: Theme,
  isSelected: boolean,
  bordered: boolean,
): void {
  if (isSelected) {
    ctx.fillStyle = theme.themeColor
    ctx.fillRect(0, 0.5, width, height)
  }
  if (bordered) {
    drawBorder(ctx, width, theme.dividerColor)
  }
}

function drawBlackKey(
  ctx: CanvasRenderingContext2D,
  keyWidth: number,
  width: number,
  height: number,
  fillStyle: CanvasFillStrokeStyles["fillStyle"],
  dividerColor: string,
): void {
  const middle = Math.round(height / 2)

  ctx.fillStyle = fillStyle
  ctx.fillRect(0, 0.5, keyWidth, height)

  ctx.lineWidth = 1
  ctx.strokeStyle = dividerColor
  ctx.beginPath()
  ctx.moveTo(keyWidth, middle)
  ctx.lineTo(width, middle)
  ctx.closePath()
  ctx.stroke()
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  keyNum: number,
  font: string,
  color: string,
) {
  const x = width - 5
  ctx.textAlign = "right"
  ctx.textBaseline = "middle"
  ctx.font = `12px ${font}`
  ctx.fillStyle = color
  ctx.fillText(noteNameWithOctString(keyNum), x, height / 2)
}

function drawKeys(
  ctx: CanvasRenderingContext2D,
  blackKeyWidth: number,
  width: number,
  keyHeight: number,
  numberOfKeys: number,
  theme: Theme,
  touchingKeys: number[],
) {
  ctx.save()
  ctx.translate(0, 0.5)

  ctx.fillStyle = theme.pianoKeyWhite
  ctx.fillRect(0, 0, width, keyHeight * numberOfKeys)

  const allKeys = [...Array(numberOfKeys).keys()]
  const whiteKeys = allKeys.filter((i) => Colors[i % Colors.length] === 0)
  const blackKeys = allKeys.filter((i) => Colors[i % Colors.length] === 1)
  const labelKeys = allKeys.filter((i) => i % 12 === 0)

  drawBorder(ctx, width, theme.dividerColor)

  // Draw white keys
  for (const keyNum of whiteKeys) {
    let y = (numberOfKeys - keyNum - 1) * keyHeight
    const isSelected = touchingKeys.includes(keyNum)

    const bordered = keyNum % 12 === 4 || keyNum % 12 === 11
    const prevKeyBlack = Colors[(keyNum - 1) % Colors.length] === 1
    const nextKeyBlack = Colors[(keyNum + 1) % Colors.length] === 1

    let height = keyHeight
    if (prevKeyBlack) {
      height += 0.5 * keyHeight
    }
    if (nextKeyBlack) {
      y -= 0.5 * keyHeight
      height += 0.5 * keyHeight
    }

    ctx.save()
    ctx.translate(0, y)

    drawWhiteKey(ctx, width, height, theme, isSelected, bordered)
    ctx.restore()
  }

  // Draw black keys
  for (const keyNum of blackKeys) {
    const y = (numberOfKeys - keyNum - 1) * keyHeight
    const isSelected = touchingKeys.includes(keyNum)

    const blackKeyFillStyle = makeBlackKeyFillStyle(ctx, blackKeyWidth)
    const grayDividerColor = Color(theme.dividerColor).alpha(0.3).string()

    ctx.save()
    ctx.translate(0, y)

    drawBlackKey(
      ctx,
      blackKeyWidth,
      width,
      keyHeight,
      isSelected ? theme.themeColor : blackKeyFillStyle,
      grayDividerColor,
    )
    ctx.restore()
  }

  // Draw labels
  for (const keyNum of labelKeys) {
    const y = (numberOfKeys - keyNum - 1) * keyHeight
    ctx.save()
    ctx.translate(0, y)

    drawLabel(
      ctx,
      width,
      keyHeight,
      keyNum,
      theme.canvasFont,
      theme.secondaryTextColor,
    )
    ctx.restore()
  }

  ctx.restore()
}

export interface PianoKeysProps {
  numberOfKeys: number
  keyHeight: number
}

const PianoKeys: FC<PianoKeysProps> = ({ numberOfKeys, keyHeight }) => {
  const theme = useTheme()
  const rootStore = useStores()
  const width = Layout.keyWidth
  const blackKeyWidth = Layout.keyWidth * Layout.blackKeyWidthRatio
  const [touchingKeys, setTouchingKeys] = useState<number[]>([])

  const draw = useCallback(
    (ctx: CanvasRenderingContext2D) => {
      drawKeys(
        ctx,
        blackKeyWidth,
        width,
        keyHeight,
        numberOfKeys,
        theme,
        touchingKeys,
      )
    },
    [keyHeight, numberOfKeys, theme, touchingKeys],
  )

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      function posToNoteNumber(x: number, y: number): number {
        const noteNumberFloat = numberOfKeys - y / keyHeight
        const noteNumber = Math.floor(noteNumberFloat)
        const isBlack = Colors[noteNumber % Colors.length] !== 0

        // If you are on the white tab below a black key, round to the nearest
        // white note.
        if (x > blackKeyWidth && isBlack) {
          const touchingNextWhite =
            Math.round(noteNumberFloat) - Math.floor(noteNumberFloat)
          return touchingNextWhite ? noteNumber + 1 : noteNumber - 1
        }
        return noteNumber
      }

      const startPosition = {
        x: e.nativeEvent.offsetX,
        y: e.nativeEvent.offsetY,
      }
      const {
        player,
        pianoRollStore: { selectedTrack },
      } = rootStore
      const channel = selectedTrack?.channel ?? 0

      let prevNoteNumber = posToNoteNumber(startPosition.x, startPosition.y)
      player.sendEvent(noteOnMidiEvent(0, channel, prevNoteNumber, 127))

      setTouchingKeys([prevNoteNumber])

      observeDrag({
        onMouseMove(e) {
          const pos = {
            x: e.offsetX,
            y: e.offsetY,
          }
          const noteNumber = posToNoteNumber(pos.x, pos.y)
          if (noteNumber !== prevNoteNumber) {
            player.sendEvent(noteOffMidiEvent(0, channel, prevNoteNumber, 0))
            player.sendEvent(noteOnMidiEvent(0, channel, noteNumber, 127))
            prevNoteNumber = noteNumber
            setTouchingKeys([noteNumber])
          }
        },
        onMouseUp(_) {
          player.sendEvent(noteOffMidiEvent(0, channel, prevNoteNumber, 0))
          setTouchingKeys([])
        },
      })
    },
    [numberOfKeys, keyHeight],
  )

  return (
    <DrawCanvas
      draw={draw}
      width={width}
      height={keyHeight * numberOfKeys}
      onMouseDown={onMouseDown}
    />
  )
}

function equals(props: PianoKeysProps, nextProps: PianoKeysProps) {
  return (
    props.keyHeight === nextProps.keyHeight &&
    props.numberOfKeys === nextProps.numberOfKeys
  )
}

export default React.memo(PianoKeys, equals)
