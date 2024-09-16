import { GLNode, useProjectionMatrix } from "@ryohey/webgl-react"
import { vec4 } from "gl-matrix"
import { FC } from "react"
import { VerticalLinesBuffer, VerticalLinesShader } from "./VerticalLinesShader"

export interface VerticalLinesProps {
  xArray: number[]
  color: vec4
  height: number
  lineWidth: number
  zIndex?: number
}

export const VerticalLines: FC<VerticalLinesProps> = ({
  xArray,
  color,
  height,
  lineWidth,
  zIndex,
}) => {
  const projectionMatrix = useProjectionMatrix()

  return (
    <GLNode
      createShader={VerticalLinesShader}
      createBuffer={(gl) => new VerticalLinesBuffer(gl)}
      uniforms={{
        projectionMatrix,
        color,
        height,
        lineWidth,
      }}
      buffer={xArray}
      zIndex={zIndex}
    />
  )
}
