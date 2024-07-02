import {
  Buffer,
  rectToTriangles,
  Shader,
  uniformFloat,
  uniformMat4,
  uniformVec4,
  VertexArray,
} from "@ryohey/webgl-react"
import isEqual from "lodash/isEqual"
import { Rect } from "../../../../entities/geometry/Rect"

export class HorizontalGridBuffer implements Buffer<Rect, "position"> {
  constructor(readonly vertexArray: VertexArray<"position">) {}

  update(rect: Rect) {
    const positions = rectToTriangles(rect)
    this.vertexArray.updateBuffer("position", new Float32Array(positions))
  }

  get vertexCount() {
    return 6
  }
}

export const HorizontalGridShader = (gl: WebGL2RenderingContext) =>
  new Shader(
    gl,
    `#version 300 es
      precision lowp float;

      uniform mat4 projectionMatrix;
      in vec4 position;
      out vec4 vPosition;

      void main() {
        gl_Position = projectionMatrix * position;
        vPosition = position;
      }
    `,
    `#version 300 es
      precision lowp float;

      uniform vec4 color;
      uniform vec4 highlightedColor;
      uniform vec4 laneColors[12];
      uniform float height;
      
      in vec4 vPosition;

      out vec4 outColor;

      float line(float inputY, float y, float lineWidth) {
       return step(y, inputY) * step(inputY, y + lineWidth);
      }
      
      void main() {
        float screenHeight = height * 128.0;
        float modY = mod(screenHeight - vPosition.y, height * 12.0);
        float laneHeight = height - 1.0;

        // draw lane colors
        for (int i = 0; i < 12; i++) {
          outColor += line(modY, height * float(i) + 1.0, laneHeight) * laneColors[i];
        }

        // draw lines
        outColor += (
          line(modY, height * 1.0, 1.0) +
          line(modY, height * 2.0, 1.0) +
          line(modY, height * 3.0, 1.0) +
          line(modY, height * 4.0, 1.0) +
          line(modY, height * 6.0, 1.0) +
          line(modY, height * 7.0, 1.0) +
          line(modY, height * 8.0, 1.0) +
          line(modY, height * 9.0, 1.0) +
          line(modY, height * 10.0, 1.0) +
          line(modY, height * 11.0, 1.0)
        ) * color;

        // draw hihglighted lines for key 0 and 5
        outColor += (
          line(modY, 0.0, 1.0) +
          line(modY, height * 5.0, 1.0)
        ) * highlightedColor;
      }
    `,
    {
      position: { size: 2, type: gl.FLOAT },
    },
    {
      projectionMatrix: uniformMat4(),
      color: uniformVec4(),
      highlightedColor: uniformVec4(),
      laneColors: {
        initialValue: new Float32Array(4 * 12),
        isEqual,
        upload: (gl, loc, value) => gl.uniform4fv(loc, value, 0, 4 * 12),
      },
      height: uniformFloat(),
    },
  )
