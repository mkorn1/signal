import { Theme as BaseTheme } from "../main/theme/Theme"

declare module "@emotion/react" {
  export interface Theme extends BaseTheme {}
}
