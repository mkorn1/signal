import { Theme as BaseTheme } from "../main/common/theme/Theme"

declare module "@emotion/react" {
  export interface Theme extends BaseTheme {}
}
