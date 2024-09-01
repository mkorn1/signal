export interface Theme {
  font: string
  canvasFont: string
  themeColor: string
  onSurfaceColor: string // content color on themeColor
  darkBackgroundColor: string
  backgroundColor: string
  secondaryBackgroundColor: string
  editorBackgroundColor: string // control pane / arrange view / tempo editor
  editorGridColor: string
  editorSecondaryGridColor: string
  dividerColor: string
  popupBorderColor: string
  textColor: string
  secondaryTextColor: string
  tertiaryTextColor: string
  pianoKeyBlack: string
  pianoKeyWhite: string
  pianoWhiteKeyLaneColor: string
  pianoBlackKeyLaneColor: string
  pianoHighlightedLaneColor: string
  ghostNoteColor: string
  recordColor: string
  shadowColor: string
  highlightColor: string
  greenColor: string
  redColor: string
  yellowColor: string
}

export const defaultTheme: Theme = {
  font: "Inter, -apple-system, BlinkMacSystemFont, Avenir, Lato",
  canvasFont: "Arial",
  themeColor: "hsl(230, 70%, 55%)",
  onSurfaceColor: "#ffffff",
  textColor: "#ffffff",
  secondaryTextColor: "hsl(223, 12%, 60%)",
  tertiaryTextColor: "#5a6173",
  dividerColor: "hsl(224, 12%, 24%)",
  popupBorderColor: "#000000",
  darkBackgroundColor: "hsl(228, 10%, 13%)",
  backgroundColor: "hsl(228, 10%, 16%)",
  secondaryBackgroundColor: "hsl(227, 10%, 22%)",
  editorBackgroundColor: "hsl(228, 10%, 13%)",
  editorSecondaryGridColor: "hsl(224, 12%, 19%)",
  editorGridColor: "hsl(224, 12%, 26%)",
  pianoKeyBlack: "#272a36",
  pianoKeyWhite: "#fbfcff",
  pianoWhiteKeyLaneColor: "hsl(228, 10%, 16%)",
  pianoBlackKeyLaneColor: "hsl(228, 10%, 13%)",
  pianoHighlightedLaneColor: "hsl(230, 23%, 20%)",
  ghostNoteColor: "#444444",
  recordColor: "#dd3c3c",
  shadowColor: "rgba(0, 0, 0, 0.1)",
  highlightColor: "#8388a51a",
  greenColor: "#31DE53",
  redColor: "#DE5267",
  yellowColor: "#DEB126",
}
