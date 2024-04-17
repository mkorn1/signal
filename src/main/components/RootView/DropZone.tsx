import styled from "@emotion/styled"
import { observer } from "mobx-react-lite"
import { FC, PropsWithChildren, useCallback } from "react"
import { localized } from "../../../common/localize/localizedString"
import { setSong } from "../../actions"
import { songFromArrayBuffer } from "../../actions/file"
import { useStores } from "../../hooks/useStores"

const Container = styled.div`
  height: 100%;
  display: flex;
  flex-direction: column;
  flex-grow: 1;
  overflow: hidden;
`

export const DropZone: FC<PropsWithChildren<{}>> = observer(({ children }) => {
  const rootStore = useStores()
  const { song } = rootStore

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      if (file.type !== "audio/midi") {
        return
      }
      const reader = new FileReader()
      reader.onload = () => {
        const data = reader.result
        if (data) {
          if (
            song.isSaved ||
            confirm(
              localized("confirm-open", "Are you sure you want to continue?"),
            )
          ) {
            const newSong = songFromArrayBuffer(
              data as ArrayBuffer,
              file.name,
              file.path,
            )
            setSong(rootStore)(newSong)
          }
        }
      }
      reader.readAsArrayBuffer(file)
    },
    [song, rootStore],
  )

  return (
    <Container onDrop={onDrop} onDragOver={onDragOver}>
      {children}
    </Container>
  )
})
