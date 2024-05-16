import styled from "@emotion/styled"
import { observer } from "mobx-react-lite"
import { FC, PropsWithChildren, useCallback } from "react"
import { useLocalization } from "../../../common/localize/useLocalization"
import { setSong } from "../../actions"
import { songFromFile } from "../../actions/file"
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
  const localized = useLocalization()

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
  }, [])

  const onDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      if (file.type !== "audio/midi") {
        return
      }
      if (song.isSaved || confirm(localized["confirm-open"])) {
        const newSong = await songFromFile(file)
        setSong(rootStore)(newSong)
      }
    },
    [song, rootStore],
  )

  return (
    <Container onDrop={onDrop} onDragOver={onDragOver}>
      {children}
    </Container>
  )
})
