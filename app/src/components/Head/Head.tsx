import { observer } from "mobx-react-lite"
import { FC } from "react"
import { Helmet } from "react-helmet-async"
import { useStores } from "../../hooks/useStores"
import Song from "../../song"

export const Head: FC = observer(() => {
  const { song } = useStores()

  return (
    <Helmet>
      <title>
        {getSongDisplayName(song)}
        {song.isSaved ? "" : " *"}
        {" - signal"}
      </title>
    </Helmet>
  )
})

function getSongDisplayName(song: Song): string {
  if (song.filepath.length > 0) {
    return song.filepath
  }
  if (song.name.length > 0) {
    return song.name
  }
  return "New song"
}