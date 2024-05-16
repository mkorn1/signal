import { observer } from "mobx-react-lite"
import { FC } from "react"
import { Localized } from "../../../common/localize/useLocalization"
import Track from "../../../common/track/Track"

// Display the track number if there is no name track name for display
export const TrackName: FC<{ track: Track }> = observer(({ track }) => {
  if (track.name && track.name.length > 0) {
    return <>{track.name}</>
  }
  if (track.channel === undefined) {
    return (
      <>
        <Localized name="conductor-track" />
      </>
    )
  }
  return (
    <>
      <Localized name="track" /> {track.channel + 1}
    </>
  )
})
