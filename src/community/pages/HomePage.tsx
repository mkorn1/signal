import { FC } from "react"
import { RecentSongList } from "../components/RecentSongList"
import { PageLayout, PageTitle } from "../layouts/PageLayout"
import { Localized } from "../localize/useLocalization"

export const HomePage: FC = () => {
  return (
    <PageLayout>
      <PageTitle>
        <Localized name="recent-tracks" />
      </PageTitle>
      <RecentSongList />
    </PageLayout>
  )
}
