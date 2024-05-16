import { FC } from "react"
import { Localized } from "../../common/localize/useLocalization"
import { RecentSongList } from "../components/RecentSongList"
import { PageLayout, PageTitle } from "../layouts/PageLayout"

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
