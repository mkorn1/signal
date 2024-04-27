import { Metadata } from "next"
import { Navigation } from "../../components/Navigation/Navigation"
import styles from "./styles.module.css"

export const metadata: Metadata = {
  title: "Support | signal",
}

export default function Page() {
  return (
    <>
      <Navigation />
      <div className={styles.content}>
        <h1 className={styles.title}>Support</h1>

        <section className={styles.section}>
          <h2>Bug Report / Feature Request</h2>
          <div className={styles.sectionContent}>
            <p>
              バグ報告や新機能のリクエストは、GitHubのIssueで受け付けています
            </p>
            <a
              href="https://github.com/ryohey/signal/issues"
              className={styles.openButton}
            >
              Open GitHub Issues
            </a>
          </div>
        </section>

        <section className={styles.section}>
          <h2>Community</h2>
          <div className={styles.sectionContent}>
            <p>
              Discord
              コミュニティで使い方について質問したり、作った曲をシェアしたりしてください
            </p>
            <a
              href="https://discord.com/invite/XQxzNdDJse"
              className={styles.openButton}
            >
              Join Discord
            </a>
          </div>
        </section>
      </div>
    </>
  )
}
