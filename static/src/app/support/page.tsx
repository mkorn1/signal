import { Metadata } from "next"
import { Footer } from "../../components/Footer/Footer"
import Localized from "../../components/Localized"
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
        <h1 className={styles.title}>
          <Localized name="support" />
        </h1>

        <section className={styles.section}>
          <h2>
            <Localized name="bug-report" />
          </h2>
          <div className={styles.sectionContent}>
            <p>
              <Localized name="bug-report-description" />
            </p>
            <a
              href="https://github.com/ryohey/signal/issues"
              className={styles.openButton}
            >
              <Localized name="open-github-issues" />
            </a>
          </div>
        </section>

        <section className={styles.section}>
          <h2>
            <Localized name="community" />
          </h2>
          <div className={styles.sectionContent}>
            <p>
              <Localized name="community-description" />
            </p>
            <a
              href="https://discord.com/invite/XQxzNdDJse"
              className={styles.openButton}
            >
              <Localized name="join-discord" />
            </a>
          </div>
        </section>

        <section className={styles.section}>
          <h2>
            <Localized name="twitter" />
          </h2>
          <div className={styles.sectionContent}>
            <p>
              <Localized name="follow-twitter" />
            </p>
            <a
              href="https://twitter.com/signalmidi"
              className={styles.openButton}
            >
              @signalmidi
            </a>
          </div>
        </section>
      </div>

      <Footer />
    </>
  )
}
