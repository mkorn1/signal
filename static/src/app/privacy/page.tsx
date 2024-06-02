import { Metadata } from "next"
import Localized from "../../components/Localized"
import { Navigation } from "../../components/Navigation/Navigation"
import { LocaleDate } from "./LocaleDate"
import styles from "./styles.module.css"

export const metadata: Metadata = {
  title: "Privacy Policy | signal",
}

export default function Page() {
  return (
    <>
      <Navigation />
      <div className={styles.content}>
        <h1 className={styles.title}>
          <Localized name="privacy-policy-title" />
        </h1>
        <p>
          <Localized name="privacy-policy-description" />
        </p>

        <section className={styles.section}>
          <h2>
            <Localized name="privacy-data-title" />
          </h2>
          <div className={styles.sectionContent}>
            <p>
              <Localized name="privacy-data-description" />
            </p>
          </div>
        </section>

        <section className={styles.section}>
          <h2>
            <Localized name="privacy-analytics-title" />
          </h2>
          <div className={styles.sectionContent}>
            <p>
              <Localized name="privacy-analytics-description" />
            </p>
          </div>
        </section>

        <p>
          <Localized name="privacy-updated" />
          <LocaleDate date={new Date("2024/05/03")} />
        </p>
      </div>
    </>
  )
}
