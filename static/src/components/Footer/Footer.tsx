import dynamic from "next/dynamic"
import styles from "./Footer.module.css"

const Localized = dynamic(() => import("../../components/Localized"), {
  ssr: false,
})

export const Footer = () => {
  return (
    <footer className={styles.footer}>
      <div className={styles.content}>
        <a href="/privacy">
          <Localized name="privacy-policy-title" />
        </a>
      </div>
    </footer>
  )
}
