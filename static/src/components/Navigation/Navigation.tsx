import githubIcon from "./images/github-icon.svg"
import logoWhite from "./images/logo-white.svg"
import styles from "./Navigation.module.css"

export const Navigation = () => {
  return (
    <header className={styles.header}>
      <div className={styles.content}>
        <a href="/" id="title">
          <img src={logoWhite.src} />
        </a>
        <div className={styles.menu}>
          <a href="/support">Support</a>
          <a href="https://github.com/ryohey/signal/" id="github-link">
            <img src={githubIcon.src} />
          </a>
        </div>
      </div>
    </header>
  )
}
