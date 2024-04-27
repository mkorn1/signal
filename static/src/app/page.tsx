import { Metadata } from "next"
import { Localized } from "../l18n/useLocalization"
import generalMidiLogo from "./images/general-midi-logo.svg"
import githubIcon from "./images/github-icon.svg"
import chartIcon from "./images/iconmonstr-chart-21.svg"
import favoriteIcon from "./images/iconmonstr-favorite-4.svg"
import twitterIcon from "./images/iconmonstr-twitter-1.svg"
import logoWhite from "./images/logo-white.svg"
import midiLogo from "./images/midi-logo.svg"
import pwaLogo from "./images/pwa-logo.svg"
import screenshot from "./images/screenshot.png"
import synthesizerKeyboard from "./images/synthesizer-keyboard-svgrepo-com.svg"
import wavFile from "./images/wav-file.svg"

export const metadata: Metadata = {
  title: "signal - Online MIDI Editor",
  description: "Fully Open-sourced Online MIDI Editor",
}

export default function Home() {
  return (
    <main>
      <header>
        <div className="content">
          <a href="/" id="title">
            <img src={logoWhite.src} />
          </a>
          <a href="https://github.com/ryohey/signal/" id="github-link">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
            >
              <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
            </svg>
          </a>
        </div>
      </header>

      <section id="hero">
        <div className="content">
          <div className="text">
            <h1>
              <Localized name="app-intro" />
            </h1>
            <p className="description">
              <Localized name="app-desc" />
            </p>
            <a href="edit" id="launch-button">
              <Localized name="launch" />
            </a>
            <p className="platform">
              <Localized name="platform" />
            </p>
          </div>
          <div className="image">
            <img src={screenshot.src} alt="Screenshot" />
          </div>
        </div>
      </section>

      <section id="features">
        <div className="content">
          <div className="left">
            <h3>
              <Localized name="features" />
            </h3>

            <div className="feature" style={{ background: "#3c2fd740" }}>
              <div className="icon" style={{ background: "#3c2fd7" }}>
                <img src={midiLogo.src} style={{ width: "3rem" }} />
              </div>
              <div className="title">
                <Localized name="feature-midi-file" />
              </div>
              <div className="description">
                <Localized name="feature-midi-file-description" />
              </div>
            </div>

            <div className="feature" style={{ background: "#91322c73" }}>
              <div className="icon" style={{ background: "#e7372c" }}>
                <img src={generalMidiLogo.src} />
              </div>
              <div className="title">
                <Localized name="feature-gm-module" />
              </div>
              <div className="description">
                <Localized name="feature-gm-module-description" />
              </div>
            </div>

            <div className="feature" style={{ background: "#ff99002f" }}>
              <div className="icon" style={{ background: "#ff9900" }}>
                <img src={wavFile.src} />
              </div>
              <div className="title">
                <Localized name="feature-export-audio" />
              </div>
              <div className="description">
                <Localized name="feature-export-audio-description" />
              </div>
            </div>

            <div className="feature" style={{ background: "#249f9f2f" }}>
              <div className="icon" style={{ background: "#249f9f" }}>
                <img src={synthesizerKeyboard.src} />
              </div>
              <div className="title">
                <Localized name="feature-midi-io" />
              </div>
              <div className="description">
                <Localized name="feature-midi-io-description" />
              </div>
            </div>

            <div className="feature" style={{ background: "#3d2fd727" }}>
              <div className="icon" style={{ background: "#3c2fd7" }}>
                <img src={chartIcon.src} />
              </div>
              <div className="title">
                <Localized name="feature-time-signature" />
              </div>
              <div className="description">
                <Localized name="feature-time-signature-description" />
              </div>
            </div>

            <div className="feature" style={{ background: "#ff99002f" }}>
              <div className="icon" style={{ background: "#ff9900" }}>
                <img src={pwaLogo.src} />
              </div>
              <div className="title">
                <Localized name="feature-pwa" />
              </div>
              <div className="description">
                <Localized name="feature-pwa-description" />
              </div>
            </div>
          </div>

          <div
            className="right"
            dangerouslySetInnerHTML={{
              __html: `
                <script
                  async
                  type="text/javascript"
                  src="https://cdn.carbonads.com/carbon.js?serve=CESI52QY&placement=signalvercelapp"
                  id="_carbonads_js"
                ></script>
            `,
            }}
          />
        </div>
      </section>

      <section id="sponsor">
        <div className="content">
          <h3>
            <Localized name="become-sponsor" />
          </h3>
          <p className="sponsor-intro">
            <Localized name="sponsor-intro" />
          </p>

          <a
            className="sponsor-button"
            href="https://github.com/sponsors/ryohey"
          >
            <img src={favoriteIcon.src} />
            <span>
              <Localized name="open-github-sponsors" />
            </span>
          </a>
        </div>
      </section>

      <section id="support">
        <div className="content">
          <h3>
            <Localized name="support" />
          </h3>

          <div className="support-row">
            <a className="external-link" href="https://twitter.com/signalmidi">
              <img src={twitterIcon.src} />
              @signalmidi
            </a>
            <span>
              <Localized name="follow-twitter" />
            </span>
          </div>

          <div className="support-row">
            <a
              className="external-link"
              href="https://github.com/ryohey/signal"
            >
              <img src={githubIcon.src} />
              ryohey/signal
            </a>
            <span>
              <Localized name="support-github-desctiption" />
            </span>
          </div>
        </div>
      </section>
    </main>
  )
}
