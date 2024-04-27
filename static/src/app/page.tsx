import { Metadata } from "next"
import dynamic from "next/dynamic"
import { Navigation } from "../components/Navigation/Navigation"
import generalMidiLogo from "./images/general-midi-logo.svg"
import githubIcon from "./images/github-icon.svg"
import chartIcon from "./images/iconmonstr-chart-21.svg"
import favoriteIcon from "./images/iconmonstr-favorite-4.svg"
import twitterIcon from "./images/iconmonstr-twitter-1.svg"
import midiLogo from "./images/midi-logo.svg"
import pwaLogo from "./images/pwa-logo.svg"
import screenshot from "./images/screenshot.png"
import synthesizerKeyboard from "./images/synthesizer-keyboard-svgrepo-com.svg"
import wavFile from "./images/wav-file.svg"

const Localized = dynamic(() => import("../components/Localized"), {
  ssr: false,
})

export const metadata: Metadata = {
  title: "signal - Online MIDI Editor",
  description: "Fully Open-sourced Online MIDI Editor",
}

export default function Home() {
  return (
    <>
      <Navigation />

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
    </>
  )
}
