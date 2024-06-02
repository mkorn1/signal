import { styled } from "@kuma-ui/core"
import { Metadata } from "next"
import Localized from "../components/Localized"
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
import twitterCard from "./images/twitter-card.png"
import wavFile from "./images/wav-file.svg"

export const metadata: Metadata = {
  title: "signal - Online MIDI Editor",
  description: "Fully Open-sourced Online MIDI Editor",
  openGraph: {
    title: "signal",
    description: "Fully Open-sourced Online MIDI Editor",
    images: [
      {
        url: twitterCard.src,
        width: 1200,
        height: 630,
        alt: "signal",
      },
    ],
  },
  twitter: {
    card: "summary",
  },
}

const Hero = styled.section`
  background: var(--themeColor);
  color: white;
  overflow: hidden;

  > div {
    display: flex;
  }
  .text h1 {
    font-weight: 800;
    font-size: 3.2rem;
    line-height: 1;
    margin: 0 0 1rem 0;
  }
  .image {
    width: 35%;
    padding-left: 2rem;
  }
  .image img {
    width: 34rem;
    border-radius: 0.5rem;
    box-shadow: 0 1rem 8rem #0000004f;
  }
  .text .description {
    font-size: 1.4rem;
    line-height: 1;
  }

  @media screen and (max-width: 896px) {
    .content {
      flex-direction: column;
    }

    .text h1 {
      font-size: 2.5rem;
    }

    .image {
      padding: 0;
      width: 100%;
    }

    .image img {
      margin-top: 3rem;
      width: 100%;
    }
  }
`

const Platform = styled.p`
  opacity: 0.8;
  font-size: 0.9rem;
`

const SectionContent = styled.div`
  padding: 5rem 0;
  max-width: 60rem;
  width: 100%;
  margin: 0 auto;

  @media screen and (max-width: 896px) {
    width: 90%;
  }
`

const SectionTitle = styled.h3`
  font-size: 1.5rem;
`

const LaunchButton = styled.a`
  background: white;
  padding: 0.8rem 2rem;
  border-radius: 9999px;
  text-decoration: none;
  color: var(--themeColor);
  font-weight: bold;
  display: inline-block;
  margin: 2rem 0 0 0;
  font-size: 1.2rem;
  border: 2px solid transparent;
  transition:
    border-color 0.2s ease,
    background 0.2s ease,
    color 0.2s ease;
  box-shadow: 0 1rem 7rem #0000004f;

  &:hover {
    border-color: white;
    background: transparent;
    color: white;
  }
`

const SponsorSection = styled.section`
  background: #1f1f23;
`

const SponsorIntro = styled.div`
  width: 40em;
  line-height: 1.9;
  margin-bottom: 1.5em;
  max-width: 100%;
`

const SponsorButton = styled.a`
  display: inline-flex;
  align-items: center;
  font-weight: 600;
  background: #bb5d9029;
  border-radius: 0.3em;
  padding: 1em 1.5em;
  color: white;
  text-decoration: none;
  border: 2px solid transparent;
  transition: 0.2s ease;

  &:hover {
    border-color: #bb5d90;
  }

  img {
    margin-right: 0.7em;
    width: 1.2em;
  }
`

const Features = styled.section`
  .content {
    display: flex;
  }

  .content .left {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
  }

  .content .right {
    flex-shrink: 0;
    min-width: 342px;
  }

  @media screen and (max-width: 896px) {
    .content {
      flex-direction: column-reverse;
    }
  }
`

const Feature = styled.div`
  background: #ffffff0d;
  padding: 2rem;
  border-radius: 1rem;
  margin-right: 2rem;
  margin-bottom: 2rem;
  box-shadow: 0 1rem 3rem 0 #0000000e;

  .icon {
    display: inline-flex;
    width: 4rem;
    border-radius: 2rem;
    height: 4rem;
    overflow: hidden;
    background: white;
    align-items: center;
    justify-content: center;
  }

  .title {
    font-size: 1.1rem;
    font-weight: bold;
    margin-top: 1rem;
    margin-bottom: 1rem;
  }

  .description {
    opacity: 0.5;
  }

  .icon img {
    width: 2rem;
  }

  @media screen and (max-width: 896px) {
    margin-right: 0;
  }
`

const ExternalLink = styled.a`
  font-size: 0.9rem;
  display: inline-flex;
  border: #3b3b44 2px solid;
  border-radius: 0.3em;
  padding: 0.5em 1em;
  color: white;
  text-decoration: none;
  align-items: center;

  img {
    margin-right: 0.7em;
  }
`

const SupportRow = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 1rem;

  span {
    padding-left: 1em;
    font-size: 0.9rem;
    opacity: 0.6;
  }
`

export default function Home() {
  return (
    <>
      <Navigation />

      <Hero>
        <SectionContent>
          <div className="text">
            <h1>
              <Localized name="app-intro" />
            </h1>
            <p className="description">
              <Localized name="app-desc" />
            </p>
            <LaunchButton href="edit">
              <Localized name="launch" />
            </LaunchButton>
            <Platform>
              <Localized name="platform" />
            </Platform>
          </div>
          <div className="image">
            <img src={screenshot.src} alt="Screenshot" />
          </div>
        </SectionContent>
      </Hero>

      <Features>
        <SectionContent className="content">
          <div className="left">
            <SectionTitle>
              <Localized name="features" />
            </SectionTitle>

            <Feature style={{ background: "#3c2fd740" }}>
              <div className="icon" style={{ background: "#3c2fd7" }}>
                <img src={midiLogo.src} style={{ width: "3rem" }} />
              </div>
              <div className="title">
                <Localized name="feature-midi-file" />
              </div>
              <div className="description">
                <Localized name="feature-midi-file-description" />
              </div>
            </Feature>

            <Feature style={{ background: "#91322c73" }}>
              <div className="icon" style={{ background: "#e7372c" }}>
                <img src={generalMidiLogo.src} />
              </div>
              <div className="title">
                <Localized name="feature-gm-module" />
              </div>
              <div className="description">
                <Localized name="feature-gm-module-description" />
              </div>
            </Feature>

            <Feature style={{ background: "#ff99002f" }}>
              <div className="icon" style={{ background: "#ff9900" }}>
                <img src={wavFile.src} />
              </div>
              <div className="title">
                <Localized name="feature-export-audio" />
              </div>
              <div className="description">
                <Localized name="feature-export-audio-description" />
              </div>
            </Feature>

            <Feature style={{ background: "#249f9f2f" }}>
              <div className="icon" style={{ background: "#249f9f" }}>
                <img src={synthesizerKeyboard.src} />
              </div>
              <div className="title">
                <Localized name="feature-midi-io" />
              </div>
              <div className="description">
                <Localized name="feature-midi-io-description" />
              </div>
            </Feature>

            <Feature style={{ background: "#3d2fd727" }}>
              <div className="icon" style={{ background: "#3c2fd7" }}>
                <img src={chartIcon.src} />
              </div>
              <div className="title">
                <Localized name="feature-time-signature" />
              </div>
              <div className="description">
                <Localized name="feature-time-signature-description" />
              </div>
            </Feature>

            <Feature style={{ background: "#ff99002f" }}>
              <div className="icon" style={{ background: "#ff9900" }}>
                <img src={pwaLogo.src} />
              </div>
              <div className="title">
                <Localized name="feature-pwa" />
              </div>
              <div className="description">
                <Localized name="feature-pwa-description" />
              </div>
            </Feature>
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
        </SectionContent>
      </Features>

      <SponsorSection>
        <SectionContent>
          <SectionTitle>
            <Localized name="become-sponsor" />
          </SectionTitle>
          <SponsorIntro>
            <Localized name="sponsor-intro" />
          </SponsorIntro>

          <SponsorButton href="https://github.com/sponsors/ryohey">
            <img src={favoriteIcon.src} />
            <span>
              <Localized name="open-github-sponsors" />
            </span>
          </SponsorButton>
        </SectionContent>
      </SponsorSection>

      <section id="support">
        <SectionContent>
          <SectionTitle>
            <Localized name="support" />
          </SectionTitle>

          <SupportRow>
            <ExternalLink href="https://twitter.com/signalmidi">
              <img src={twitterIcon.src} />
              @signalmidi
            </ExternalLink>
            <span>
              <Localized name="follow-twitter" />
            </span>
          </SupportRow>

          <SupportRow>
            <ExternalLink href="https://github.com/ryohey/signal">
              <img src={githubIcon.src} />
              ryohey/signal
            </ExternalLink>
            <span>
              <Localized name="support-github-desctiption" />
            </span>
          </SupportRow>
        </SectionContent>
      </section>
    </>
  )
}
