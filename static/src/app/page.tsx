import { styled } from "@kuma-ui/core"
import { Metadata } from "next"
import { FeatureList } from "../components/FeatureList/FeatureList"
import Localized from "../components/Localized"
import { Navigation } from "../components/Navigation/Navigation"
import githubIcon from "./images/github-icon.svg"
import favoriteIcon from "./images/iconmonstr-favorite-4.svg"
import twitterIcon from "./images/iconmonstr-twitter-1.svg"
import macAppStore from "./images/mac-app-store.svg"
import screenshot from "./images/screenshot.png"
import twitterCard from "./images/twitter-card.png"

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
  background: #1f1f23;
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

const Platform = styled.div`
  color: var(--secondaryTextColor);
  font-size: 0.9rem;
`

const SectionContent = styled.div`
  padding: 5rem 0;
  max-width: 60rem;
  width: 90%;
  margin: 0 auto;
`

const SectionTitle = styled.h3`
  font-size: 1.5rem;
`

const LaunchWrapper = styled.div`
  display: flex;
  margin-top: 3rem;
  align-items: center;
  gap: 1rem;
`

const LaunchButton = styled.a`
  background: var(--themeColor);
  padding: 1rem 3rem;
  border-radius: 9999px;
  text-decoration: none;
  color: var(--textColor);
  font-weight: bold;
  display: inline-block;
  font-size: 1.2rem;
  border: 2px solid transparent;
  transition:
    border-color 0.2s ease,
    background 0.2s ease,
    color 0.2s ease;
  box-shadow: 0 1rem 7rem #0000004f;
  flex-shrink: 0;

  &:hover {
    background: color-mix(in srgb, var(--themeColor), white 20%);
  }
`

const StoreIntro = styled.p`
  color: var(--secondaryTextColor);
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
        <SectionContent className="content">
          <div className="text">
            <h1>
              <Localized name="app-intro" />
            </h1>
            <p className="description">
              <Localized name="app-desc" />
            </p>
            <LaunchWrapper>
              <LaunchButton href="edit">
                <Localized name="launch" />
              </LaunchButton>
              <Platform>
                <pre>
                  <Localized name="platform" />
                </pre>
              </Platform>
            </LaunchWrapper>
            <div style={{ marginTop: "4rem" }}>
              <StoreIntro>
                <Localized name="store-intro" />
              </StoreIntro>
              <a
                href="https://apps.apple.com/us/app/signal-midi-editor/id6499489458"
                target="_blank"
              >
                <img
                  src={macAppStore.src}
                  alt="Mac App Store"
                  style={{ width: "12rem" }}
                />
              </a>
            </div>
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
            <FeatureList />
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
