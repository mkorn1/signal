import { styled } from "@kuma-ui/core"
import { Metadata } from "next"
import { Footer } from "../../components/Footer/Footer"
import Localized from "../../components/Localized"
import { Navigation } from "../../components/Navigation/Navigation"

export const metadata: Metadata = {
  title: "Support | signal",
}

const Content = styled.div`
  max-width: 60rem;
  width: 100%;
  margin: 0 auto;
`

const Title = styled.h1`
  margin-top: 4rem;
`

const Section = styled.section`
  margin: 3rem 0;
`

const SectionContent = styled.div`
  margin-top: 1rem;
`

const OpenButton = styled.a`
  display: block;
  background: var(--themeColor);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 9999px;
  text-decoration: none;
  font-weight: bold;
  display: inline-block;
  font-size: 1rem;
  border: 2px solid transparent;

  &:hover {
    opacity: 0.9;
  }
`

export default function Page() {
  return (
    <>
      <Navigation />
      <Content>
        <Title>
          <Localized name="support" />
        </Title>

        <Section>
          <h2>
            <Localized name="bug-report" />
          </h2>
          <SectionContent>
            <p>
              <Localized name="bug-report-description" />
            </p>
            <OpenButton href="https://github.com/ryohey/signal/issues">
              <Localized name="open-github-issues" />
            </OpenButton>
          </SectionContent>
        </Section>

        <Section>
          <h2>
            <Localized name="community" />
          </h2>
          <SectionContent>
            <p>
              <Localized name="community-description" />
            </p>
            <OpenButton href="https://discord.com/invite/XQxzNdDJse">
              <Localized name="join-discord" />
            </OpenButton>
          </SectionContent>
        </Section>

        <Section>
          <h2>
            <Localized name="twitter" />
          </h2>
          <SectionContent>
            <p>
              <Localized name="follow-twitter" />
            </p>
            <OpenButton href="https://twitter.com/signalmidi">
              @signalmidi
            </OpenButton>
          </SectionContent>
        </Section>
      </Content>

      <Footer />
    </>
  )
}
