import { styled } from "@kuma-ui/core"
import { Metadata } from "next"
import Localized from "../../components/Localized"
import { Navigation } from "../../components/Navigation/Navigation"
import { LocaleDate } from "./LocaleDate"

export const metadata: Metadata = {
  title: "Privacy Policy | signal",
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

export default function Page() {
  return (
    <>
      <Navigation />
      <Content>
        <Title>
          <Localized name="privacy-policy-title" />
        </Title>
        <p>
          <Localized name="privacy-policy-description" />
        </p>

        <Section>
          <h2>
            <Localized name="privacy-data-title" />
          </h2>
          <SectionContent>
            <p>
              <Localized name="privacy-data-description" />
            </p>
          </SectionContent>
        </Section>

        <Section>
          <h2>
            <Localized name="privacy-analytics-title" />
          </h2>
          <SectionContent>
            <p>
              <Localized name="privacy-analytics-description" />
            </p>
          </SectionContent>
        </Section>

        <p>
          <Localized name="privacy-updated" />
          <LocaleDate date={new Date("2024/05/03")} />
        </p>
      </Content>
    </>
  )
}
