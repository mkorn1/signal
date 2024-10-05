import { styled } from "@kuma-ui/core"
import { FC } from "react"
import { Feature } from "./Feature"
import chartIcon from "./images/iconmonstr-chart-21.svg"
import midiLogo from "./images/midi-logo.svg"
import pwaLogo from "./images/pwa-logo.svg"
import soundFontIcon from "./images/soundfont-icon.svg"
import synthesizerKeyboard from "./images/synthesizer-keyboard-svgrepo-com.svg"
import wavFile from "./images/wav-file.svg"

const Container = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(20rem, 1fr));
  gap: 2rem;
`

export const FeatureList: FC = () => {
  return (
    <Container>
      <Feature
        backgroundColor="#3c2fd740"
        iconBackgroundColor="#3c2fd7"
        iconSrc={midiLogo.src}
        iconWidth={"3rem"}
        titleKey="feature-midi-file"
        descriptionKey="feature-midi-file-description"
      />
      <Feature
        backgroundColor="#91322c73"
        iconBackgroundColor="#e7372c"
        iconSrc={soundFontIcon.src}
        titleKey="feature-soundfont"
        descriptionKey="feature-soundfont-description"
      />
      <Feature
        backgroundColor="#ff99002f"
        iconBackgroundColor="#ff9900"
        iconSrc={wavFile.src}
        iconWidth={"1.8rem"}
        titleKey="feature-export-audio"
        descriptionKey="feature-export-audio-description"
      />
      <Feature
        backgroundColor="#249f9f2f"
        iconBackgroundColor="#249f9f"
        iconSrc={synthesizerKeyboard.src}
        titleKey="feature-midi-io"
        descriptionKey="feature-midi-io-description"
      />
      <Feature
        backgroundColor="#3d2fd727"
        iconBackgroundColor="#3c2fd7"
        iconSrc={chartIcon.src}
        iconWidth={"1.6rem"}
        titleKey="feature-time-signature"
        descriptionKey="feature-time-signature-description"
      />
      <Feature
        backgroundColor="#ff99002f"
        iconBackgroundColor="#ff9900"
        iconSrc={pwaLogo.src}
        titleKey="feature-pwa"
        descriptionKey="feature-pwa-description"
      />
    </Container>
  )
}
