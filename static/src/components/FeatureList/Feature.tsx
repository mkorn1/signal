import { styled } from "@kuma-ui/core"
import { FC } from "react"
import { LocalizationKey } from "../../l18n/useLocalization"
import Localized from "../Localized"

const FeatureWrapper = styled.div`
  background: #ffffff0d;
  padding: 2rem;
  border-radius: 1rem;
  box-shadow: 0 1rem 3rem 0 #0000000e;

  .title {
    font-size: 1.1rem;
    font-weight: bold;
    margin-top: 1rem;
    margin-bottom: 1rem;
  }

  .description {
    opacity: 0.5;
  }

  @media screen and (max-width: 896px) {
    margin-right: 0;
  }
`

const Icon = styled.div`
  display: inline-flex;
  width: 4rem;
  border-radius: 2rem;
  height: 4rem;
  overflow: hidden;
  background: white;
  align-items: center;
  justify-content: center;
  box-shadow: 0 0.5rem 3rem 0 #0000000e;

  img {
    width: 2rem;
  }
`

interface FeatureProps {
  backgroundColor: string
  iconBackgroundColor: string
  iconSrc: string
  iconWidth?: string
  titleKey: LocalizationKey
  descriptionKey: LocalizationKey
}

export const Feature: FC<FeatureProps> = ({
  backgroundColor,
  iconBackgroundColor,
  iconSrc,
  iconWidth,
  titleKey,
  descriptionKey,
}) => {
  return (
    <FeatureWrapper style={{ background: backgroundColor }}>
      <Icon style={{ background: iconBackgroundColor }}>
        <img src={iconSrc} style={{ width: iconWidth }} />
      </Icon>
      <div className="title">
        <Localized name={titleKey} />
      </div>
      <div className="description">
        <Localized name={descriptionKey} />
      </div>
    </FeatureWrapper>
  )
}
