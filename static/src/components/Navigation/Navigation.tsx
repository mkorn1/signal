import { styled } from "@kuma-ui/core"
import Localized from "../Localized"
import githubIcon from "./images/github-icon.svg"
import logoWhite from "./images/logo-white.svg"

const Header = styled.header`
  background: #1f1f23;
`

const Content = styled.div`
  padding: 2rem 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 60rem;
  width: 90%;
  margin: 0 auto;
`

const LogoLink = styled.a`
  display: flex;

  img {
    height: 1.7rem;
  }

  &:hover {
    opacity: 0.7;
  }
`

const Menu = styled.div`
  display: flex;
  align-items: center;
  gap: 2rem;

  & > a {
    color: white;
    text-decoration: none;
    font-size: 0.9rem;
  }
`

export const Navigation = () => {
  return (
    <Header>
      <Content>
        <LogoLink href="/">
          <img src={logoWhite.src} />
        </LogoLink>
        <Menu>
          <a href="/support">
            <Localized name="support" />
          </a>
          <a href="https://github.com/ryohey/signal/" id="github-link">
            <img src={githubIcon.src} />
          </a>
        </Menu>
      </Content>
    </Header>
  )
}
