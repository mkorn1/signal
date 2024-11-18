import { styled } from "@kuma-ui/core"
import Localized from "../Localized"

const Content = styled.div`
  display: flex;
  gap: 1rem;
  padding: 2rem 0;
  display: flex;
  justify-content: center;
  align-items: center;
  max-width: 60rem;
  width: 100%;
  margin: 0 auto;
`

const Link = styled.a`
  font-size: 0.8rem;
  color: white;
  opacity: 0.5;
  text-decoration: none;

  &:hover {
    opacity: 1;
  }
`

export const Footer = () => {
  return (
    <footer>
      <Content>
        <Link href="/support">
          <Localized name="support" />
        </Link>
        <Link href="/privacy">
          <Localized name="privacy-policy-title" />
        </Link>
      </Content>
    </footer>
  )
}
