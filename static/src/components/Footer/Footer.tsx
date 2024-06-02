import dynamic from "next/dynamic"
import { styled } from "@kuma-ui/core"

const Localized = dynamic(() => import("../../components/Localized"), {
  ssr: false,
})
const Content = styled.div`
  padding: 2rem 0;
  display: flex;
  justify-content: center;
  align-items: center;
  max-width: 60rem;
  width: 100%;
  margin: 0 auto;
`

const Link = styled.a`
  font-size: 0.9rem;
  color: white;
  opacity: 0.6;
  text-decoration: none;
`

export const Footer = () => {
  return (
    <footer>
      <Content>
        <Link href="/privacy">
          <Localized name="privacy-policy-title" />
        </Link>
      </Content>
    </footer>
  )
}
