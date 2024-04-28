import { GoogleAnalytics } from "@next/third-parties/google"
import { Inter } from "next/font/google"
import Head from "next/head"
import { Provider } from "./Provider"
import "./carbon.css"
import "./globals.css"

const inter = Inter({ subsets: ["latin"] })

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <Provider>
      <html lang="en">
        <GoogleAnalytics gaId="G-C4N96XS293" />
        <Head>
          <link rel="canonical" href="https://signal.vercel.app/" />
          <link
            rel="alternate"
            hrefLang="x-default"
            href="https://signal.vercel.app/"
          />
          <link
            rel="alternate"
            hrefLang="en"
            href="https://signal.vercel.app/?lang=en"
          />
          <link
            rel="alternate"
            hrefLang="ja"
            href="https://signal.vercel.app/?lang=ja"
          />
          <link
            rel="alternate"
            hrefLang="zh"
            href="https://signal.vercel.app/?lang=zh"
          />
          <link
            rel="alternate"
            hrefLang="zh-cmn-Hans"
            href="https://signal.vercel.app/?lang=zh-Hans"
          />
          <link
            rel="alternate"
            hrefLang="zh-cmn-Hant"
            href="https://signal.vercel.app/?lang=zh-Hant"
          />
          <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
        </Head>
        <body className={inter.className}>{children}</body>
      </html>
    </Provider>
  )
}
