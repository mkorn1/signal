import { withKumaUI } from "@kuma-ui/next-plugin"

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
}

export default withKumaUI(nextConfig)
