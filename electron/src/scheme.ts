const bundleId =
  process.mas === true || process.windowsStore === true
    ? "jp.codingcafe.signal"
    : "jp.codingcafe.signal.dev"
const scheme = `${bundleId}://`

export const authCallbackUrl = `${scheme}auth-callback`
