export function getPlatform() {
  var platform = navigator.platform.toLowerCase()
  var os = null

  if (platform.indexOf("win") >= 0) {
    os = "Windows"
  } else if (platform.indexOf("mac") >= 0) {
    os = "macOS"
  }

  return os
}

export function isRunningInElectron() {
  // Check if we are running in Electron using the user agent
  return navigator.userAgent.toLowerCase().indexOf(" electron/") > -1
}
