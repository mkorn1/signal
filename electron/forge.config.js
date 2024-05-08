require("dotenv").config()

const platform = process.argv[process.argv.indexOf("--platform") + 1]

const packagerConfig = {
  appCategoryType: "public.app-category.music",
  buildVersion: process.env.BUILD_VERSION,
  icon: "./icons/icon",
  ignore: [
    "^/.gitignore",
    "^/src",
    "^/README.md",
    "^/tsconfig.json",
    "^/tsconfig.preload.json",
    "^/node_modules",
    "^/scripts",
    "^/out",
    "^/rollup.config.js",
    "^/forge.config.js",
    "^/entitlements.plist",
    "^/entitlements.child.plist",
    "^/nodemon.json",
    "^/icons",
    "^/.env",
  ],
  overwrite: true,
  prune: false,
  osxUniversal: {
    x64ArchFiles: "*_mac",
  },
  extendInfo: {
    CFBundleDocumentTypes: [
      {
        CFBundleTypeExtensions: ["mid"],
        CFBundleTypeName: "MIDI File",
        CFBundleTypeRole: "Editor",
        LSHandlerRank: "Owner",
      },
    ],
  },
}

switch (platform) {
  case "darwin":
    packagerConfig.appBundleId = "jp.codingcafe.signal.dev"
    packagerConfig.osxSign = {
      platform: "darwin",
      identity: process.env.APPLE_DEVELOPER_CERTIFICATE_NAME,
      preEmbedProvisioningProfile: true,
      provisioningProfile: process.env.APPLE_PROVISIONING_PROFILE,
      optionsForFile: (filePath) => {
        const entitlements = filePath.includes(".app/")
          ? "entitlements.child.plist"
          : "entitlements.plist"
        return {
          hardenedRuntime: false,
          entitlements,
        }
      },
    }
    break
  case "mas":
    packagerConfig.appBundleId = "jp.codingcafe.signal"
    packagerConfig.osxSign = {
      platform: "mas",
      identity: process.env.APPLE_DISTRIBUTION_CERTIFICATE_NAME,
      preEmbedProvisioningProfile: true,
      provisioningProfile: process.env.APPLE_PROVISIONING_PROFILE,
      optionsForFile: (filePath) => {
        const entitlements = filePath.includes(".app/")
          ? "entitlements.child.plist"
          : "entitlements.plist"
        return {
          hardenedRuntime: false,
          entitlements,
        }
      },
    }
    break
}

module.exports = {
  packagerConfig,
  makers: [
    {
      name: "@electron-forge/maker-pkg",
      config: {
        identity: process.env.APPLE_INSTALLER_CERTIFICATE_NAME,
      },
    },
  ],
}
