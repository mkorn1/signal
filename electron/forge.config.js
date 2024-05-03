require("dotenv").config()

module.exports = {
  packagerConfig: {
    appBundleId: "jp.codingcafe.signal",
    appCategoryType: "public.app-category.music",
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
    ],
    overwrite: true,
    prune: false,
    platform: "mas",
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
    osxSign: {
      identity: process.env.APPLE_CERTIFICATE_NAME,
      platform: "mas",
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
    },
  },
  makers: [
    {
      name: "@electron-forge/maker-pkg",
      platform: ["mas"],
      config: {
        identity: process.env.APPLE_INSTALLER_CERTIFICATE_NAME,
      },
    },
  ],
}
