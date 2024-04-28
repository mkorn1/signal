module.exports = {
  packagerConfig: {
    icon: "./icons/icon",
    ignore: [
      "^/src",
      "^/README.md",
      "^/tsconfig.json",
      "node_modules/electron$",
      "node_modules/@electron-forge$",
    ],
    prune: false,
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
      optionsForFile: () => ({
        entitlements: "entitlements.plist",
      }),
    },
    osxNotarize: {
      tool: "notarytool",
      appleId: process.env.APPLE_ID,
      appleIdPassword: process.env.APPLE_PASSWORD,
      teamId: process.env.APPLE_TEAM_ID,
    },
  },
}
