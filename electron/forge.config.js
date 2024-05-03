module.exports = {
  packagerConfig: {
    appBundleId: "jp.codingcafe.signal",
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
      "^/nodemon.json",
      "^/icons",
    ],
    overwrite: true,
    prune: false,
    appCategoryType: "public.app-category.music",
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
      identityValidation: false,
      provisioningProfile: process.env.APPLE_PROVISIONING_PROFILE,
      optionsForFile: () => ({
        entitlements: "entitlements.plist",
      }),
    },
  },
  makers: [
    {
      name: "@electron-forge/maker-pkg",
      config: {
        identity: process.env.APPLE_INSTALLER_CERTIFICATE_NAME,
      },
    },
  ],
}
