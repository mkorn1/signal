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
  },
}
