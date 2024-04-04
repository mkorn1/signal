module.exports = {
  packagerConfig: {
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
