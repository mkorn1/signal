const { merge } = require("webpack-merge")
const common = require("./webpack.common.js")
const CopyPlugin = require("copy-webpack-plugin")
const WorkboxPlugin = require("workbox-webpack-plugin")

module.exports = merge(common, {
  mode: "production",
  optimization: {
    concatenateModules: false,
    splitChunks: {
      chunks: "all",
    },
  },
  module: {
    rules: [
      {
        test: /\.(j|t)sx?$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader",
        },
      },
    ],
  },
  plugins: [
    new WorkboxPlugin.GenerateSW({
      maximumFileSizeToCacheInBytes: 50000000,
      clientsClaim: true,
      skipWaiting: true,
      runtimeCaching: [
        {
          urlPattern: /^\/.*$/,
          handler: "StaleWhileRevalidate",
        },
        {
          urlPattern: /^.+\.sf2$/,
          handler: "StaleWhileRevalidate",
        },
        {
          urlPattern: /^https:\/\/fonts\.googleapis\.com/,
          handler: "StaleWhileRevalidate",
        },
        {
          urlPattern: /^https:\/\/fonts\.gstatic\.com/,
          handler: "StaleWhileRevalidate",
        },
      ],
    }),
    new CopyPlugin({
      patterns: [
        { from: "public/*.svg", to: "[name][ext]" },
        { from: "public/*.png", to: "[name][ext]" },
        { from: "public/*.webmanifest", to: "[name][ext]" },
      ],
    }),
  ],
})
