import express from "express"

interface Options {
  port: number
  onReceiveIdToken: (idToken: string) => void
}

export const launchAuthCallbackServer = ({
  port,
  onReceiveIdToken,
}: Options) => {
  const app = express()

  // IDトークンをクエリパラメータから受け取るエンドポイント
  app.get("/", (req, res) => {
    const idToken = req.query.idToken // クエリパラメータからIDトークンを取得

    if (!idToken) {
      return res.status(400).send("ID Token is missing")
    }

    // ここでIDトークンの検証を行う
    // 検証が成功した場合の処理
    res.send(`ID Token is received and verified: ${idToken}`)
    // 本来は、ここでJWTの検証などセキュリティ関連の処理を行います。
    onReceiveIdToken(idToken as string)
  })

  const server = app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`)
  })

  return server
}
