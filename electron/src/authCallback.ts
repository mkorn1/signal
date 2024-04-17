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
    res.send(successHTML)
    // 本来は、ここでJWTの検証などセキュリティ関連の処理を行います。
    onReceiveIdToken(idToken as string)
  })

  const server = app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`)
  })

  return server
}

const successHTML = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Authentication Success - signal</title>
    <style>
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, Avenir, Lato;
            background-color: hsl(228, 10%, 16%);
            color: #ffffff;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            background-color: hsl(228, 10%, 13%);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: hsl(230, 70%, 55%);
        }
        p {
            color: hsl(223, 12%, 60%);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Authentication Successful</h1>
        <p>You may now close this page.</p>
        <p>Thank you for using signal.</p>
    </div>
</body>
</html>
`
