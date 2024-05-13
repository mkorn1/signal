import { app, shell } from "electron"
import express from "express"
import { getPort } from "get-port-please"
import { FirebaseCredential } from "./ipc"

interface Options {
  port: number
  onComplete: (credential: FirebaseCredential) => void
}

export const signInWithBrowser = async (
  onComplete: (credential: FirebaseCredential) => void,
) => {
  const port = await getPort()
  let closeTimeout: NodeJS.Timeout

  const server = launchAuthCallbackServer({
    port,
    onComplete: (credential) => {
      server.close()
      clearTimeout(closeTimeout)
      onComplete(credential)
    },
  })

  const parameter = `redirect_uri=http://localhost:${port}`

  shell.openExternal(
    app.isPackaged
      ? `https://signal.vercel.app/auth?${parameter}`
      : `http://localhost:3000/auth?${parameter}`,
  )

  // close server after 5 minutes
  closeTimeout = setTimeout(
    () => {
      if (server.listening) {
        server.close()
      }
    },
    1000 * 60 * 5,
  )
}

const launchAuthCallbackServer = ({ port, onComplete }: Options) => {
  const app = express()

  // IDトークンをクエリパラメータから受け取るエンドポイント
  app.get("/", (req, res) => {
    const credential = req.query.credential // クエリパラメータからIDトークンを取得

    if (!credential) {
      return res.status(400).send("ID Token is missing")
    }

    res.send(successHTML)
    onComplete(JSON.parse(credential as string))
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
