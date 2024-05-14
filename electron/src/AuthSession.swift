import Foundation
import AuthenticationServices

let url = URL(string: CommandLine.arguments[1])!
let callbackURLScheme = CommandLine.arguments[2]

class AuthSessionDelegate: NSObject, ASWebAuthenticationPresentationContextProviding {
    func presentationAnchor(for session: ASWebAuthenticationSession) -> ASPresentationAnchor {
        return ASPresentationAnchor()
    }
}

let delegate = AuthSessionDelegate()

let session = ASWebAuthenticationSession(url: url, callbackURLScheme: callbackURLScheme) { callbackURL, error in
    if let error = error {
        print("Error: \(error.localizedDescription)")
        exit(1)
    }
    if let callbackURL = callbackURL {
        print(callbackURL.absoluteString)
        exit(0)
    }
}

session.presentationContextProvider = delegate
session.start()

RunLoop.current.run()
