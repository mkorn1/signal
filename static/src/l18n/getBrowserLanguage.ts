export function getBrowserLanguage() {
  // Use URL parameter ?lang=ja or navigator.language
  const navigatorLanguage = navigator.language
  const langParam = new URL(location.href).searchParams.get("lang")
  return langParam ?? navigatorLanguage
}
