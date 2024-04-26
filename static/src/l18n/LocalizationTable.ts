export class LocalizationTable<Languages extends string, Keys extends string> {
  constructor(
    private readonly table: Record<Languages, Record<Keys, string>>,
    private readonly aliases: [RegExp, Languages][],
  ) {}

  getString(language: Languages, key: Keys): string | null {
    if (
      this.table[language] !== undefined &&
      this.table[language][key] !== undefined
    ) {
      return this.table[language][key]
    }
    return null
  }

  getLanguage(language: string): Languages | null {
    for (const [alias, lang] of this.aliases) {
      if (alias.test(language)) {
        return lang
      }
    }
    if (language in this.table) {
      return language as Languages
    }
    return null
  }
}
