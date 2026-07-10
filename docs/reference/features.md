# Features

Text feature extractors for Polish dialect classification. All are scikit-learn
transformers operating on sequences of strings, registered under canonical names
in `tulip.features.registries.TEXT_FEATURES` (`char_tfidf`, `word_tfidf`,
`stylometry`, `affix_frequency`, `dialect_keywords`, `phonological_markers`).
`build_text_features` composes any subset into a single `FeatureUnion`.

## Composition

::: tulip.features.text.build_text_features

## Vectorizers

::: tulip.features.text.make_char_tfidf

::: tulip.features.text.make_word_tfidf

## Extractors

::: tulip.features.text.StylometryExtractor

::: tulip.features.text.AffixFrequencyExtractor

::: tulip.features.text.DialectKeywordExtractor

::: tulip.features.text.PhonologicalMarkerExtractor

## Lexicon and isogloss resources

::: tulip.features.text.load_lexicon

::: tulip.features.text.load_isoglosses
