# Features

Text feature extractors for Polish dialect classification. All are scikit-learn
transformers operating on sequences of strings, registered under canonical names
in `tulip.features.registries.TEXT_FEATURES` (`char_tfidf`, `word_tfidf`,
`stylometry`, `affix_frequency`, `dialect_keywords`, `phonological_markers`,
`phonological_rules`, `dialect_intensity`). `build_text_features` composes any
subset into a single `FeatureUnion`.

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

## Phonological rule engine

Bidirectional rewrite rules for the group-defining isoglosses: a per-document
`applicable`/`fired` feature, a dialect -> standard normaliser, and the shared
core the dialect-intensity feature and the dialect-evidence explainer compose
over.

::: tulip.features.text.PhonologicalRuleExtractor

::: tulip.features.text.load_phonological_rules

::: tulip.features.text.normalize_to_standard

::: tulip.features.text.PhonologicalRule

## Dialect intensity

::: tulip.features.text.DialectIntensityExtractor

## Lexicon and isogloss resources

::: tulip.features.text.load_lexicon

::: tulip.features.text.load_isoglosses
