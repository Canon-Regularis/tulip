"""Linguistic resource for the synthetic dialect generator.

This module is the *data* half of :mod:`tulip.data.synthetic`: the literal
corpus tables the generator draws from -- carrier templates, slot pools, filler
particles, per-dialect geography, and the two phonological substitution tables
(*mazurzenie* and asynchronous soft-labial respelling). It is separated from the
generator so the linguistic resource can be read, reviewed, and extended without
touching generation logic, and vice versa.

It contains **zero logic** and deliberately imports nothing but
``__future__`` -- no numpy, no yaml, no sklearn -- so that ``import tulip.data``
stays light and pulls in none of the scientific stack.

Ordering is load-bearing. The generator consumes a single
``numpy.random.default_rng(seed)`` in a fixed order and indexes into these
tuples by position, so tuple order is part of the published determinism
guarantee and must not be reordered. In particular :data:`MAZURZENIE` lists
``dż`` before ``ż`` so the longer digraph is consumed first.
"""

from __future__ import annotations

#: Ordered *mazurzenie* substitutions (cz/sz/ż/dż -> c/s/z/dz). ``dż`` is
#: listed first so it maps to ``dz`` before the bare ``ż`` rule fires.
MAZURZENIE: tuple[tuple[str, str], ...] = (
    ("dż", "dz"),
    ("Dż", "Dz"),
    ("cz", "c"),
    ("Cz", "C"),
    ("sz", "s"),
    ("Sz", "S"),
    ("ż", "z"),
    ("Ż", "Z"),
)

#: Ordered asynchronous soft-labial substitutions (pi/bi/wi/mi -> psi/bzi/wzi/mni).
SOFT_LABIALS: tuple[tuple[str, str], ...] = (
    ("pi", "psi"),
    ("Pi", "Psi"),
    ("bi", "bzi"),
    ("Bi", "Bzi"),
    ("wi", "wzi"),
    ("Wi", "Wzi"),
    ("mi", "mni"),
    ("Mi", "Mni"),
)

#: Plausible region and voivodeship pool per lexicon key, so the label
#: hierarchy and the geo/map layer are grounded rather than empty.
GEOGRAPHY: dict[str, tuple[str, tuple[str, ...]]] = {
    "podhale": ("Podhale", ("małopolskie",)),
    "silesia": ("Górny Śląsk", ("śląskie", "opolskie")),
    "kashubia": ("Kaszuby", ("pomorskie",)),
    "kurpie": ("Puszcza Zielona (Kurpie)", ("mazowieckie", "podlaskie")),
    "greater_poland": ("Wielkopolska", ("wielkopolskie",)),
    "masovia": ("Mazowsze", ("mazowieckie", "łódzkie")),
}

# ---------------------------------------------------------------- carrier pool

#: Standard-Polish carrier templates with independently filled slots. They are
#: numerous and multi-slot so slot-filling produces lexically varied texts that
#: (a) comfortably exceed ``DataConfig.min_text_chars`` and (b) survive the
#: character-shingle Jaccard near-dedup in :mod:`tulip.data.dedup`. Several
#: contain cz/sz/ż and pi/bi/wi/mi so the phonological transforms have material
#: to act on and thus produce character-level signal.
CARRIERS: tuple[str, ...] = (
    "W {place} nasz {person} {time} {action} {object}.",
    "{person} z sąsiedniej {place} {action} i {action2} {object2}.",
    "Kiedy {person} {action} do {place} to {person2} {action2} {object}.",
    "{time} na {place} {person} {action} a {person2} {action2}.",
    "Nasz {person} {action} {object} bo {time} nie było {object2}.",
    "Stary {person} opowiadał jak {time} {action} {object} na {place}.",
    "Na {place} {person} i {person2} razem {action} {object}.",
    "{person} {action} {object} zanim {time} poszedł do {place}.",
    "Po {place} {person} {action} niosąc {object} i {object2}.",
    "Tego roku {person} {action} więcej {object} niż {person2}.",
    "{time} w {place} zebrał się {person} żeby {action} {object}.",
    "Mój {person} {action} {object} a potem {action2} przy {place}.",
    "We {place} {person} {time} {action} i długo {action2}.",
    "{person} pamięta jak {person2} {action} {object} na {place}.",
    "Zanim {time} {person} {action} {object} przy {place}.",
    "Na {place} rósł {object} więc {person} {action} go {time}.",
    "{person} {action} do {place} bo {person2} {action2} {object}.",
    "Cała {place} widziała jak {person} {action} {object} {time}.",
    "{time} {person} {action} {object} chociaż padało nad {place}.",
    "Wujek {person} {action} {object2} a ciotka {action2} przy {place}.",
    "{person} {action} {object} i {object2} zaraz po {time}.",
    "Na jarmarku w {place} {person} {action} {object} {time}.",
)

PLACES: tuple[str, ...] = (
    "mieście",
    "wsi",
    "lesie",
    "polu",
    "rzece",
    "dolinie",
    "sadzie",
    "ogrodzie",
    "stodole",
    "kościele",
    "karczmie",
    "młynie",
    "chałupie",
    "zagrodzie",
)
PEOPLE: tuple[str, ...] = (
    "gospodarz",
    "sąsiad",
    "młynarz",
    "kowal",
    "pasterz",
    "wujek",
    "dziadek",
    "chłopak",
    "dziewczyna",
    "nauczyciel",
    "proboszcz",
    "szewc",
    "tkacz",
    "rybak",
)
ACTIONS: tuple[str, ...] = (
    "poszedł",
    "wrócił",
    "śpiewał",
    "pracował",
    "gadał",
    "siedział",
    "budował",
    "kopał",
    "zbierał",
    "warzył",
    "orał",
    "młócił",
    "tańczył",
    "szykował",
)
OBJECTS: tuple[str, ...] = (
    "piwo",
    "chleb",
    "siano",
    "drewno",
    "ziemniaki",
    "grzyby",
    "jagody",
    "masło",
    "mleko",
    "żyto",
    "proso",
    "wełnę",
    "płótno",
    "miód",
)
TIMES: tuple[str, ...] = (
    "rano",
    "wieczorem",
    "w niedzielę",
    "latem",
    "zimą",
    "wczoraj",
    "dzisiaj",
    "o świcie",
    "po żniwach",
    "przed wojną",
    "na jarmarku",
    "na weselu",
)

#: Colloquial filler particles from which each speaker draws a personal subset.
#: They are dialect-neutral (assigned independent of class), so they encode
#: speaker identity -- the leakage that speaker-disjoint splitting defends
#: against -- rather than dialect signal.
FILLERS: tuple[str, ...] = (
    "no",
    "prawda",
    "wiesz",
    "ano",
    "toć",
    "pewnie",
    "juści",
    "widzisz",
    "ha",
    "oj",
    "hej",
    "ejże",
    "dyć",
    "wiadomo",
    "ponoć",
    "otóż",
)
