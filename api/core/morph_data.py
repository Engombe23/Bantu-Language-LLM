# Core morphology table for English -> Swahili Morphological Translation

# Subject prefixes (keys normalized to lowercase simple pronouns)
SUBJECT_PREFIXES = {
    "i": "ni-",      # 1st person singular
    "you": "u-",     # 2nd person singular (default)
    "he": "a-",      # 3rd person singular
    "she": "a-",     # 3rd person singular
    "it": "a-",      # 3rd person singular
    "we": "tu-",     # 1st person plural
    "they": "wa-"    # 3rd person plural
}

# Tense markers
TENSE_MARKERS = {
    "present": "na-",
    "past": "li-",
    "future": "ta-",
    "perfect": "me-",
    "conditional": "nge-",
    "subjunctive": "ki-",
    "neg_present": "si-",
    "neg_past": "ku-",
    "habitual": "hu-",  # default simple present/habitual aspect
}

# Verb roots (examples)
VERB_ROOTS = {
  "read": "soma",
  "write": "andika",
  "eat": "kula",
  "go": "enda",
  "come": "kuja",
  "see": "ona",
  "speak": "sema",
  "learn": "jifunza",
  "play": "cheza",
  "work": "fanya"
}

OBJECT_MARKERS = {
    "me": "ni",     # 1st person singular
    "you": "ku",    # 2nd person (default)
    "him": "m",     # 3rd person singular
    "her": "m",     # 3rd person singular
    "it": "m",      # 3rd person singular
    "us": "tu",     # 1st person plural
    "them": "wa"     # 3rd person plural
}

def get_morphemes(subject: str, tense: str, verb: str, object: str = None):
  """
  Return morpheme sequence tokens (as a space-separated string) for a given
  subject, tense, verb, and optional object.

  Example: get_morphemes("I", "present", "read") -> "ni- na- soma"
  """
  try:
    subject = subject.lower()
    tense = tense.lower()
    verb = verb.lower()

    subject_prefix = SUBJECT_PREFIXES[subject]
    tense_marker = TENSE_MARKERS[tense]
    verb_root = VERB_ROOTS[verb]

    # Optional object
    if object:
      object = object.lower()
      object_marker = OBJECT_MARKERS.get(object, "")
      if object_marker:
        return f"{subject_prefix} {tense_marker}{object_marker}- {verb_root}"
      # If object not recognized, fall back to no object rather than emit empty hyphen
    return f"{subject_prefix} {tense_marker} {verb_root}"
  except KeyError as e:
      return f"[Unknown morphological component: {e}]"