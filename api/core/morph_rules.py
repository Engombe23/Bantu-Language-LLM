from api.core.morph_data import get_morphemes

# Morphological Preprocessing
def morph_preprocess(text: str) -> str:
    """
    Parse simple English sentence into (subject, tense, verb).
    Returns Swahili morpheme sequence string.
    """

    text = text.lower().strip()
    tokens = text.split()

    subject, tense, verb = None, None, None

    # Detect subjects
    for sub in ["i", "you", "he", "she", "it", "we", "they"]:
        if tokens[0] == sub:
            subject = sub
            break
    
    # Detect tenses
    if "will" in tokens:
        tense = "future"
        verb = tokens[-1]
    elif "have" in tokens or "has" in tokens:
        tense = "perfect"
        verb = tokens[-1]
    elif "am" in tokens or "is" in tokens or "are" in tokens:
        if tokens[-1].endswith("ing"):
            tense = "present"
            verb = tokens[-1].replace("ing", "")
        else:
            tense = "present"
            verb = tokens[-1]
    elif tokens[-1].endswith("ed"):
        tense = "past"
        verb = tokens[-1].replace("ed", "")
    else:
        tense = "habitual"
        verb = tokens[-1]

    if not (subject and tense and verb):
        return "[Parsing error: could not detect subject/tense/verb]"

    return get_morphemes(subject, tense, verb)

# Postprocessing

def morph_postprocess(morpheme_str: str) -> str:
    """
    Convert morpheme sequence into Swahili sentence.
    """
    morphemes = morpheme_str.replace("-", "").split()
    return "".join(morphemes)