import nltk

# Asigură-te că ai resursele necesare
nltk.download("brown")
nltk.download("universal_tagset")  # Necesar pentru mapări interne

from nltk.corpus import brown


def get_brown_as_universal():
    brown_tagged = brown.tagged_sents(tagset="universal")  # Obținem etichetele Brown
    return brown_tagged


def get_brown_sentences():
    brown_tagged = brown.sents()  # Obținem doar propozițiile fără etichete
    return brown_tagged


def brown_to_training_data(
    data: list[list[tuple[str, str]]],
) -> list[tuple[str, list[str]]]:
    training_data = []
    for sent in data:
        words, tags = zip(*sent)
        training_data.append((" ".join(words), list(tags)))
    return training_data


def get_tag_list():
    return [
        "DET",
        ".",
        "ADV",
        "X",
        "NOUN",
        "ADJ",
        "VERB",
        "CONJ",
        "PRT",
        "NUM",
        "ADP",
        "PRON",
    ]
    # Obținem lista de etichete unificate
    # return list(set(tag for sent in get_brown_as_universal() for _, tag in sent))


# Testăm
if __name__ == "__main__":
    data = get_brown_as_universal()
    print(data[0:10])
    print(len(data))
