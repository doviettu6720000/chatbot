import math
import re
import string
from gensim.models import Word2Vec
from vncorenlp import VnCoreNLP


regex = re.compile(f"[{re.escape(string.punctuation)}]")
rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
w2v = Word2Vec.load("./models/word2vec.model")
numerics = ["không", "một", "hai", "ba", "tư", "năm", "sáu", "bảy", "tám", "chín"]


def clean_text(text):
    text = text.strip().lower()
    for i in range(10):
        val = f" {i} "
        if val in text:
            text = text.replace(val, f" {numerics[i]} ")
    return regex.sub('', text.strip())


def tokenize(text):
    sentences = rdrsegmenter.tokenize(text)
    out = []
    for sentence in sentences:
        out.extend(sentence)
    return out


def compute_idfs(doc_list):
    """
    Given a list of `docs` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # count the number of files that contain word
    vocab = set()
    for doc in doc_list:
        vocab = vocab.union(doc.get_tokenized_vocab())

    num_file = dict()
    for word in vocab:
        num_file[word] = 0
        for doc in doc_list:
            if doc.contains(word):
                num_file[word] += 1

    # compute the idfs
    idfs = dict()
    for word in num_file:
        idfs[word] = math.log(len(doc_list) / num_file[word])
    return idfs


def top_dialogs(query_sentence, dialog_list, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the dialogs of the the `n` top
    dialogs that match the query, ranked according to tf-idf.
    """
    sorted_dialogs = list()
    unsorted_dialogs = dict()

    # count the word frequencies in each file
    for i, dialog in enumerate(dialog_list):
        unsorted_dialogs[i] = dialog.similarity(query_sentence)

    # sort the list via dict's values
    count = 0
    for index, val in sorted(unsorted_dialogs.items(),
                                key=lambda item: item[1],
                                reverse=True):
        # print("Q", val)
        sorted_dialogs.append(dialog_list[index])
        count += 1
        if (count == n):
            break

    return sorted_dialogs


def top_paragraphs(query, paragraph_list, p_idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the paragraphs of the the `n` top
    paragraphs that match the query, ranked according to tf-idf.
    """
    sorted_paragraphs = list()
    unsorted_paragraphs = dict()

    # count the word frequencies in each file
    for i, paragraph in enumerate(paragraph_list):
        val = 0
        for word in query:
            if paragraph.contains(word):
                val += paragraph.count(word) * p_idfs[word]
#         unsorted_paragraphs[i] = paragraph.similarity(query)
        unsorted_paragraphs[i] = val

    # sort the list via dict's values
    count = 0
    for index, val in sorted(unsorted_paragraphs.items(),
                                key=lambda item: item[1],
                                reverse=True):
        # print("P", val)
        sorted_paragraphs.append(paragraph_list[index])
        count += 1
        if (count == n):
            break

    return sorted_paragraphs


def top_sentences(query_sentence, sentences, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    unsorted_sentences = dict()

    # compute IDF values
    for i, sentence in enumerate(sentences):
        unsorted_sentences[i] = sentence.similarity(query_sentence)

    # sort the sentence
    count = 0
    chosen_indices = []
    for index, value in sorted(unsorted_sentences.items(),
                               key=lambda item: item[1],
                               reverse=True):
        # print("S", value)
        if value < 0.42:
            break

        chosen_indices.append(index)
        count += 1
        if count == n:
            break

    chosen_indices.sort()
    return [sentences[i] for i in chosen_indices]
