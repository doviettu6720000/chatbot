import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import w2v


class Sentence:
    def __init__(self, text, tokenized_text=None):
        self.text = text
        if tokenized_text:
            self.tokenized_text = tokenized_text
        else:
            self.tokenized_text = text
        self.vocab = set(self.tokenized_text.split(" "))
        self.freq = {}
        self.vec = []
        for word in self.tokenized_text.split(" "):
            if word not in self.freq:
                self.freq[word] = 0
            self.freq[word] += 1
            try:
                self.vec.append(w2v.wv[word])
            except:
                pass
        self.vec = np.array(self.vec)

    def get_raw_vocab(self):
        return set(self.text.split(" "))

    def get_tokenized_vocab(self):
        return self.vocab

    def similarity(self, sentence):
        sim = cosine_similarity(sentence.vec, self.vec)
        q_sim = np.max(sim, axis=-1)
        q_sim = q_sim[q_sim > 0.5]
        q_dense = len(q_sim) / len(sentence.vec)
        if len(q_sim) == 0:
            return 0

        t_sim = np.max(sim, axis=0)
        t_sim = t_sim[t_sim > 0.5]
        t_dense = len(t_sim) / len(self.vec)

        return np.mean(q_sim) * (q_dense + t_dense) / 2

    def contains(self, word):
        if type(word) == str:
            return word in self.vocab

        if len(self.vec) == 0:
            return False

        sim = cosine_similarity(np.expand_dims(word, axis=0), self.vec)
        maxVal = np.max(sim)
        return maxVal > 0.6

    def count(self, word):
        if type(word) == str:
            return self.freq.get(word, 0)

        if len(self.vec) == 0:
            return 0

        sim = cosine_similarity(np.expand_dims(word, axis=0), self.vec)
        return len(sim[sim > 0.8])

    def size(self):
        return sum(self.freq.values())


class Paragraph:
    def __init__(self, sentences):
        self.sentences = sentences

    def get_raw_sentence_list(self):
        content = []
        for sentence in self.sentences:
            content.append(sentence.text)
        return content

    def get_tokenized_sentence_list(self):
        content = []
        for sentence in self.sentences:
            content.append(sentence.tokenized_text)
        return content

    def raw_content(self):
        return ". ".join(self.get_raw_sentence_list()) + "."

    def tokenized_content(self):
        return ". ".join(self.get_tokenized_sentence_list()) + "."

    def get_raw_vocab(self):
        vocab = set()
        for s in self.sentences:
            vocab = vocab.union(s.get_raw_vocab())
        return vocab

    def get_tokenized_vocab(self):
        vocab = set()
        for s in self.sentences:
            vocab = vocab.union(s.get_tokenized_vocab())
        return vocab

    def contains(self, word):
        for s in self.sentences:
            if s.contains(word):
                return True
        return False

    def count(self, word):
        return sum([s.count(word) for s in self.sentences])

    def similarity(self, sentence):
        sim = np.array([s.similarity(sentence) for s in self.sentences])
        sim = sim[sim > 0]
        if len(sim) == 0:
            return 0

        return np.mean(sim) * len(sim) / len(self.sentences)


class Dialog:
    def __init__(self, data):
        self.question = Sentence(data["question"], data["question"])
        self.category = data["category"]
        self.paragraphs = []
        for para, t_para in zip(data["paragraphs"]["raw"], data["paragraphs"]["tokenized"]):
            sentences = []
            for s, t_s in zip(para, t_para):
                if s.strip() == "" or t_s.strip() == "":
                    continue
                sentences.append(Sentence(s.strip(), t_s.strip()))
            self.paragraphs.append(Paragraph(sentences))

    def get_para_dict(self):
        raw = [para.get_raw_sentence_list() for para in self.paragraphs]
        tokenized = [para.get_tokenized_sentence_list() for para in self.paragraphs]
        return {
            "raw": raw,
            "tokenized": tokenized
        }

    def to_dict(self):
        result = {
            "question": self.question,
            "category": self.category,
            "paragraphs": self.get_para_dict()
        }
        return result

    def get_raw_vocab(self):
        vocab = self.question.get_raw_vocab()
        for p in self.paragraphs:
            vocab = vocab.union(p.get_raw_vocab())
        return vocab

    def get_tokenized_vocab(self):
        vocab = self.question.get_tokenized_vocab()
        for p in self.paragraphs:
            vocab = vocab.union(p.get_tokenized_vocab())
        return vocab

    def contains(self, word):
        if self.question.contains(word):
            return True

        for p in self.paragraphs:
            if p.contains(word):
                return True
        return False

    def count(self, word):
        return sum([p.count(word) for p in self.paragraphs])

    def similarity(self, sentence):
        q_sim = self.question.similarity(sentence)

        sim = np.array([p.similarity(sentence) for p in self.paragraphs])
        sim = sim[sim > 0]
        if len(sim) == 0:
            return q_sim

        return 0.4 * np.mean(sim) * len(sim) / len(self.paragraphs)  + 0.6 * q_sim
