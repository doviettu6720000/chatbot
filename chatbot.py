from distutils.log import Log
import json
from termcolor import colored
from utils import *
from components import Sentence, Paragraph, Dialog

def main():
    # load data
    with open("./data/final_data.json", 'r',encoding="utf8") as f:
        data = json.load(f)["data"]

    dialogs = []
    for dialog in data:
        dialogs.append(Dialog(dialog))

        # Prompt user for query
        text = input("Query: ")
        if text == "q":
            break

        tokenized_query = tokenize(clean_text(text))
        text_query = " ".join(tokenized_query)
        query_sentence = Sentence(text_query)
        query = set(tokenized_query)

        # Determine top file matches according to TF-IDF
        chosen_dialogs = top_dialogs(query_sentence, dialogs, n=2)

        paragraphs = []
        for dialog in chosen_dialogs:
            print("Chosen questions:", colored(dialog.question.text, "red"))
            paragraphs.extend(dialog.paragraphs)
        p_idfs = compute_idfs(paragraphs)
        chosen_paragraphs = top_paragraphs(query, paragraphs, p_idfs, n=2)

        # Extract sentences from top files
        sentences = []
        for para in chosen_paragraphs:
            sentences.extend(para.sentences)

    #     Determine top sentence matches
        matches = top_sentences(query_sentence, sentences, n=5)
        if len(matches) == 0:
            response = colored("Xin lỗi! Tôi không có cầu trả lời cho câu hỏi của bạn.", "red")
        else:
            response = colored(". ".join([s.text for s in matches]) + ".", "green")

        print("BOT:", response)
        print()



if __name__ == "__main__":
    main()
