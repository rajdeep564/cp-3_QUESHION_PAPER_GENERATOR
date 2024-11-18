import os
import tempfile
import PyPDF2
import google.generativeai as genai
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import re
import random
import requests
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import sent_tokenize
import pke
from nltk.corpus import stopwords
import string
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from nltk.corpus import wordnet as wn
import collections
from collections import OrderedDict

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('popular')
nltk.download('wordnet')
nltk.download('famous')
nltk.download("brown")
import pprint
import spacy

# Set your API key as an environment variable
os.environ["GOOGLE_API_KEY"] = 'AIzaSyD42TVrhk5qzJNNPoDVjrZBfPcWGXs2V7I'

# Configure the API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


# Function to extract text from PDF
def get_pdf_text(pdf_file):
    text = ""
    try:
        with pdf_file as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error: {e}")
    return text.strip()


# Function to generate question paper from text
def generate_question_paper_from_text(subject_name, timing, subject_code, text):
    # Define the structure of the question paper
    question_paper = ""
    question_paper += " " * 70 + "END SEMESTER EXAMINATION\n\n"
    question_paper += f"          Subject : {subject_name}{' ' * (60 - len(subject_name))}     {' ' * 40}Max Marks : 60\n"
    question_paper += f"          Time : {timing}{' ' * (65 - len(timing))}      {' ' * 40}Subject code : {subject_code}\n"
    question_paper += "       " + "-" * 124  # Line separator
    question_paper += "Instructions :\n"
    question_paper += "* All Questions are Compulsory\n"
    question_paper += "* Start Each question on New Page\n"
    question_paper += "* There are total 6 Questions Q.1 to Q.6 each consisting of 10 marks\n"
    question_paper += "* Draw Diagrams wherever necessary\n"
    question_paper += "* Use of Calculators is not allowed\n"
    question_paper += "* Any type of cheating will lead to the disqualification of Candidate\n\n"
    question_paper += f"      {'-' * 123}\n\n"

    # Initialize the generative model
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Start a chat session
    chat = model.start_chat(history=[])

    prompt = f"Based on the given text Generate the questions such that there will be total 6 questions from Q.1) to Q.6) each carrying 10 marks.I want only questions, note the answers. sentence grammer should be correct.for each subquestion display marks on right hand side only with some spacing in round brackets like (5 marks).There will be compulsary one line spacing between each question from Q.1) to Q.6) (like if Q.1 finishes on 3rd line,Q2 will start on 5th line okay).there should be no * in a question like **Q1)**, remove them Each question has two subquestions A) and B) of 5 marks each. All questions from 1 to 6 should be descriptive and may be of type (not limited to) like What do you mean by , explain with suitable example, write a short note on, write the diffference between ,explain the similarities in between, explain the advantages and disadvanatages of, explain in detail with diagrams,write the applications and uses of, explain law/theorem/algorithm in detail, give reasoons in detail, derivation if any etc. Q.4)Contains three subquestions A, B, C. B and C are for two marks each and A is for 6 marks. There is option for question A that is in Q.4A) their will be choice to solve one among two  questions are separated by OR(compulsarily center allign in page) in between middle line .Q.5) Contains three questions A) B) C) for 3,3,4 marks respectively. question format for Q.4 & Q.5 are same as first three, just two marks questions will be very easy Q.6) contains two questions A) B), A) will be for 6 marks aand B for 4 marks. also proper spacing and formating (formatted text) and spacing of one line between each main  question(Q.1,2,3...). all question should be from text only. if you do not find any text, then make question relevant to topics of the subject of paper provided, and in such case make questions from distinct topics. also try to give euqla wieghtage to each unit or section. question should be easy and not complex. only display Q.1)A) Q.2)B) like this and not heading question like Q.1) and then A) and B). also do not display * in between questions like **Q1)**, dont do this. In case of any errors like stop indentation ,error 500 use the same trick.note that each subquestion should start from new line and there is compulsary spacing of 1 line between each question (like Q1 and one line spacing then Q2)....).i want only questions not answers,each subquestion should start on new line:\n{subject_name}\n\n{text}"
    response = chat.send_message(prompt, stream=True)
    for chunk in response:
        question_paper += chunk.text
    question_paper += "\n\n"
    question_paper += " " * 75 + "\n\n---------------------------------------------- BEST OF LUCK !!!!---------------------------------------------\n"
    return question_paper


# Function to generate PDF
def generate_pdf(subject_name, max_marks, timing, subject_code, question_paper):
    pdf_file_path = "question_paper.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)

    # Define styles for headings and paragraphs
    question_style = styles["Normal"]
    question_style.fontSize = 11
    question_style.leading = 14  # Adjust line spacing

    # Add question paper content
    content = []

    # Split question paper content into lines and add to PDF
    lines = question_paper.strip().split('\n')
    for line in lines:
        paragraph = Paragraph(line.strip(), question_style)
        content.append(paragraph)
        content.append(Spacer(1, 10))  # Add space between questions

    doc.build(content)
    return pdf_file_path


# Function to initialize the T5 question generation model and tokenizer
def question_model_tokenizer():
    question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
    return question_model, question_tokenizer


# Function to generate a question using the T5 model
def get_question(sentence, answer, mdl, tknizer):
    text = "context: {} answer: {}".format(sentence, answer)
    max_len = 256
    encoding = tknizer.encode_plus(text, max_length=max_len, pad_to_max_length=False, truncation=True,
                                   return_tensors="pt")

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = mdl.generate(input_ids=input_ids,
                        attention_mask=attention_mask,
                        early_stopping=True,
                        num_beams=5,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        max_length=300)

    dec = [tknizer.decode(ids, skip_special_tokens=True) for ids in outs]
    question = dec[0].replace("question:", "")
    question = question.strip()
    return question


# Function to tokenize sentences from the text
def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences


def Keyword_Extraction(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    named_entities = set([ent.text for ent in doc.ents])

    extractor = pke.unsupervised.TfIdf()
    extractor.load_document(input=text, language='en', normalization=None)
    extractor.candidate_selection(n=3)
    extractor.candidate_weighting()
    keyphrases_TFIDF = extractor.get_n_best(n=10)

    extractor = pke.unsupervised.KPMiner()
    extractor.load_document(input=text, language='en', normalization=None)
    extractor.candidate_selection(lasf=5, cutoff=200)
    extractor.candidate_weighting(alpha=2.3, sigma=3.0)
    keyphrases_KPMiner = extractor.get_n_best(n=10)

    extractor = pke.unsupervised.YAKE()
    extractor.load_document(input=text, language='en', normalization=None)
    extractor.candidate_selection(n=3)
    extractor.candidate_weighting(window=2, use_stems=False)
    keyphrases_Yake = extractor.get_n_best(n=10, threshold=0.8)

    pos = {'NOUN', 'PROPN', 'ADJ'}
    extractor = pke.unsupervised.TextRank()
    extractor.load_document(input=text, language='en', normalization=None)
    extractor.candidate_weighting(window=2, pos=pos, top_percent=0.33)
    keyphrases_TextRank = extractor.get_n_best(n=10)

    extractor = pke.unsupervised.SingleRank()
    extractor.load_document(input=text, language='en', normalization=None)
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases_SingleRank = extractor.get_n_best(n=10)

    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=text)
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(threshold=0.74, method='average')
    keyphrases_TopicRank = extractor.get_n_best(n=10)

    grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
    extractor = pke.unsupervised.TopicalPageRank()
    extractor.load_document(input=text, language='en', normalization=None)
    extractor.candidate_selection(grammar=grammar)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases_TopicalPageRank = extractor.get_n_best(n=10)

    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=text, language='en', normalization=None)
    extractor.candidate_selection(grammar=grammar, maximum_word_number=3)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases_PositionRank = extractor.get_n_best(n=10)

    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text)
    extractor.candidate_selection(pos=pos)
    extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')
    keyphrases_MultipartiteRank = extractor.get_n_best(n=10)

    stoplist = stopwords.words('english')
    extractor = pke.supervised.Kea()
    extractor.load_document(input=text, language='en', normalization=None)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases_Kea = extractor.get_n_best(n=10)

    extractor = pke.supervised.WINGNUS()
    extractor.load_document(input=text)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases_WINGNUS = extractor.get_n_best(n=10)

    Keywords = keyphrases_TFIDF + keyphrases_KPMiner + keyphrases_Yake + keyphrases_TextRank + keyphrases_SingleRank + keyphrases_TopicRank + keyphrases_TopicalPageRank + keyphrases_PositionRank + keyphrases_MultipartiteRank + keyphrases_Kea + keyphrases_WINGNUS
    Keywords = [k for k, v in Keywords if k not in named_entities]

    duplicated_Keywords = [item for item, count in collections.Counter(Keywords).items() if count > 1]
    return duplicated_Keywords


# Function to get distractors using WordNet
def get_distractors_wordnet(syn, word):
    distractors = []
    word = word.lower()
    orig_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        if name == orig_word:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors


# Function to get distractors using ConceptNet
def get_distractors_conceptnet(word):
    word = word.lower()
    original_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    distractor_list = []
    url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5" % (word, word)
    obj = requests.get(url).json()

    for edge in obj['edges']:
        link = edge['end']['term']
        url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10" % (link, link)
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)

    return distractor_list


# Function to generate MCQ options for each keyword
# Function to generate MCQ options for each keyword
def generate_mcq_options(keyword):
    synsets = wn.synsets(keyword, 'n')
    distractors_wordnet = []
    for syn in synsets:
        distractors_wordnet += get_distractors_wordnet(syn, keyword)
    distractors_conceptnet = get_distractors_conceptnet(keyword)
    distractors = list(set(distractors_wordnet + distractors_conceptnet))
    options = [keyword] + random.sample(distractors, min(3, len(distractors)))
    random.shuffle(options)
    return options


# Function to get sentences containing keywords
def get_sentences_for_keyword(keywords, sentences):
    keyword_sentence_mapping = {}
    for keyword in keywords:
        keyword_sentence_mapping[keyword] = []
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                keyword_sentence_mapping[keyword].append(sentence)
    return keyword_sentence_mapping


# Function to generate questions and MCQs from the text
# Function to generate questions and MCQs from the text
def generate_questions_and_mcqs(examination_name, text):
    mdl, tknizer = question_model_tokenizer()
    sentences = tokenize_sentences(text)
    duplicated_keywords = Keyword_Extraction(text)
    keyword_sentence_mapping = get_sentences_for_keyword(duplicated_keywords, sentences)

    question_number = 1
    generated_questions = set()  # Keep track of generated questions to avoid duplicates
    question_paper = ""  # Initialize the question paper

    # Concatenate examination name and instructions to the question paper
    question_paper += ' ' * 40 + f"{examination_name}\n"

    # Concatenate instructions to the question paper
    question_paper += """
    ------------------------------------------------------------------------------------------
        Instructions :

        * All Questions are Compulsory
        * Start Each question on New Page
        * There are total 6 Questions Q.1 to Q.6 each consisting of 10 marks
        * Draw Diagrams wherever necessary
        * Use of Calculators is not allowed
        * Any type of cheating will lead to the disqualification of Candidate
    -------------------------------------------------------------------------------------------
        """
    # Generate questions and MCQs
    for keyword, sentences in keyword_sentence_mapping.items():
        if not sentences:  # Skip if the list of sentences is empty
            continue
        if keyword in generated_questions:  # Skip if question already generated for this keyword
            continue
        generated_questions.add(keyword)
        sentence = random.choice(sentences)  # Randomly select one sentence for the keyword
        answer = keyword
        short_sentence = sentence[:200]  # Limiting the sentence to 200 characters
        question = get_question(short_sentence, answer, mdl, tknizer)
        options = generate_mcq_options(keyword)
        if len(options) == 4:
            # Add question number, question, and options to the question paper
            question_paper += f"Q.{question_number})\n"
            question_paper += f"{question}\n"
            for idx, option in enumerate(options, start=97):  # start=97 corresponds to 'a' in ASCII
                question_paper += f"{chr(idx)}. {option}\n"
            question_paper += '\n'

            question_number += 1

    # Return the generated question paper
    return question_paper



nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")

# Function to remove subordinate clauses from text using spaCy
def remove_subordinate_clauses(text):
    # Process the text with spaCy
    doc = nlp(text)
    # Initialize an empty list to store the main clauses
    main_clauses = []
    # Iterate through each sentence
    for sent in doc.sents:
        # Check if the sentence contains a subordinate conjunction or a dependent clause
        if any(token.dep_ == "mark" or token.dep_ == "advcl" for token in sent):
            # Remove the subordinate clause
            main_clause = ""
            for token in sent:
                if token.dep_ == "mark" or token.dep_ == "advcl":
                    break
                main_clause += token.text_with_ws
            main_clauses.append(main_clause.strip())
        else:
            main_clauses.append(sent.text)
    return main_clauses

# Function to tokenize sentences from the text
def tokenize_sentences(text):
    sentences = remove_subordinate_clauses(text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

# Function to extract named entities from text using spaCy
def get_named_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'LOC', 'GPE', 'PERSON']:
            entities.add(ent.text)
    return list(entities)

# Function to get sentences containing keywords
def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

# Function to generate fill-in-the-blanks questions from the given sentence mapping
def get_sentences_for_keyword(keywords, sentences):
    """
    Get sentences containing the given keywords.

    Args:
        keywords (list): List of keywords.
        sentences (list): List of sentences.

    Returns:
        dict: Dictionary mapping keywords to sentences.
    """
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

def get_fill_in_the_blanks(sentence_mapping, max_questions=6):
    out = {"Q)": "Q) Fill in the blanks with matching words at the top : "}
    blank_sentences = []
    keys = []
    used_sentences = set()
    question_count = 0
    for key in sentence_mapping:
        if len(sentence_mapping[key]) > 0:
            for sent in sentence_mapping[key]:
                if sent not in used_sentences and key.capitalize() not in keys:
                    insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
                    no_of_replacements = len(re.findall(re.escape(key), sent, re.IGNORECASE))
                    line = insensitive_sent.sub(' _____________ ', sent)
                    if no_of_replacements < 2:
                        # Ensure the last symbol is a full stop
                        if line[-1] not in string.ascii_letters:
                            line = line.rstrip(string.punctuation) + "."
                        blank_sentences.append(line)
                        keys.append(key.capitalize())  # Capitalizing the options
                        used_sentences.add(sent)
                        question_count += 1
                        if question_count >= max_questions:
                            break
        if question_count >= max_questions:
            break
    # Remove spaces around each item in Options and join them with commas
    out["Options"] = keys[:10]
    # Remove the comma between questions
    out["Questions"] = blank_sentences[:10]

    return out


def generate_match_the_pair(text):
    # Define the structure of the question paper
    question_paper = ""
    # Initialize the generative model
    model = genai.GenerativeModel('gemini-pro')
    # Start a chat session
    chat = model.start_chat(history=[])

    prompt = f'''
     for the given text i want to generate match the pairs question with two columns A and B Both having four entries like:
     (just example)
     Match the column A with Column B:
     
     Q.1)  column A                            column B
           1)                                     A) 
           2)                                     B)
           3)                                     C)
           4)                                     D)
    represent each match the pair question in tabular format
    if multiple questions then spacing of 1-2 lines between each question that is like example if Q.1 finish on 5th line and Q.2 start on 7th
    if possible then maximum 2 match the following 4*4 questions with numberoing Q1), Q2), else one match the pair is also fine
    all the question should be strictly based on provided text only\n\n{text}
     '''
    response = chat.send_message(prompt, stream=True)
    for chunk in response:
        question_paper += chunk.text
    question_paper += "\n\n"
    return question_paper


def generate_True_or_False(text):
    # Define the structure of the question paper
    question_paper = ""
    # Initialize the generative model
    model = genai.GenerativeModel('gemini-pro')
    # Start a chat session
    chat = model.start_chat(history=[])

    prompt = f'''
     for the given text generate True or false questions (only questions not answers) like example (just example) formating should be like this only :
     Q)State Whether the following statements are true or false : (heading question and all question below it as shown)
     1) Virat Kohli is the captain Indian National Hockey Team
     2) Narendra Modi is the prime minster of India
     and so on..............
     you can use the logic that for true printing the fsact statement as it is whereas for false converting positive statement to negative and negative statement to positive.
     you can change the statement as well,like if ram mandir is in ayodhya to ram mandir is in mithila (just example)
     equal number of true and false questions should be there compulsarily,like if tootal 6 then 3 should be false and 3 should be true
     question should be in range 3-10 questions based on content
     all the questions should be strictly on text only\n\n{text}
     '''
    response = chat.send_message(prompt, stream=True)
    for chunk in response:
        question_paper += chunk.text
    question_paper += "\n\n"
    return question_paper

# Main function
def main():
    st.title('Question Paper Generator')

    # Slider for selecting subjective or objective
    mode = st.sidebar.radio("Select Mode:", ("Subjective", "Objective"))

    if mode == "Subjective":
        st.subheader("Subjective Mode")
        # User inputs
        subject_name = st.text_input('Enter Subject Name:', '')
        timing = st.text_input('Enter Timing:', '')
        subject_code = st.text_input('Enter Subject Code:', '')
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

        # Generate question paper
        if pdf_file is not None and st.button('Generate Question Paper'):
            text = get_pdf_text(pdf_file)
            question_paper = generate_question_paper_from_text(subject_name, timing, subject_code, text)
            st.text_area('Question Paper:', question_paper, height=400)

            # Generate and provide download link for PDF
            pdf_file_path = generate_pdf(subject_name, 60, timing, subject_code, question_paper)
            with open(pdf_file_path, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(label="Download PDF", data=pdf_bytes, file_name="question_paper.pdf",
                               mime="application/pdf")

            # Remove the temporary PDF file
            os.unlink(pdf_file_path)
    else:
        # Objective Mode implementation
        st.subheader("Objective Mode")
        examination_name = st.text_input('Enter Examination Name:', '')
        input_text = st.text_area("Enter Text for Question Generation:")
        question_type = st.selectbox("Select Question Type:", (
            "Multiple Choice (MCQ)", "Fill in the Blanks", "True or False", "Match the Pairs"))


        # Here you can implement the logic to generate objective questions based on the input text and selected question type
        if st.button('Generate Questions'):
            if question_type == "Multiple Choice (MCQ)":
                # Generate questions and MCQs
                question_paper = generate_questions_and_mcqs(examination_name, input_text)

                # Display question paper
                st.text_area('Question Paper:', question_paper, height=400)

                # Generate and provide download link for PDF
                pdf_file_path = generate_pdf(examination_name, 60, "Timing", "Subject Code", question_paper)
                with open(pdf_file_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(label="Download PDF", data=pdf_bytes, file_name="question_paper.pdf",
                                   mime="application/pdf")
                os.unlink(pdf_file_path)

            elif question_type == "Fill in the Blanks":
                # Tokenize sentences from the input text
                sentences = tokenize_sentences(input_text)
                # Extract named entities from the input text
                named_entities = get_named_entities(input_text)
                # Get sentences containing keywords
                keyword_sentence_mapping = get_sentences_for_keyword(named_entities, sentences)
                # Generate fill-in-the-blanks questions
                fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping)
                # Print the fill-in-the-blanks questions without JSON formatting
                st.write(fill_in_the_blanks["Q)"])
                st.write("\n")
                st.write(fill_in_the_blanks["Options"])
                st.write("\n")
                st.write("\n".join([f"{i + 1}) {question}" for i, question in enumerate(fill_in_the_blanks["Questions"])]))


            elif question_type == "True or False":
                question_paper = generate_True_or_False(input_text)
                st.write(question_paper)


            elif question_type == "Match the Pairs":
                question_paper = generate_match_the_pair(input_text)
                st.write(question_paper)


if __name__ == '__main__':
    main()
