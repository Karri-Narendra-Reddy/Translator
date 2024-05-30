import json
from langdetect import detect
from deep_translator import GoogleTranslator
import sacrebleu
from bert_score import score

# Load the JSON files
with open('english.json', 'r', encoding='utf-8') as ref_file:
    english_data = json.load(ref_file)

with open('telugu.json', 'r', encoding='utf-8') as trans_file:
    telugu_data = json.load(trans_file)

# Extract questions and options from JSON files
english_questions = [q['question'] for q in english_data['questions']]
english_options = [opt for q in english_data['questions'] for opt in q['options']]

print("English Questions:", english_questions)
print("English Options:", english_options)

telugu_questions = [q['question'] for q in telugu_data['questions']]
telugu_options = [opt for q in telugu_data['questions'] for opt in q['options']]

print("Telugu Questions:", telugu_questions)
print("Telugu Options:", telugu_options)

# Combine questions and options for evaluation
english_texts = english_questions + english_options
telugu_texts = telugu_questions + telugu_options

# Back-translate dynamically using deep-translator and langdetect
translator = GoogleTranslator(target='en')
back_translated_texts = []

for text in telugu_texts:
    try:
        # Detect the source language
        source_language = detect(text)
        # Translate text to English
        translated_text = translator.translate(text, source=source_language)
        back_translated_texts.append(translated_text)
    except Exception as e:
        print(f"Error translating text: {text}. Error: {e}")
        back_translated_texts.append(text)  # Append original text if translation fails

print("Back Translated Texts:", back_translated_texts)

# BLEU Score calculation using sacrebleu
bleu = sacrebleu.corpus_bleu(back_translated_texts, [english_texts])
print(f"BLEU score: {bleu.score}")

# BERTScore calculation using bert-score
P, R, F1 = score(back_translated_texts, english_texts, lang="en")
print(f"BERTScore P: {P.mean().item():.6f}")
print(f"BERTScore R: {R.mean().item():.6f}")
print(f"BERTScore F1: {F1.mean().item():.6f}")
