from flask import Flask, render_template, request
import json
import os
from openai import AzureOpenAI
import demjson3
import sacrebleu
from bert_score import score
from langdetect import detect
from deep_translator import GoogleTranslator

app = Flask(__name__)

def bleuscore(original_text_str, translated_text_str):
    # Load JSON data from strings
    original_text = json.loads(original_text_str)
    translated_text = json.loads(translated_text_str)

    print(original_text)
    print(translated_text)

    english_questions = [q['question'] for q in original_text['questions']]
    english_options = [opt for q in original_text['questions'] for opt in q['options']]

    print(english_questions)
    print(english_options)

    translated_questions = [q['question'] for q in translated_text['questions']]
    translated_options = [opt for q in translated_text['questions'] for opt in q['options']]

    print(translated_questions)
    print(translated_options)

    # Combine questions and options for evaluation
    english_texts = english_questions + english_options
    translated_texts = translated_questions + translated_options

    translator = GoogleTranslator(target='en')
    back_translated_texts = []

    for text in translated_texts:
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
    return bleu.score

def bertscore(original_text_str, translated_text_str):
    # Load JSON data from strings
    original_text = json.loads(original_text_str)
    translated_text = json.loads(translated_text_str)

    english_questions = [q['question'] for q in original_text['questions']]
    english_options = [opt for q in original_text['questions'] for opt in q['options']]

    print(english_questions)
    print(english_options)

    translated_questions = [q['question'] for q in translated_text['questions']]
    translated_options = [opt for q in translated_text['questions'] for opt in q['options']]

    print(translated_questions)
    print(translated_options)

    # Combine questions and options for evaluation
    english_texts = english_questions + english_options
    translated_texts = translated_questions + translated_options

    translator = GoogleTranslator(target='en')
    back_translated_texts = []

    for text in translated_texts:
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


    # BERTScore calculation using bert-score
    P, R, F1 = score(back_translated_texts, english_texts, lang="en")
    print(f"BERTScore P: {P.mean().item():.6f}")
    print(f"BERTScore R: {R.mean().item():.6f}")
    print(f"BERTScore F1: {F1.mean().item():.6f}")
    return P.mean().item(), R.mean().item(), F1.mean().item()

def translate_json(json_to_translate, language):

    prompt = (
        "Translate this " f"{json_to_translate}"
        "\n into this "f"{language}."
        "I need translated accurarcy to be more that 80%"
    )


    os.environ["AZURE_OPENAI_API_VERSION"] = "2023-12-01-preview"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://analyst-augmentation-non-us.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_KEY"] = "73141bf1b4eb4f6bba5216cb6a8e0a33"
    os.environ["OPENAI_API_TYPE"] = "azure_ad"
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gpt-4-32k"

    client = AzureOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    completion = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )

    response_content = completion.choices[0].message.content
    print("Response Content:", response_content)  # Print the response for debugging

    try:
        translated_json = json.loads(response_content)
    except json.JSONDecodeError as e:
        print("JSONDecodeError:", e)
        print("Response content might not be a valid JSON. Attempting to fix it.")

        # Try to fix the JSON
        try:
            repaired_json = demjson3.decode(response_content)
            print("Repaired JSON:", repaired_json)
        except demjson3.JSONError as e:
            print("demjson3 JSONError:", e)
            repaired_json = "Error: Unable to repair JSON."

        return {
            "original": json_to_translate,
            "translated": repaired_json
        }

    return {
        "original": json_to_translate,
        "translated": translated_json
    }


def read_json_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def translate_route():
    language = request.form['language']
    # evaluation = request.form['option']
    print(language)
    # print(evaluation)
    json_data = read_json_from_file("reference-files/ToBeTranslatedFile.json")
    translation_result = translate_json(json_data, language)
    original_text = json.dumps(translation_result["original"], indent=4, ensure_ascii=False)
    translated_text = json.dumps(translation_result["translated"], indent=4, ensure_ascii=False)
    bleu = bleuscore(original_text,translated_text)
    P, R, F1  = bertscore(original_text,translated_text)
    return render_template('result.html', original_text=original_text, translated_text=translated_text,
                           language=language, bleu=bleu, P = P, R=R, F1=F1)


if __name__ == "__main__":
    app.run(debug=True)
