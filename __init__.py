import json
import os
from openai import AzureOpenAI


def translate(text_to_translate, language):
    prompt = (
        "So, I have some translation to be done from English language to "
        f"{language}"
        "Translate the following text: "
        f"{text_to_translate} \n"
    )
    print(prompt)

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
    return completion.choices[0].message.content


def translate_json(json_to_translate, language):
    prompt = (
        "So, I have a json file to translate which is in English language \n following is the json file data : \n " f"{json_to_translate}"
        "\n Thus, convert the json above into "f"{language}."
        "Translate only the questions value and the options."
        "Return the original text, the translated text in json format."
    )
    print(prompt)

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
    return completion.choices[0].message.content


def read_json_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


if __name__ == "__main__":
    json_data = read_json_from_file("reference-files/ToBeTranslatedFile.json")
    translated_text = translate_json(json_data, "Hindi")
    print(translated_text)
