import requests
import json
import csv

# Azure subscription keys and service region
speech_key = "ae867631672942ffb0a8accc63ba4339"
service_region = "eastus"
translator_key = "16d3dee4bd37424a97dc72a7722cfa31"
translator_endpoint = "https://api.cognitive.microsofttranslator.com/"

# Default target language for translation
target_language = "kn"  # Kannada
source_language = "en-GB"  # English (Great Britain)

# Function to translate text from English (GB) to Kannada
def translate_to_kannada(text):
    path = '/translate?api-version=3.0'
    params = f'&from={source_language}&to={target_language}'
    constructed_url = translator_endpoint + path + params
    
    # Headers
    headers = {
        'Ocp-Apim-Subscription-Key': translator_key,
        'Ocp-Apim-Subscription-Region': service_region,
        'Content-type': 'application/json',
    }
    
    # Body with the text to translate
    body = [{
        'text': text
    }]
    
    # Make the request
    response = requests.post(constructed_url, headers=headers, json=body)
    
    # Parse the response
    if response.status_code == 200:
        result = response.json()
        translated_text = result[0]['translations'][0]['text']
        return translated_text
    else:
        return f"Error: {response.status_code}, {response.text}"

# Function to save the translated text to a CSV file
def save_translation_to_csv(original_text, translated_text, filename='translated_text.csv'):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Original Text', 'Translated Text'])  # CSV header
        writer.writerow([original_text, translated_text])  # Original and translated text

# Example usage
text_to_translate = "Our aim is to contribute to society by developing a tool that translates between English and Tulu. The Tulu Translator will facilitate communication and enhance understanding between speakers of these languages, supporting the preservation and promotion of local culture."
translated_text = translate_to_kannada(text_to_translate)
print(f'Translated Text: {translated_text}')

# Save to CSV
save_translation_to_csv(text_to_translate, translated_text)
print(f'Translation saved to CSV file.')
