import logging
import azure.functions as func
from openai import AzureOpenAI
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import re
import json
# from langchain.callbacks import get_openai_callback

def preprocess_text(text):
    return re.sub(r'\W+', ' ', text.lower())

def DocProcessor(image_url):
    endpoint = "https://test-ocr-puc.cognitiveservices.azure.com/"
    key = "d20dff6d7c094caeb6f86f7f7c126005"

    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    visual_features = [
        VisualFeatures.TAGS, VisualFeatures.OBJECTS, VisualFeatures.CAPTION,
        VisualFeatures.DENSE_CAPTIONS, VisualFeatures.READ, VisualFeatures.SMART_CROPS, VisualFeatures.PEOPLE,
    ]

    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=visual_features,
        smart_crops_aspect_ratios=[0.9, 1.33],
        gender_neutral_caption=True,
        language="en"
    )

    descriptions_to_match = []

    if result.caption is not None:
        main_caption = result.caption.text
        descriptions_to_match.append(main_caption)

    if result.dense_captions is not None:
        for caption in result.dense_captions.list:
            descriptions_to_match.append(caption.text)

    client = AzureOpenAI(
        azure_endpoint="https://aisqlintegration.openai.azure.com/",
        api_key="d912d0d7acee490ab6e6adc797abe8ff",
        api_version="2024-02-15-preview"
    )

    # Predefined list of categories
    categories = [
        "Abandoned Vehicle", "Accessibility", "Animal - Deceased", "Animal - Domestic", "Animal - General",
        "Damaged Road", "Damaged Street Sign", "Dumped Rubbish", "Dumped Tyres", "Facility - General Request",
        "Fallen Tree", "General - Abandoned Trolley", "General Request", "Graffiti - General", "Graffiti - Public Property",
        "Graffiti - Signage", "Illegal Parking", "Litter", "Noise - Animal", "Noise - Construction", "Noise - General",
        "Overgrown Vegetation", "Park - General Request", "Parking - Disabled", "Pavement - Damaged", "Pavement - General",
        "Pest / Vermin", "Pit and Equipment - General", "Playground Equipment", "Poles and Signage - General",
        "Pollution - General", "Pothole", "Public Toilet", "Request Bin Repair or Replacement", "Road Blockage",
        "Road Signage", "Roads - General", "Rubbish and Bins - General", "Street Cleaning", "Street Gutters / Storm Water",
        "Trees - General", "Vandalism - General", "Water Fountain"
    ]

    prompt_messages = []

    prompt_messages.append({"role": "system", "content": "You are an AI assistant. Only suggest categories from the following list based on the description:"})

    prompt_messages.append({"role": "user", "content": main_caption})

    for description in descriptions_to_match:
        prompt_messages.append({"role": "user", "content": description})

    # for category in categories:
    prompt_messages.append({"role": "system", "content": f"Suggest one best suitable category strictly from the following list only based on the description: {categories}"})

    # for category in categories:
    #     prompt_messages.append({"role": "system", "content": f"Based on MainCaption, DenseCaptions identify best suggested category Type using Enhanced Keyword Extraction, Contextual Analysis and Named Entity Recognition (NER)techniques.Provide suggested category strictly from the category only don't fabricate category on your own.: {category} . Provide answer only strictly."})
   
    
    completion = client.chat.completions.create(
        model="sqlintegration",
        messages=prompt_messages,
        temperature=0,
        max_tokens=1000,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    suggested_category = completion.choices[0].message.content

    # result_with_source = {
    #     'main_caption': main_caption,
    #     'dense_captions': list(descriptions_to_match),
    #     'suggested_category': suggested_category
    # }
    # json_data = json.dumps(result_with_source)
    # return json_data
    input_cost=0.00003*completion.usage.prompt_tokens
    output_cost=0.00006*completion.usage.completion_tokens
    price= input_cost+output_cost
    result_with_source = {
        'main_caption': main_caption,
        'dense_captions': list(descriptions_to_match),
        'suggested_category': suggested_category,
        'completion_token':completion.usage.completion_tokens,
        'total_token':completion.usage.total_tokens,
        'prompt_token': completion.usage.prompt_tokens,
        'total_cost': round(price,4)
    }
    json_data = json.dumps(result_with_source)
    return json_data
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    image_url = req.params.get('imageurl')
    if not image_url:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            image_url = req_body.get('imageurl')

    if image_url:
        return DocProcessor(image_url)
    else:
        return func.HttpResponse(
            "Please provide an image URL in the query string or in the request body.",
            status_code=400
        )