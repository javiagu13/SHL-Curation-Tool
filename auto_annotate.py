from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import spacy
from bert_pipeline.predict import predictBERT
from regex_pipeline.fns.label import predictREGEX
import json 
from argparse import Namespace

nlp = spacy.load("en_core_web_md")

class TextToAnnotate(BaseModel):
    text: str

app = FastAPI()

@app.post("/auto_annotate_BERT_PHI")
async def auto_annotate(document: TextToAnnotate):
    print("preBERT: ---------------------------------------------------")
    print(document.text)
    text_with_labels = predictBERT(document.text)
    print("postBERT: ---------------------------------------------------")
    print(text_with_labels)
    
    try:
        labels_dict = json.loads(text_with_labels)
        ent_label_list = []

        for label_info in labels_dict.get("label", []):
            start_char, end_char, label = label_info
            ent_label_list.append({
                "label": label,
                "start_offset": start_char,
                "end_offset": end_char
            })

        print("_________________________")
        print("RESPONSE TO BE RETURNED:")
        print("_________________________")
        print(ent_label_list)
        return ent_label_list

    except json.JSONDecodeError:
        print("ERROR: Unable to parse JSON. Check the format of predictBERT output.")
        return []  


@app.post("/auto_annotate_REGEX_PHI")
async def auto_annotate(document: TextToAnnotate):
    # Provide the required arguments for predictREGEX
    args = Namespace(configs=["./regex_pipeline/config/smc.yml"], findlog=True)
    
    text_with_labels = predictREGEX(document.text, args)
    print(text_with_labels)
    
    try:
        labels_dict = json.loads(text_with_labels)
        ent_label_list = []

        for label_info in labels_dict.get("label", []):
            start_char, end_char, label = label_info
            ent_label_list.append({
                "label": label,
                "start_offset": start_char,
                "end_offset": end_char
            })

        print("_________________________")
        print("RESPONSE TO BE RETURNED:")
        print("_________________________")
        print(ent_label_list)
        return ent_label_list
    except json.JSONDecodeError:
        print("ERROR: Unable to parse JSON. Check the format of predictBERT output.")



if __name__=="__main__":
    uvicorn.run("auto_annotate:app", host='127.0.0.1', port=7000)
    