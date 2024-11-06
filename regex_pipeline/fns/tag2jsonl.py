import spacy
import re

def preprocess_text(text):
    # Add space before and after the tags
    text = re.sub(r"(<[^>]+>)", r" \1 ", text)
    # Replace < and > with unique markers
    text = text.replace("<", "@#@")
    text = text.replace(">", "#@@")
    return text

def postprocess_text(parsed_text, bio_labels):
    processed_text = []
    processed_labels = []
    for word, label in zip(parsed_text, bio_labels):
        if "@#@" in word:
            continue
        if word.strip():  # Ignore pure whitespace tokens
            processed_text.append(word.strip())
            processed_labels.append(label)
    return processed_text, processed_labels

def parse_text(text):
    nlp = spacy.load("en_core_web_sm")
    preprocessed_text = preprocess_text(text)
    doc = nlp(preprocessed_text)
    
    parsed_text = []
    bio_labels = []
    
    currentBIO = ""
    currentLabel = ""
    
    for token in doc:
        #print(token)
        if token.text.startswith("@#@/") and currentBIO:
            currentBIO = ""
            currentLabel = ""
        elif token.text.startswith("@#@") and token.text.endswith("#@@") and currentBIO == "":
            label = token.text.strip("@#@")
            #label = label.text.strip("@#@")
            currentLabel = label
            currentBIO = "B"
        elif currentBIO and not token.text.startswith("@#@"):
            if token.text.strip():  # Ensure the token is not just whitespace
                parsed_text.append(token.text)
                bio_labels.append(currentBIO + "-" + currentLabel) # change this if you want B-LABEL or LABEL-B
                currentBIO = "I"
            else:
                # Handle space as part of the same tag
                currentBIO = "B"
        else:
            parsed_text.append(token.text)
            bio_labels.append("O")
    
    return postprocess_text(parsed_text, bio_labels)

def parse_text_no_BI(text):
    nlp = spacy.load("en_core_web_sm")
    preprocessed_text = preprocess_text(text)
    doc = nlp(preprocessed_text)
    
    parsed_text = []
    bio_labels = []
    
    currentBIO = ""
    currentLabel = ""
    
    for token in doc:
        #print(token)
        if token.text.startswith("@#@/") and currentBIO:
            currentBIO = ""
            currentLabel = ""
        elif token.text.startswith("@#@") and token.text.endswith("#@@") and currentBIO == "":
            label = token.text.strip("@#@")
            #label = label.text.strip("@#@")
            currentLabel = label
            currentBIO = "B"
        elif currentBIO and not token.text.startswith("@#@"):
            if token.text.strip():  # Ensure the token is not just whitespace
                parsed_text.append(token.text)
                bio_labels.append(currentLabel) # change this if you want B-LABEL or LABEL-B
                currentBIO = "I"
            else:
                # Handle space as part of the same tag
                currentBIO = "B"
        else:
            parsed_text.append(token.text)
            bio_labels.append("O")
    
    return postprocess_text(parsed_text, bio_labels)


# Example usage
text = "<CVL>떡갈비구이간의</CVL> 당돌은 <DAT>지난 7일</DAT> <ORG>친박에서</ORG> 열린 개혁파 <CVL> 개인투자자들의</CVL> 금요예배 이관 <DAT>초나흗날</DAT> 만입니다."
parsed_text, bio_labels = parse_text(text)
parsed_text_no_BI, bio_labels_no_BI = parse_text_no_BI(text)
print(parsed_text)
print(bio_labels)
print(bio_labels_no_BI)

#This code takes a sentence like this: <CVL>떡갈비구이간의</CVL> 당돌은 <DAT>지난 7일</DAT> <ORG>친박에서</ORG>
#It outputs the following 떡갈비구이간의 당돌은 지난 7일 친박에서 \t CVL O DAT ORG 
#Or it outputs the following 떡갈비구이간의 당돌은 지난 7일 친박에서 \t CVL O DAT ORG 
def tag_to_BIO_oneLineFormat(line, isBIO):
    #modifiedString = re.sub(r'\s+', ' ', csv_rows[1])
    if isBIO==True:
        parsed_text, bio_labels = parse_text(line)
    else:
        parsed_text, bio_labels = parse_text_no_BI(line)
    #print(parsed_text)
    #print(bio_labels)

    return " ".join(parsed_text)+"\t"+" ".join(bio_labels)

def tag_to_BIO_coNLLFormat(line, isBIO):
    #modifiedString = re.sub(r'\s+', ' ', csv_rows[1])
    result=""
    if isBIO==True:
        parsed_text, bio_labels = parse_text(line)
    else:
        parsed_text, bio_labels = parse_text_no_BI(line)
    #print(parsed_text)
    #print(bio_labels)
    for i in range(0,len(parsed_text)):
        result+=parsed_text[i]+"\t"+bio_labels[i]+"\n"
        
    return result+"\n"
#tag_to_BIO_oneLineFormat(csv_rows, True):


#By inputting a whole array of lines, asking for style (conll or one line BIO) and asking whether you want BIO format or not it returns the result 
#text_array: is an array which each element corresponds to one line of a text file
#style: CoNLL style or one_line are available
#isBIO: if it is true it will provide the result with BI after label otherwise simply label or O.
def tag_to_BIO(text_array, style, isBIO):
    result=""
    if style=="CoNLL":
        for line in text_array:
            aux_result=tag_to_BIO_coNLLFormat(line, True)
            result+=aux_result
    elif style=="one_line":
        for line in text_array:
            aux_result=tag_to_BIO_oneLineFormat(line, True)
            print(aux_result)
            result+=aux_result
    else:
        print("Please, provide an existing style type: CoNLL or one_line :)")
    return result


def load_text_file_as_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    print("load_text_file_as_list_pre: ---------------------------------------------------")
    print(str(file_content))
    tokens=file_content.split(" ")
    line = []
    lines=[]
    current_token = ""

    for token in tokens:
        if token == "":
            current_token += ' '  # Concatenate spaces
        else:
            if current_token != "":
                line.append(current_token)  # Add accumulated spaces
                current_token = ""  # Reset for the next non-empty token
            if "\n" not in token:
                line.append(token)
            if "\n" in token:
                result = [match for match in re.split(r'(\n|\s+)', token) if match != '']
                for elem in result:
                    if elem=="\n":
                        lines.append(line)
                        line=[]
                    else:
                        line.append(elem)
            
    lines.append(line)
    print("load_text_file_as_list_post: ---------------------------------------------------")
    print(str(lines))
    return lines

    
def save_list_as_text_file(file_path, text):
    with open(file_path, 'w', encoding='utf-8') as file: 
        file.write(text)
    print(f"The text has been saved as a text file at: {file_path}")


    """def read_input_text(input_text):
    print("read_input_text_pre: ---------------------------------------------------")
    print(str(input_text))
    tokens=input_text.split(" ")
    line = []
    lines=[]
    current_token = ""

    for token in tokens:
        if token == "":
            current_token += ' '  # Concatenate spaces
        else:
            if current_token != "":
                line.append(current_token)  # Add accumulated spaces
                current_token = ""  # Reset for the next non-empty token
            if "\n" not in token:
                line.append(token)
            if "\n" in token:
                result = [match for match in re.split(r'(\n|\s+)', token) if match != '']
                for elem in result:
                    if elem=="\n":
                        lines.append(line)
                        line=[]
                    else:
                        line.append(elem)
            
    lines.append(line)
    print("read_input_text_post: ---------------------------------------------------")
    print(str(input_text))
    return lines"""