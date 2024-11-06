#For this code to work make sure the following dependencies are installed
#!python -m spacy download en_core_web_sm
#!pip install spacy
import re
import spacy
import json
import string
######################################## PARSER 1 - TEXT TAGGER <TAG> </TAG> GIVEN A TEXT ########################################

def regular_expression_tagger(text, tag_name, token_to_find):
    # Escape special characters in the ID for accurate matching
    escaped_token_to_find = re.escape(token_to_find)

    # Create the pattern to match the ID in the text
    pattern = r'(?<!\w)' + escaped_token_to_find + r'(?!\w)'

    # Replace the matched ID with the wrapped version
    text = re.sub(pattern, r'<'+tag_name+'>\g<0></'+tag_name+'>', text)

    return text


#USAGE
#text="hola me llamo javi"
#tag_name="PER"
#token_to_find="javi"
#tagged_text = regular_expression_tagger(text, tag_name, token_to_find)
#print(tagged_text)



######################################## PARSER 2 - <TAG> to BIO_seq ########################################

def parse_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    parsed_text = []
    bio_labels = []
    
    currentBIO=""
    
    currentLabel=""
    
    for token in doc:

        if token.text.startswith("@@/"):
            currentBIO=""
            currentLabel=""
        elif token.text.startswith("@@"):
            label = token.text.strip("@@")
            currentLabel=label
            currentBIO="B"
        elif (currentBIO=="B" or currentBIO=="I") and not token.text.startswith("@@"):
            parsed_text.append(token.text)
            bio_labels.append(currentLabel + "-" + currentBIO)
            currentBIO="I"
        else:
            #print(token.text)
            parsed_text.append(token.text)
            bio_labels.append("O")
    
    return " ".join(parsed_text)+"\t"+" ".join(bio_labels)

def wrap_labels(listOflabels):
    wrapped_list=[]
    wrapped_target_list=[]
    for label in listOflabels:
        wrapped_label = "@@" + label + "@@"
        wrapped_target_list.append(wrapped_label)
        wrapped_label = "<" + label + ">"
        wrapped_list.append(wrapped_label)
    for label in listOflabels:
        wrapped_label = "@@/" + label + "@@"
        wrapped_target_list.append(wrapped_label)
        wrapped_label = "</" + label + ">"
        wrapped_list.append(wrapped_label)
    return wrapped_list, wrapped_target_list

def replace_date_string(text, listOflabels, listOfTargetlabels):
    replaced_text=text
    for i in range(0,len(listOflabels)):
        replaced_text = replaced_text.replace(listOflabels[i], listOfTargetlabels[i])
    return replaced_text


######################################## BIO to <TAG> ########################################

import re

def parse_sentence_with_bio(sentence, tags):
    parsed_sentence = ""
    inside_tag = False
    current_label = None

    for word, tag in zip(sentence.split(), tags.split()):
        if tag.startswith("B-"):
            if inside_tag:
                if(parsed_sentence[-1]==" "):
                    parsed_sentence=parsed_sentence[:-1]
                parsed_sentence += f"</{current_label}>"
            current_label = re.sub(r"^B-", "", tag)
            parsed_sentence += f"<{current_label}>"
            parsed_sentence += f"{word} "
            inside_tag = True
        elif tag.startswith("I-"):
            if current_label == re.sub(r"^I-", "", tag):
                parsed_sentence += f"{word} "
            else:
                if inside_tag:
                    if(parsed_sentence[-1]==" "):
                        parsed_sentence=parsed_sentence[:-1]
                    parsed_sentence += f"</{current_label}>"
                current_label = re.sub(r"^I-", "", tag)
                parsed_sentence += f"<{current_label}>"
                parsed_sentence += f"{word} "
                inside_tag = True
        else:
            if inside_tag:
                if(parsed_sentence[-1]==" "):
                    parsed_sentence=parsed_sentence[:-1]
                parsed_sentence += f"</{current_label}>"
                inside_tag = False
            parsed_sentence += f"{word} "

    if inside_tag:
        if(parsed_sentence[-1]==" "):
            parsed_sentence=parsed_sentence[:-1]
        parsed_sentence += f"</{current_label}>"

    return parsed_sentence.strip()


# Example usage
sentence = "the best of james was that he went to the university of arizona"
tags = "O O O B-PER O O O O O O B-ORG I-ORG I-ORG"

parsed_sentence = parse_sentence_with_bio(sentence, tags)
print(parsed_sentence)

######################################## Predictions to BIO ########################################
##Call this function from predToTAG if the data is as follows [SMC:OTHER_HOSPITAL-B] and call bio_to_tagged_no_BI if it doesnt have the B or I as follows [SMC:OTHER_HOSPITAL]
def bio_to_tagged(input_string, label):
    # Define the regex pattern to match the BIO tags
    pattern = r"\[(.+?):(\w+-\w+)\]"
    
    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, input_string)
    
    # Replace the BIO tags with the tagged format
    tagged_string = re.sub(pattern, r"<\2>\1</\2>", input_string)
    #simplified_tag_1=tagged_to_simplified_tags(tagged_string, label)
    simplified_tag_1=tagged_to_simplified_tags(tagged_string, label)
    simplified_tag_2=simplify_tags(simplified_tag_1, label)
    return simplified_tag_2

def bio_to_tagged_no_BI(input_string, label):
    # Define the regex pattern to match the BIO tags
    pattern = r"\[(.+?):(\w+)\]"
    
    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, input_string)
    
    # Replace the BIO tags with the tagged format
    tagged_string = re.sub(pattern, r"<\2>\1</\2>", input_string)
    simplified_tag_1=tagged_to_simplified_tags_no_BI(tagged_string, label)
    return simplified_tag_1
    
#THIS CODE MODIFIES FROM 
#This: <PERSON-B>james</PERSON-B> <PERSON-I>stuart</PERSON-I> <PERSON-I>anderson</PERSON-I> is one of my best friends
#to This: <PERSON-B>james stuart anderson</PERSON-I> is one of my best friends
def tagged_to_simplified_tags(input_string, label):
    # Define the regex pattern to match the BIO tags
    pattern = r"\[([\w\s]+):(\w)-(\w+)\]"
    
    # Replace the BIO tags with the tagged format for the given label
    tagged_string = re.sub(pattern, lambda match: f"<{label}>{match.group(1)}</{label}>" if match.group(3) == label else match.group(0), input_string)
    
    # Remove duplicate closing and opening tags
    duplicate_pattern = f"</{label}-\w+>\s*<{label}-\w+>"
    tagged_string = re.sub(duplicate_pattern, ' ', tagged_string)
    
    return tagged_string

def tagged_to_simplified_tags_no_BI(input_string, label):
    # Define the regex pattern to match the BIO tags
    pattern = r"\[([\w\s]+):(\w)-(\w+)\]"
    
    # Replace the BIO tags with the tagged format for the given label
    tagged_string = re.sub(pattern, lambda match: f"<{label}>{match.group(1)}</{label}>" if match.group(3) == label else match.group(0), input_string)
    
    # Remove duplicate closing and opening tags
    duplicate_pattern = f"</{label}>\s*<{label}>"
    tagged_string = re.sub(duplicate_pattern, ' ', tagged_string)
    
    return tagged_string

#THIS CODE MODIFIES FROM 
#This: <PERSON-B>james stuart anderson</PERSON-I> is one of my best friends
#to This: <PERSON>james stuart anderson</PERSON> is one of my best friends
def simplify_tags(input_string, label):
    # Define the regex pattern to match the tags
    opening_pattern = r"<" + label + r"-B>"
    opening_pattern2 = r"<" + label + r"-I>"
    closing_pattern = r"</" + label + r"-I>"
    closing_pattern2 = r"</" + label + r"-B>"
    
    # Replace the opening and closing tags with the simplified tags
    simplified_string = re.sub(opening_pattern, f"<{label}>", input_string)
    simplified_string = re.sub(closing_pattern, f"</{label}>", simplified_string)
    simplified_string = re.sub(opening_pattern2, f"<{label}>", simplified_string)
    simplified_string = re.sub(closing_pattern2, f"</{label}>", simplified_string)
    
    return simplified_string


######################################## Equal Length Tester ########################################
def checkLength(sentence, tags):
    sentence = sentence.split()
    tags = tags.split()
    print(len(sentence))
    print(len(tags))
    return len(sentence)==len(tags)


######################################## this is for the prediction moment it transforms each line to BIO ######################################
def linesInTAGtoBIOseq(lines,label_lst):
    #Javi code: it transfor the <TAG> into BIO if needed, hiding this code will allow BIO loading directly
    inputLabels=[]
    for label in label_lst:
        if label!="O" or label!="UNK":
            inputLabels.append(label[:-2])
    #inputLabels2=["DATE_YMD", "OTHER_HOSPITAL", "TIME_KHMP", "TIME_KHP", "DATE_MD", "DATE_YM", "NAME", "DATETIME_YMDKHP", "DATETIME_YMDKHMP", "PERIOD_YMD2MD", "PERIOD_YMD2YMD", "PERIOD_YMD2D", "PERIOD_MD2MD", "PERIOD_MD2D", "DATETIME_MDKHP", "DATE_YMD", "DATE_YM", "OTHER_HOSPITAL", "TIME_KHMP", "DATE_MD", "NAME", "DATETIME_YMDKHP", "DATETIME_YMDKHMP", "PERIOD_YMD2MD", "PERIOD_YMD2YMD", "DATETIME_YMDKHKP", "PERIOD_YMD2D", "PERIOD_MD2MD", "DATETIME_MDKHP", "TIME_KHP", "PERIOD_MD2D", "DATETIME_MDKHMP", "TIME_KHKP", "DATE_YM", "DATE_MDY", "ADDRESS", "DATE_YMD", "PERIOD_KHP2KHP", "PERIOD_KHP2KHP"]
    listOflabels,listOfTargetlabels=wrap_labels(inputLabels)
    for i in range(0,len(lines)):
        lines[i]=replace_date_string(lines[i], listOflabels, listOfTargetlabels)
    return lines


def predToTAG(line,label_lst):
    #Javi code: it transfor the <TAG> into BIO if needed, hiding this code will allow BIO loading directly
    inputLabels=[]
    for label in label_lst:
        line=bio_to_tagged_no_BI(line,label)
    return line


def remove_duplicates_and_special(input_list):
    special_values = {"O", "UNK"}
    unique_list = []
    seen = {}

    for i in range(0,len(input_list)):
        if input_list[i] not in special_values:
            input_list[i]=input_list[i][:-2]

    for item in input_list:
        if item not in special_values and item not in seen:
            unique_list.append(item)
            seen[item] = True
    return unique_list



######################################## USAGE: examples ########################################

########################################Parser 1 - Regular expression tagger
###TEXT TAGGER <TAG> </TAG> GIVEN A TEXT
#text="hola me llamo javi"
#tag_name="PER"
#token_to_find="javi"
#tagged_text = regular_expression_tagger(text, tag_name, token_to_find)
#print(tagged_text)


########################################Parser 2 - <TAG> to BIO
#inputLabels=["DATE_YMD", "OTHER_HOSPITAL", "TIME_KHMP", "TIME_KHP", "DATE_MD", "DATE_YM", "NAME", "DATETIME_YMDKHP", "DATETIME_YMDKHMP", "PERIOD_YMD2MD", "PERIOD_YMD2YMD", "PERIOD_YMD2D", "PERIOD_MD2MD", "PERIOD_MD2D", "DATETIME_MDKHP", "DATE_YMD", "DATE_YM", "OTHER_HOSPITAL", "TIME_KHMP", "DATE_MD", "NAME", "DATETIME_YMDKHP", "DATETIME_YMDKHMP", "PERIOD_YMD2MD", "PERIOD_YMD2YMD", "DATETIME_YMDKHKP", "PERIOD_YMD2D", "PERIOD_MD2MD", "DATETIME_MDKHP", "TIME_KHP", "PERIOD_MD2D", "DATETIME_MDKHMP", "TIME_KHKP", "DATE_YM", "DATE_MDY", "ADDRESS", "DATE_YMD", "PERIOD_KHP2KHP", "PERIOD_KHP2KHP"]
#here you change your string
#listOflabels,listOfTargetlabels=wrap_labels(inputLabels)
#csv_string=replace_date_string("<DATE_YMD> 19.20.30 </DATE_YMD> is the date he born", listOflabels, listOfTargetlabels)
#modifiedString = re.sub(r'\s+', ' ', csv_string)
#parse_text(csv_string)
#csv_string
#TO WORK IT MUST CALL FIRST LABEL SPACE WHATEVER SPACE LABEL, FIX THIS!

########################################Parser 3 - BIO to <TAG>
#sentence = "nausea / vomiting \, F/70rnrnlocal 의원에서 골다공증 \, 고지혈증 약 처방받아 복용해옴 . rnrn그외 특이병력없던 환자로 \, rnrn 19.03.01   버스에서 내리던 중 걸려 넘어지며 인도에 오른쪽 광대뼈 부딪혔다함 . LOC(-)rn          당시 정형외과에서 X - ray 만 찍었으며 뼈에는 이상 없다는 소견 들었다함 . rn          머리쪽 검사는 시행하지 않았다함 . rnrn 19.03.07 부터 Lt.facial pain \, headache(찌릿한 양상 ) \, skin rash 있어rn 19.03.08   local 피부과 진료후 대상포진 진단받고 항바이러스제 복용 및 cream 발라옴 . rnrn지속되는 headache에 대해 further evaluation 원해rnrn 19.03.09   ER로 내원함 . rn          Lt.side로는 headache 및 facial pain 현재는 없다고 함.rn          brain \, facial bone CT 촬영 후 귀가rnrn 19.03.10   보호자 진술에 의하면 환자 morphine 투여 후 nausea sense 급히 악화됨.rn          환자 Lt side zoster 병변 진행되며 좌안 안구 충혈 발생하였고rn          HA \, N / V 지속되어 내원 . \,M / S alertrn     GCS 4/5/6 / rn     orientation : T / P / P ( + /+/+)rnCNE isocoric pupilrn     PRL ( + /+ ) promptrn     EOM : intact rn     Nystagmus ( -)rn     Facial motor    : intactrn            sensory : intactrn     Motor V / Vrn           V / Vrn     Sensory 100/100rn             100/100rnrnneck stiffness(-)\,Nausea and vomitingrn \, \, \, \, \, \, \, \, \, \, \, \,"
#tags = "O O O O O O O O O O O O O O O O O O B-DATE_YMD O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O B-DATE_YMD O O O O O O O O O O O B-DATE_YMD O O O O O O O O O O O O O O O O O O B-DATE_YMD O O O O O O O O O O O O O O O O O O O O O O O B-DATE_YMD O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O O"
#parse_sentence_with_bio(sentence, tags)


########################################Parser 4 - Predictions to <TAG>
#input_string = "[james:PERSON-B] [stuart:PERSON-I] [anderson:PERSON-I] is one of my best friends"
#output_string = bio_to_tagged(input_string,"PERSON")
#print(output_string)

##############################CODE FOR THOR STYLE JSON PARSING######################
def transform_to_labeled_string(array1, array2):
    # Join the words to form a sentence
    sentence = ' '.join(array1)

    # Initialize an empty list for labels
    labels = []

    # Initialize a variable to keep track of the character position
    char_position = 0

    # Iterate over the words and their labels
    for word, label in zip(array1, array2):
        if label != "O":
            # Calculate the start and end positions of the word
            start = char_position
            end = char_position + len(word) - 1
            labels.append([start, end+1, label])
        
        # Update the character position (add 1 for the space)
        char_position += len(word) + 1

    # Construct the final string
    #result = '{"text": '+'"'+str(sentence)+'"'+', "label": '+str(labels)+'}'
    result = json.dumps({"text": sentence, "label": labels}, ensure_ascii=False)

    return result

#This function receives an array of arrays and performs the labeling in all of them
#it adds \n when necessary to skip line
def transform_arOfar_to_labeled_string(arrayAr1, arrayAr2):
    print("arOfar_pre: ---------------------------------------------------")
    print(str(arrayAr1))
    # Initialize a variable to keep track of the character position
    char_position = 0
    # Initialize an empty list for labels
    labels = []
    sentence=""
    for i in range(0,len(arrayAr1)):
        array1=arrayAr1[i]
        array2=arrayAr2[i]
        # Join the words to form a sentence
        sentence += ' '.join(array1)

        # Iterate over the words and their labels
        for word, label in zip(array1, array2):                
            if label != "O":
                # Calculate the start and end positions of the word
                start = char_position
                end = char_position + len(word) - 1
                labels.append([start, end+1, label])
                
            # Unless it is an empty array always add the length of the word +1 for the space
            if array1 !=[]:
                if word.isspace():
                    char_position += len(word)
                else:
                    char_position += len(word) + 1
        #this will add a intro if and update char position if there is an empty array (jump line)
        if array1==[]:
            sentence+="\n"
            char_position += 1
        else: #this one simply adds always jump line ( +1 is no necessary since it is always done +1 before for spaces)
            sentence+="\n"
    # Construct the final string
    #result = '{"text": '+'"'+str(sentence)+'"'+', "label": '+str(labels)+'}'
    result = json.dumps({"text": sentence, "label": labels}, ensure_ascii=False)
    print("arOfar_post: ---------------------------------------------------")
    print(str(result))
    return result

    def concatenateSentenceExcludingSpaces(words):
        sentence = ''
        for i in range(0,len(words)):
            print(sentence)
            word=words[i]
            if word.strip():  # Check if the word is not just spaces
                if i < len(words) - 1 and words[i + 1].isspace(): # If next word is a space dont concatenate one previously
                    print("entre1")
                    sentence += word 
                elif i < len(words) - 1 and words[i + 1] in string.punctuation:#here check if it is a punctuation mark: 
                    print("entre2")
                    sentence += word  
                else:
                    print("entre3")
                    sentence += word + ' '
            else:
                print("entre4")
                sentence += word

        # Remove trailing spaces
        sentence = sentence.strip()
        return sentence