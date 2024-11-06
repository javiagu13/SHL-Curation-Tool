
import random
import re
import zipfile
import os

#DEIDENTIFIER: Basic Dates Functions
dateErrors="If there are errors here. please correct your formatting to fit yyyy.mm.dd or yy.mm.dd format, either with '.', '-' or '/' symbols: \n"
substitutedDates="This are the subsitutions of dates performed: \n"
maskedText="This is the list of tags that have been masked: \n"

#txt saver auxiliary function
def save_string_to_file(input_string, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(input_string)


#### Date String to Array: 23.03.01 -> ['23', '03', '01', '.']
#As an input it recieves a date as follows: (it basically separates the number and stores the separation at the end)
# 03/01
# 23.03.01
# 23-03-04
#The results are as follows
# ['03', '01', '/']
# ['23', '03', '01', '.']
# ['23', '03', '04', '-']
def dateToNumericParser(date):
    parsedDate=date
    if "." in date:
        parsedDate=date.split(".")
        parsedDate.append(".")
    elif "/" in date:
        parsedDate=date.split("/")
        parsedDate.append("/")
    elif "-" in date:
        parsedDate=date.split("-")
        parsedDate.append("-")
    return parsedDate

#dateToNumericParser('23-03-04')


#### Date Deidentifier Calculator: ['23', '03', '01', '.'] -> + 5 -> false, 03, 06
#As an input the month and day and the range of days to add randomly. If you choose 3 days randomly it will choose to add 1, 2 or 3 days
#As output the new month and day will be given 
def dayChanger(month, day, randDayAdd):
    daysToAdd=random.randint(1,randDayAdd)
    #print("adding days: "+str(daysToAdd))
    newMonth=month
    newDay=day
    yearChange=False
    if month in [1,3,5,7,8,10,12]: #if the month has 31 days (1:january, 3: march... 12:december)
        if (day+daysToAdd) > 31:
            if month!=12: # this if checks whether its december to make next month 1 not 13
                newMonth=month+1
            else:
                yearChange=True
                newMonth=1
            newDay=day+daysToAdd-31 # this calculation is for the jump of month to start from 0
        else:
            newDay=day+daysToAdd
    elif month==2: #if the month is 2:february
        if (day+daysToAdd) > 28:
            if month!=12: # this if checks whether its december to make next month 1 not 13
                newMonth=month+1
            else:
                yearChange=True
                newMonth=1
            newDay=day+daysToAdd-28 # this calculation is for the jump of month to start from 0
        else:
            newDay=day+daysToAdd
    else: #if the month has 30 days (which are the rest)
        if (day+daysToAdd) > 30:
            if month!=12: # this if checks whether its december to make next month 1 not 13
                newMonth=month+1
            else:
                yearChange=True
                newMonth=1
            newDay=day+daysToAdd-30 # this calculation is for the jump of month to start from 0
        else:
            newDay=day+daysToAdd
    return yearChange, newMonth, newDay




#dayChanger(7, 31, 3)





#### Date Deidentifier Calculator: ['23', '03', '01', '.'] -> + 5 -> ['23', '03', '06', '.']
#input is the array with the original date, output is the new array (it will add x amount of days randomly) randDayAdd will tell the limit of days to add
def dateArrayDeidentifier(array, randDayAdd):
    try:
        if len(array)==4: #this means full date was found
            yearChange,month,day=dayChanger(int(array[1]),int(array[2]),randDayAdd)
            if(yearChange==True):
                array[0]=int(array[0])+1
            else:
                array[0]=int(array[0])
            array[1]=month
            array[2]=day
        elif len(array)==3:#this means just month and year or month and day were found
            if int(array[0])<12: #it means the first number is a month and the second number is a day: MONTH AND DAY
                yearChange,month,day=dayChanger(int(array[0]),int(array[1]),randDayAdd)
                array[0]=month
                array[1]=day
            #else: #it means that first number is >12 therefore it cannot be a month it must be a year and the second variable month: YEAR AND MONTH
        #elif len(array)<3: #Special case probably someone handwrote 12일 01월 or similar (choose policy)
    except:
        print("THERE HAS BEEN AN ERROR TRYING TO CONVERT: "+str(array))
        global dateErrors
        dateErrors+="THERE HAS BEEN AN ERROR TRYING TO CONVERT: "+str(array)+"\n"
    return array


#### MAIN PROCEDURE, USES ALL PREVIOUS CODE: Date Deidentifier String Version: 23.03.01 -> + 5 -> 23.03.06
def deidentifierStringToString(originalDate, randDayAdd):
    parsedDate=dateToNumericParser(originalDate)
    if isinstance(parsedDate, list):
        array=dateArrayDeidentifier(parsedDate, randDayAdd)
        result=""
        for elem in array[0:-1]:
            result+=str(elem)+str(array[-1])
        result=result[0:-1]
    else:
        result=originalDate
    print(str(originalDate)+" ---> "+ result)
    global substitutedDates
    substitutedDates+=str(originalDate)+" ---> "+ result+"\n"
    return result
    


#print("REGULAR CASES TEST:")
#result=deidentifierStringToString('23-11-30',5)
#result=deidentifierStringToString('23.11.30',5)
#result=deidentifierStringToString('23-11',5)
#result=deidentifierStringToString('10/11',5)

#print("")
### EXTREME CASES (JUST 120 out of 24500)
#print("EXTREME CASES TEST:")
#result=deidentifierStringToString('10 11 12',5)
#result=deidentifierStringToString('2013년 5월',5)
#result=deidentifierStringToString('202209',5)
#result=deidentifierStringToString('3년11개월11일',5)
#result=deidentifierStringToString('2017 09 22',5)
#result=deidentifierStringToString('OCT 16, 2018',5)


## DEIDENTIFIER: Basic Masking Functions
def mask_labels(text, replacement_pairs):
    global maskedText
    for i in range(0,len(replacement_pairs),2):
        maskedText+="The following pattern: <"+replacement_pairs[i]+">ANY TEXT</"+replacement_pairs[i]+"> Has been subsituted by the following mask: "+replacement_pairs[i+1]+"\n"
        print("The following pattern: <"+replacement_pairs[i]+">ANY TEXT</"+replacement_pairs[i]+"> Has been subsituted by the following mask: "+replacement_pairs[i+1]+"\n")
    for i in range(0, len(replacement_pairs), 2):
        old_word = replacement_pairs[i]
        new_word = replacement_pairs[i + 1]
        pattern = rf'<{old_word}>(.*?)</{old_word}>'
        text = re.sub(pattern, " "+new_word+" ", text)

    return text

# Example usage
#input_text = "<NAME>adam aronson</NAME> a <gender>man</gender> was heading to <address>east side manhattan</address>."
#replacements = ['NAME', 'MASKED_NAME', 'GENDER', 'MASKED_GENDER', 'ADDRESS', 'MASKED_ADDRESS']
#masked_text = mask_labels(input_text, replacements)
#print(masked_text)




def extract_date(text):
    # Define the regex pattern to match content within <DATE> tags
    pattern = r'<DATE>(.*?)</DATE>'

    # Use re.search to find the pattern in the text
    match = re.search(pattern, text)

    # If a match is found, return the content inside the tags
    if match:
        return match.group(1)
    else:
        return None


def deidentify_dates_in_text(text, randDayAdd):
    # Define a regular expression to match date placeholders
    date_pattern = re.compile(r'<DATE>.*?</DATE>')

    # Iterate over matches in the text
    for date_match in date_pattern.finditer(text):
        # Extract the matched date placeholder
        date_placeholder = date_match.group(0)
        original_date=extract_date(date_placeholder)
        deidentified_date = deidentifierStringToString(original_date, randDayAdd)
        # Replace the date placeholder with the new date
        text = text.replace(date_placeholder, " "+deidentified_date+" ", 1)
    global dateErrors
    global substitutedDates
    global maskedText
    return text, dateErrors, substitutedDates, maskedText


# Example usage
#text = "Today is <DATE>23.02.1</DATE>. Tomorrow will be <DATE>2023/01</DATE>. <DATE>2023.02.1</DATE>"
#date_array = ["23-01-01", "2023-01-02", "a"]

#result = deidentify_dates_in_text(text, 5)
#print(result)





def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), 
                           os.path.relpath(os.path.join(root, file), 
                                           os.path.join(folder_path, '..')))


