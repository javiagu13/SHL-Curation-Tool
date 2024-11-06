### Running Guide
Simply run it by using: (make sure you installed requirements.txt of each pipeline and the one on inferenceAPI first) 
```
python auto_annotate.py
```

About 24-02-23 version:
BERT pipeline has been built to annotate PHI automatically with BERT
REGEX pipelien has been built to annnotate PHI automatically using REGEX (it may make a little mistake if two words are stuck together, future issue to solve)

In order to connect it to doccano:
### Auto-labeling setup

url:
```
http://127.0.0.1:7000/auto_annotate_BERT_PHI
or
http://127.0.0.1:7000/auto_annotate_REGEX_PHI
```
method:
```
POST
```
body:
```
text
{{ text }}
```


```
[
    {% for entity in input %}
        {
            "start_offset": {{ entity.start_offset }},
            "end_offset": {{ entity.end_offset}},
            "label": "{{ entity.label }}"
        }{% if not loop.last %},{% endif %}
    {% endfor %}
]
```
  
