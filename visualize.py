import spacy
from spacy import displacy

ex = [{'text': 'But Google is starting from behind.',
       'ents': [{'start': 4, 'end': 10, 'label': 'ORG'}],
       'title': None}]

html = displacy.serve(ex, style='ent', manual=True)
