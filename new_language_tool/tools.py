# nlp tools

import spacy
nlp = spacy.load("en_core_web_sm")

from new_language_tool.params import *

def singularize_word(word):
    for suffix, singular_suffix in C_SINGULAR_SUFFIX:
        if word.endswith(suffix):
            return word[:-len(suffix)] + singular_suffix
    return word

def parse_sentence_into_spatial_relation(sentence:str):
    sentence = sentence.lower()
    doc = nlp(sentence)
    
    subject = None
    relation = None
    object = None

    subject_desc = {}
    object_desc = {}
    
    case = 0
    subject_token = None
    relation_token = None
    object_token = None
    # step one: find PREP
    for token in doc:
        if token.dep_ == "prep":
            if token.text in ["on","behind","beside","near","at"]:     
                relation_token = token
                relation =  relation_token.text
                case = 1
                break
            elif token.text == "in":
                if not "in front of" in sentence:
                    relation_token = token
                    relation =  relation_token.text
                    case = 1
                else:
                    child = list(token.children)[0]
                    if child.text == "front":
                        relation_token = list(child.children)[0] # of
                        relation = "in front of"
                        case = 2
            elif token.text == "to":
                if token.head.text == "close":
                    relation_token = token # to
                    relation = "close to"
                    case = 2
            elif token.text == "from":
                if token.head.text == "away":
                        relation = "away from"
                        relation_token = token # from
                        case = 2
                    
    # print("case:", case)
    ## CASE zero: object only (I prefer to have two shelfs.)
    if case == 0:
        relation = "in"
        object = "__room__"
        for token in doc:
            if "obj" in token.dep_ or ("there" in sentence and token.pos_ == "NOUN") \
                or ("nsubj" in token.dep_ and ("necessary" in sentence or "for" in sentence)):
                subject_token = token
                subject = subject_token.text
                break

    ## CASE ONE: s,p,o  (e.g.some flowers should be on the colorful sofa)
    elif case == 1:
        # step two: find object
        object_token = list(relation_token.children)[0]   
        object = object_token.text
        assert("obj" in object_token.dep_)

        # step three: find subject
        subject_token = relation_token.head
        if subject_token.pos_ == "NOUN":
            subject = subject_token.text
        else:
            if subject_token.text in ["are","is", "be"]:
                for child in subject_token.children:
                    if "subj" in child.dep_ or ("there" in sentence and "attr" in child.dep_):
                        subject = child.text
                        subject_token = child
                        break
            else: # need
                for child in subject_token.children:
                    if "obj" in child.dep_:
                        subject = child.text
                        subject_token = child
                        break

    
    ## CASE TWO: in front of
    elif case == 2:
        object_token = list(relation_token.children)[0] # of -> ?
        object = object_token.lemma_
        assert("obj" in object_token.dep_)

        if relation in ["in front of"]:
            subject_token = relation_token.head.head.head # ? -> in -> front -> of
        elif relation in ["close to", "away from"]:
            subject_token = relation_token.head.head # ? -> close -> to
        else:
            raise("CASE NOT AVAILABLE: {}".format(relation))
        #print(relation_token, relation, "2", object_token)
        #print("sub", subject_token)
        if subject_token.pos_ == "NOUN":
            subject = subject_token.lemma_
        else:
            if subject_token.text in ["are","is", "be"]:
                for child in subject_token.children:
                    if "subj" in child.dep_ or ("there" in sentence and "attr" in child.dep_):
                        subject = child.text
                        subject_token = child
                        break
            else: # need
                for child in subject_token.children:
                    if "obj" in child.dep_:
                        subject = child.text
                        subject_token = child
                        break
        
    # Add description
    if object_token != None:
        for desc_child in object_token.children:
            if desc_child.dep_ == "det":
                object_desc["det"] = desc_child.text
            elif desc_child.dep_ == "amod":
                object_desc["amod"] = desc_child.text
            elif desc_child.dep_ == "compound":
                object_desc["compound"] = desc_child.text

    if subject_token != None:
        for desc_child in subject_token.children:
            if desc_child.dep_ == "det":
                subject_desc["det"] = desc_child.text
            elif desc_child.dep_ == "amod":
                subject_desc["amod"] = desc_child.text
            elif desc_child.dep_ == "nummod":
                subject_desc["nummod"] = desc_child.text 
            elif desc_child.dep_ == "compound":
                subject_desc["compound"] = desc_child.text 

    ## modfiy: (('tables', 'close to', 'other'), ({'nummod': 'Two'}, {'det': 'each'}))
    if object == "other": #and "nummod" in subject_desc and subject_desc["nummod"].lower() == "two":
        object = subject
        subject_desc["nummod"] = "one"
        object_desc = subject_desc

    ## singlarize word
    subject = singularize_word(subject)
    object = singularize_word(object)

    return (subject, relation, object),(subject_desc, object_desc)


def parse_text_into_spatial_relations(text:str):
    all_sentences = []
    for sentence in text.strip().split('.'):
        sentence = sentence.strip().lower()
        if ',' in sentence: # (e.g. I need two beds, a coffee table, and some flowers.)
            # split "there are" into several sentence
            subsentences = sentence.split(",")
            for i in range(len(subsentences)):
                if i == 0:
                    all_sentences.append(subsentences[0].strip())
                else:
                    subsentences[i] = subsentences[i].replace("and","")
                    all_sentences.append("I need " + subsentences[i].strip())
        else:
            all_sentences.append(sentence)

    all_relations_with_attributes = []
    for sentence in all_sentences:
        if len(sentence) == 0:
            continue
        print("sentence:",sentence)
        all_relations_with_attributes.append(parse_sentence_into_spatial_relation(sentence))

    return all_relations_with_attributes
