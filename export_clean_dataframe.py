#!/usr/local/bin/python
# -*- coding: utf-8
import pymysql
import os
import sys
import gc
import re
import pytils
import string
import pandas as pd

from pymystem3 import Mystem
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

reload(sys)
sys.setdefaultencoding('utf-8')

#REGEXPS
re0  = re.compile(ur'[:;_—.,!?§ї±©®*@#$%^&()`\'“”]|[+=]|[[]|[]]|[/]|"|\s{2,}|\-{2,}|\n')
re2 = re.compile(r'<.+?>') # HTML tags
re4 = re.compile(r"((www\.[^\s]+)|(https?://[^\s]+))") #URLs
re6 = re.compile(r'([' + string.punctuation + ']){1,}') # punctuation
re8 = re.compile(ur"(\S+)-{1,}\b", re.UNICODE) # word-
re10 = re.compile(ur"([^0-9]+?)4([^0-9]+?)\b", re.UNICODE) #word4word
re12 = re.compile(ur"\B-{1,}(\S+)\b", re.UNICODE) #-word
re14 = re.compile(r'\s\d{5,}?\s') #deleting digits 5 or more
re16 = re.compile(ur"((дата:\s\d{2}\.\d{2}.\d{4}\s\d{2}:\d{2}:\d{2})|(оценка:\s\d{1,2})|(отправить\sсообщение)|(автор:\s(.+?)\s))", re.UNICODE) # dates\scores
re18 = re.compile(ur"\b4(\w+?)\b", re.UNICODE) #4word
re20 = re.compile(ur"([а-жзк-мор-я])\1{2,}", re.UNICODE) #wooorrd -> word
re22 = re.compile(ur"([а-я])\1{2,}", re.UNICODE) #wooorrd -> word
re24 = re.compile(r'-(.)\s') #word-s\s -> word
re26 = re.compile(r'\s{1,}-\s{1,}') # " one - two " -> "one-two"
re28 = re.compile(ur'(\S{4,}?)\s{0,}-\s{0,}(\S{3,}.?)')
re30 = re.compile(r'\s{1,}-(\S+?)\s')
re32 = re.compile(ur'\s{1,}(\S{1,2})-(\S|\D+?)\s') # " to-any " -> "any"
re34 = re.compile(ur"\b([a-zа-я]+?)\d\b",re.UNICODE) # word1 -> word 1
re36 = re.compile(ur"(?:\b(\w{1,2})-(.+?)\b)",re.UNICODE)   #to-word ->  to word
re38 = re.compile(ur"(?:\b([a-zа-я]{5,})-(.+?)\b)",re.UNICODE) #abcdef-word -> abcdef word
re40 = re.compile(ur"\b(цитата)(.+?)\b",re.UNICODE) #quoteword -> quote word
reUSP = re.compile(r"\r|\n|\t|\x0b|\x0c|\x1c|\x1d|\x1e|\x1f|\s|\x85|\xa0|\u1680|\u180e|\u2000|\u2001|\u2002|\u2003|\u2004|\u2005|\u2006|\u2007|\u2008|\u2009|\u200a|\u2028|\u2029|\u202f|\u205f|\u3000|\xe2|\u2800") #unicode spaces
reSP = re.compile(r'\s{2,}') #spaces
reEN = re.compile(r'[a-z]')
#reENW = re.compile(r'\b([a-z]{2,}).*?\b') #words
reENW = re.compile(r'\b([a-z_.-]{2,}).*?\b', re.IGNORECASE)
reEML = re.compile(r"([a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)") #email
reSML = re.compile(r"(?:<.+?([a-z.0-9_]+?\.gif).+?>)")
def clean_text_pre(text):
    text = text.lower()
    text = text.replace("\\"," ").replace("&quot;"," ").replace("&nbsp;"," ").replace("&gt;"," ").replace("&lt;"," ").replace("&amp;"," ").replace("\"r"," ").replace("\"n"," ").replace(u'—','-').replace(u'…','.')
    text = text.replace(u'№', ' ').replace(u'«',' ').replace(u'»',' ').replace('[quote]',' ').replace('[/quote]',' ')
    text = reSML.sub(lambda x: x.group(1).upper(), text)
    text = re2.sub(' ', text)
    text = re4.sub(' URL ', text)
    text = reEML.sub(r' MAIL ', text)
    text = re6.sub(r'\1', text)
    text = re8.sub(r' \1 ', text)
    text = re10.sub(ur' \1ч\2 ', text)
    text = re12.sub(r' \1 ', text)
    text = re14.sub(' ', text)
    text = re16.sub(' ', text)
    text = re18.sub(ur' ч\1 ', text)
    text = re20.sub(r'\1', text)
    text = re22.sub(r'\1', text)
    text = re24.sub(r' \1 ', text)
    text = re26.sub('-', text)
    text = re28.sub(r'\1 \2', text)
    text = re30.sub(r'-\1', text)
    text = re32.sub(r' \1 ', text)
    text = re34.sub(r' \1 ', text)
    text = re36.sub(ur' \1 \2 ', text)
    text = re38.sub(ur' \1 \2 ', text)
    text = re40.sub(ur' \1 \2 ', text)
    #text = reENW.sub(r' \1 ', text)
    return text

def clean_text_post(text, tils_enable=True):
    if tils_enable:
       #iterate all matches and translit word by word
       for m in re.finditer(reENW, text):
           txt = m.group(1).strip()
           if txt != 'URL' and txt != 'MAIL' and txt.find('.GIF') == -1:
              text = text.replace(txt, pytils.translit.detranslify(txt))
           elif txt.find('.GIF') != -1:
              text = text.replace('.GIF', ' ')
    text = re0.sub(' ', text)
    text = text.decode("utf-8")
    text = reUSP.sub(' ', text)
    text = reSP.sub(' ', text)
    return text

def lemmatize_ya(text):
    text = ''.join(ma_ya.lemmatize(text))
    text = re.sub('\n', '', text)
    text = reUSP.sub(' ', text)
    text = reSP.sub(' ', text)
    text = text.encode("utf-8")
    return text

def tokenize_ru(text):
    tokens = word_tokenize(text) 
    tokens = [i for i in tokens if (i not in string.punctuation)]
    return tokens


def clean_text(text):
    txt = [tokenize_ru(sent) for sent in sent_tokenize(clean_text_pre(text), 'russian')]
    txt = clean_text_post(" ".join(sum(txt, [])))
    return txt

print "Fetch data"
conn = pymysql.connect(host='127.0.0.1', port=3306, user='user', passwd='pwd', db='dnn', charset='utf8mb4')
df = pd.read_sql("SELECT * FROM pages_corpus", conn)
conn.close()
print "Done"

print "Cleanup"
df['comment'] = df.apply(lambda x: clean_text(x['comment']), axis=1)
df = df[(df['comment'] != "")] # filter empty
df.to_pickle('dataframe_ver_6.pkl')
print df.tail(50)
