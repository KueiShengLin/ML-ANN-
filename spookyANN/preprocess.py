import pandas as pd
import string
import re
import nltk


def write_csv(list, word_list, name):
    list_dict = {}
    # list_dict['id'] = list['id']
    # author = list['author']
    for num, word in enumerate(word_list):
        if (num+1)%100 == 0:
            print(str(num+1) + '/' + str(len(word_list)))
        new_word_list = []
        for l_id, l in enumerate(list['text']):
            ls = l.split(' ')
            if word in ls:
                haveword = sum(1 for lw in ls if word == lw)
                new_word_list.append(haveword)
            else:
                new_word_list.append(0)
        list_dict[word] = new_word_list
    list_dict['_author'] = list['author']
    df = pd.DataFrame(list_dict)
    df.to_csv(name + '.csv', encoding='utf-8', index=False)


train = pd.read_csv("dataset2/train.csv")
# print(train['text'][0])

RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])   # 標點符號表

train['text'] = train['text'].str.replace(RE_PUNCTUATION, "")   # 去標點符號
train['text'] = train['text'].str.lower()   # 大轉小
# train.to_csv('train_nopun.csv', encoding='utf-8', index=False)

all_words = train['text'].str.split(expand=True).unstack().value_counts()
print(len(all_words))
print('train_split')
# write_csv(train, all_words, 'train_split')

print('train_clean')
stopwords = nltk.corpus.stopwords.words('english')
all_words_clean = [word for word in all_words.index.values if word not in stopwords]    # 去掉 a, of, the, ...
print(len(all_words_clean))
# write_csv(train, all_words_clean, 'train_clean')

print('top30')
eap = train[train.author == "EAP"]["text"].str.split(expand=True).unstack().value_counts()
eap = [word for word in eap.index.values if word not in stopwords]
hpl = train[train.author == "HPL"]["text"].str.split(expand=True).unstack().value_counts()
hpl = [word for word in hpl.index.values if word not in stopwords]
mws = train[train.author == "MWS"]["text"].str.split(expand=True).unstack().value_counts()
mws = [word for word in mws.index.values if word not in stopwords]

top100 = eap[0:100]
for w in hpl[0:100]:
    if w not in top100:
        top100.append(w)
for w in mws[0:100]:
    if w not in top100:
        top100.append(w)
write_csv(train, top100, 'top100')

# lemm = nltk.stem.WordNetLemmatizer()
# all_words_clean_format = [lemm.lemmatize(word) for word in all_words_clean]
# print("The lemmatized form of leaves is: {}".format(lemm.lemmatize("leaves")))
# print(all_words_clean_format)

# print(train)


#
