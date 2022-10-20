import json
import math
import os
import re
from collections import Counter

import demoji
import textstat
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from tqdm import tqdm

demoji.download_codes()
translator = Translator()

USER_FEATURES_PATH = 'user_features_no_extra.json'
PHOTO_FEATURES_PATH = 'photo_features_no_extra'


def compute_entropy(data, unit='natural'):
    base = {
        'shannon': 2.,
        'natural': math.exp(1),
        'hartley': 10.
    }

    if len(data) <= 1:
        return 0

    counts = Counter()
    for d in data:
        counts[d] += 1

    ent = 0
    probs = [float(c) / len(data) for c in counts.values()]
    for p in probs:
        if p > 0.:
            ent -= p * math.log(p, base[unit])

    return ent


def clean_comment(text):
    clean_text = text.lower()  # to lowercase
    clean_text = re.sub(r"<[^>]+>", '', clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"&[^;]+;", '', clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r"\n", ' ', clean_text, flags=re.MULTILINE)
    clean_text = demoji.replace_with_desc(clean_text)

    stops = set()  # stopwords.words("english")) | set(stopwords.words("spanish"))# | set(stopwords.words("dutch")) | set(stopwords.words("french")) | set(stopwords.words("german")) | set(stopwords.words("italian")) | set(stopwords.words("portuguese")) | set(stopwords.words("romanian"))

    words = clean_text.split(" ")
    filtered_words = [
        word for word in words if word not in stops and word.isalpha()]
    clean_text = ' '.join(filtered_words)

    clean_text = ''.join([y for y in clean_text if str.isalnum(y) or y == " "])

    return clean_text


def is_relevant_user(occupation):
    words = ["fot", "phot", "valokuv", "zdjęcie", "dealbh", "bild", "grianghraf",
             "nuotrauk", "pictur", "myndin", "billed", "ljósmyndari", "ritratt"]
    return any(w in occupation.lower() for w in words)

################################################

print("Reading user features...")
with open(USER_FEATURES_PATH, "r", encoding="utf8") as f:
    user_features = json.load(f)

print("Labelling user features...")
for user in user_features:
    user['is_photographer'] = is_relevant_user(user['occupation'])

print("Reading photo features...")
data = []
for filename in tqdm(os.listdir(PHOTO_FEATURES_PATH)):
    full_path = os.path.join(PHOTO_FEATURES_PATH, filename)
    #simple_name = filename.split("\\")[-1][:-5]
    with open(full_path, "r", encoding="utf8") as f:
        data.extend(json.load(f))
print(data[0].keys())
print(len(data))

n_comments = 0
n_final_comments = 0
max_iter = len(data)
for i, row_by_img in enumerate(data):

    if i % (len(data)//100) == 0:
        print(f"{i} of {len(data)}")

    # delete empty comments
    filtered_comments = []
    n_comments += len(row_by_img["comments"])
    row_by_img["comments"] = [
        c for c in row_by_img["comments"] if c["comment"]]
    for comment in row_by_img["comments"]:
        comment["comment"] = clean_comment(comment["comment"])
        comment_text = comment["comment"]

        try:
            # translator.detect(comment_text) # "en"
            detected_lang = translator.detect(comment_text)
            if isinstance(detected_lang, list):
                detected_lang = detected_lang[0]
            # if detected_lang in top_languages:
            comment["lang"] = detected_lang
        except Exception as e:
            print(comment_text)
            print(e)
            continue
        if comment_text:
            filtered_comments.append(comment)
    row_by_img["main_language"] = detected_lang
    row_by_img["comments"] = filtered_comments
    n_final_comments += len(filtered_comments)

default_subj_pola = 0.5
default_read = 0
comments_list = []
sia = SentimentIntensityAnalyzer()

for x in data:
    comments = [y["comment"] for y in x["comments"] if y["comment"]]
    avg_subj = default_subj_pola
    avg_diff_words = default_read
    avg_read_time = default_read
    avg_entropy = default_read
    avg_length = default_read
    avg_polarity = default_read
    sum_subj = 0
    sum_diff_words = 0
    sum_read_time = 0
    sum_entropy = 0
    sum_length = 0
    sum_polarity = 0

    for comment in comments:
        testimonial = TextBlob(comment)
        sum_subj += testimonial.sentiment.subjectivity
        sum_diff_words += textstat.difficult_words(comment)
        sum_read_time += textstat.reading_time(comment, ms_per_char=14.69)
        sum_entropy += compute_entropy(comment)
        sum_length += len(comment)

        sum_polarity += sia.polarity_scores(comment).get("compound")

        comments_list.append(comment)

    if comments:
        avg_subj = sum_subj / len(comments_list)
        avg_diff_words = sum_diff_words / len(comments_list)
        avg_read_time = sum_read_time / len(comments_list)
        avg_entropy = sum_entropy / len(comments_list)
        avg_length = sum_length / len(comments_list)
        avg_polarity = sum_polarity / len(comments_list)

    x["avg_subj"] = avg_subj
    x["avg_diff_words"] = avg_diff_words
    x["avg_read_time"] = avg_read_time
    x["avg_entropy"] = avg_entropy
    x["avg_length"] = avg_length
    x["avg_polarity"] = avg_polarity

# save new files with secondary features for photos
for x in data:
    file_name = x["owner"]
    path = f'photo_features_no_extra_secondary/{file_name}.json'
    with open(path, "w") as outfile:
        json.dump(x, outfile)
