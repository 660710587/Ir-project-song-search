import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

input_filename = 'Genius_Top_Chart_Songs.csv'
output_filename = 'Genius_Top_Chart_Songs_Cleaned.csv'

try:
    df = pd.read_csv(input_filename, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"ไฟล์ที่ชื่อ {input_filename} หาไม่เจออะคับ.")
    exit()

lemmatizer = WordNetLemmatizer()
english_stopwords = set(stopwords.words('english'))

custom_lyrics_stopwords = {
    'verse', 'chorus', 'intro', 'outro', 'bridge', 'hook', 'pre', 'interlude', 
    'vamp', 'spoken', 'skit', 'part', 'instrumental', 'vocals', 'vocal', 'beat',
    'refrain', 'post','lyrics', 'contributor', 'contributors', 'read', 'more', 
    'embed', 'cancel', 'share', 'copy', 'url', 'genius', 'com', 'translation', 
    'translations', 'romanized', 'click','produced', 'producer', 'written', 
    'writer', 'feat', 'featuring', 'ft','nederlands', 'thai', 'svenska', 'russian',
    'portugu', 'português', 'deutsch', 'english', 'polski', 'rk', 'italiano', 'hindi',
    'fran', 'ais', 'espa', 'ol', 'cymraeg', 'az', 'rbaycan', 'hebrew', 'srpski', 'rom',
    'esky', 'sterreichisches', 'zbek', 'simplified', 'chinese', 'korean', 'japanese',
    'ooh', 'ahh', 'yeah', 'yeh', 'woo', 'woah', 'ayy', 'ay', 'uh', 'huh', 'la', 'da', 
    'na', 'oh', 'hey', 'yo', 'boy', 'girl', 'man', 'bitch', 'nigga', 'niggas',
    'got', 'get', 'like', 'know', 'go', 'let', 'make', 'see', 'say', 'tell', 'come',
    'cause', 'cuz', 'wanna', 'gonna', 'gotta', 'ain', 'don', 'll', 've', 're', 'm'
}

all_stopwords = english_stopwords.union(custom_lyrics_stopwords)

def clean_title(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text) 
    tokens = word_tokenize(text) 
    
    cleaned_tokens = []
    for token in tokens:
        if token not in english_stopwords and len(token) > 1:
            lemma = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)
    return ' '.join(cleaned_tokens)

def clean_lyrics(text): 
    text = str(text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'\(.*?\)', ' ', text)
    
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text) 
    tokens = word_tokenize(text)

    cleaned_tokens = []
    for token in tokens:
        if token not in all_stopwords and len(token) > 1:
            lemma = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)
    return ' '.join(cleaned_tokens)

print("กำลังคลีนนิ่งข้อมูลนะคับ ... รอแป็ปนึงคับ")
print("."*67)

df['Cleaned Title'] = df['Song Title'].apply(clean_title)
df['Cleaned Lyrics'] = df['Lyrics'].apply(clean_lyrics)
df_final = df[['Cleaned Title', 'Artist', 'Song URL', 'Cleaned Lyrics']]
df_final = df_final.rename(columns={'Cleaned Title': 'Song Title', 'Cleaned Lyrics': 'Lyrics'})

df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"คลีนนิ่งข้อมูลเพลงเสร็จแล้วนะคับ บันทึกเป็นไฟล์ชื่่อ '{output_filename}'นะคับ!")
print("."*67)
print("ขอบคุณคับ😁")