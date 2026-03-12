import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

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

stop_words = english_stopwords.union(custom_lyrics_stopwords)

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if not retrieved_k: return 0.0
    relevant_and_retrieved = set(retrieved_k).intersection(set(relevant))
    return len(relevant_and_retrieved) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if not relevant: return 0.0
    relevant_and_retrieved = set(retrieved_k).intersection(set(relevant))
    return len(relevant_and_retrieved) / len(relevant)

def average_precision(retrieved, relevant):
    if not relevant: return 0.0
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(retrieved):
        if p in relevant:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(relevant)

filename = 'Genius_Top_Chart_Songs_Cleaned.csv'
try:
    df = pd.read_csv(filename, encoding='utf-8-sig')
    df['Lyrics'] = df['Lyrics'].fillna('')
    
except FileNotFoundError:
    print(f"{filename} หาชื่อไฟล์ไม่เจออะคับ")
    exit()


vectorizer = TfidfVectorizer()
df['Search_Text'] = (df['Song Title'].astype(str) + " ") * 20 + df['Lyrics'].astype(str)

tfidf_matrix = vectorizer.fit_transform(df['Search_Text'])

feature_names = vectorizer.get_feature_names_out()
inverted_index = {}

for col_idx, term in enumerate(feature_names):
    doc_indices = tfidf_matrix[:, col_idx].nonzero()[0]
    postings_list = [f"Doc{doc_id}" for doc_id in doc_indices]
    inverted_index[term] = {
        'DF': len(doc_indices),
        'Postings': postings_list
    }

def expand_query(clean_query, top_local_docs=3, top_expansion_terms=2):
    """
    Local Context Analysis (Pseudo-Relevance Feedback)
    """
    if len(clean_query.split()) == 1:
        return clean_query # ถ้าหาคำเดียวเดี่ยวๆ ไม่ต้องทำ Expansion
    # 1. ค้นหาเอกสารเริ่มต้นเพื่อสร้าง Local Context
    query_vec = vectorizer.transform([clean_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # ดึง Index ของเอกสารที่ตรงที่สุด (top_local_docs)
    top_indices = similarity_scores.argsort()[-top_local_docs:][::-1]
    
    # [🌟 ส่วนที่แก้ไข 🌟] กรองเอาเฉพาะเอกสารที่เกี่ยวข้องจริงๆ (Score ต้องมากกว่า 0)
    valid_indices = [idx for idx in top_indices if similarity_scores[idx] > 0]
    
    # ถ้าไม่มีเอกสารที่ตรงเลย ให้คืนค่าคำเดิมกลับไป
    if not valid_indices:
        return clean_query
        
    # 2. รวมน้ำหนัก TF-IDF เฉพาะในกลุ่ม Local Context ที่ผ่านเงื่อนไข
    local_tfidf_sum = np.sum(tfidf_matrix[valid_indices], axis=0)
    local_tfidf_sum = np.squeeze(np.asarray(local_tfidf_sum))
    
    # 3. เรียงลำดับคำและสกัดคำขยาย (Expansion Terms)
    original_terms = set(clean_query.split())
    top_term_indices = local_tfidf_sum.argsort()[::-1]
    
    expansion_terms = []
    feature_names_array = vectorizer.get_feature_names_out()
    
    for idx in top_term_indices:
        term = feature_names_array[idx]
        if term not in original_terms and term not in stop_words:
            expansion_terms.append(term)
        
        if len(expansion_terms) >= top_expansion_terms:
            break
            
    # 4. ประกอบร่าง Query เดิม เข้ากับคำขยาย
    if expansion_terms:
        expanded_query = clean_query + " " + " ".join(expansion_terms)
        return expanded_query
        
    return clean_query

N = 10

def search_songs(original_query, top_n=N):
    expanded_q = expand_query(original_query)
    if expanded_q != original_query.lower():
        print(f"\n   [Query Expansion] : '{expanded_q}'")

    query_vec = vectorizer.transform([expanded_q])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        score = similarity_scores[idx]
        results.append({
            'Index': idx,
            'Title': df.iloc[idx]['Song Title'],
            'Artist': df.iloc[idx]['Artist'],
            'Score': score,
            'URL': df.iloc[idx]['Song URL'],
            'Cluster': df.iloc[idx]['Cluster_ID'] + 1 
            })
    return results

print("กำลังจักลุ่มด้วยการใช้ K-Means Clustering.....")
true_k = 7
kmeans_model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=0)
kmeans_model.fit(tfidf_matrix)
df['Cluster_ID'] = kmeans_model.labels_
order_centroids = kmeans_model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()
print("ผลของการทำ Clustering ที่ได้คับ")


all_top_words = []
for i in range(true_k):
    top_words = [terms[ind] for ind in order_centroids[i, :7]]
    all_top_words.append(top_words)

col_width = 18
headers = [f"🎵 คลัสเตอร์ {i+1}" for i in range(true_k)]
header_str = " |  ".join([h.ljust(col_width) for h in headers])
print(header_str)
print("-" * len(header_str))

for row_words in zip(*all_top_words):
    row_str = " | ".join([word.ljust(col_width) for word in row_words])
    print(row_str)
print("." * len(header_str))

session_history = []

while True:
    print("\nหวัดดีค้าบ ขอต้อนรับเข้าสู่.. ค้นเพลงฮิตสุดตลอดการจาก Genius! 🥳")
    print("(หากต้องการออก สามารถกด q หรือ exit ได้เลยครับ )")
    user_query = input("\nป้อนคำค้นหาได้เลยคับ! ⌨️   ===> ")
    
    if user_query.lower() in ['q', 'exit', 'quit']:
        print("\nขอบคุณสำหรับการใช้งานนะคับ บ๊ายบายจั้ฟ 🙃")
        break
        
    if not user_query.strip():
        print("เหมือนคำค้นหาจะยังไม่ถูกน้า ลองใหม่คับ 🧐")
        continue

    clean_text = re.sub(r'[^a-z\s]', ' ', user_query.lower())
    clean_query_words = []
    for w in clean_text.split():
        if w not in stop_words:
            clean_query_words.append(lemmatizer.lemmatize(w))

    clean_query = " ".join(clean_query_words)
    
    if not clean_query.strip():
        print("ป้อนคำที่ถูกต้องทีคับ เดี๋ยวหาไม่เจอน้า 🤯")
        continue

    if clean_query != user_query.lower():
        print(f"Clean query : '{clean_query}'")

    print('\n')
    print(f"Inverted Index ===> '{clean_query}'")
    print("." * 128)
    print(f"{'Term':<15} | {'DF':<5} | {'Postings List '}")
    print("." * 128)

    query_terms = clean_query.split()
    for term in query_terms:
        if term in inverted_index:
            df_count = inverted_index[term]['DF']
            postings = inverted_index[term]['Postings'][:10] 
            postings_str = ", ".join(postings)
            if df_count > 10:
                postings_str += ", ..."
            print(f"{term:<15} | {df_count:<5} | [{postings_str}]")
        else:
            print(f"{term:<15} | 0     | [หาไม่เจอคับ]")
    print("." * 128)

    # ... (โค้ดส่วนปริ้นท์ Inverted Index ก่อนหน้านี้) ...
    print("." * 128)

    # ==========================================
    # 🌟 โค้ดส่วน [Initial Search Results] ที่ปรับแก้แล้ว 🌟
    # ==========================================
    print("\n   🔍 [Initial Search Results (ก่อนทำ Query Expansion)] :")
    init_query_vec = vectorizer.transform([clean_query])
    init_sim = cosine_similarity(init_query_vec, tfidf_matrix).flatten()
    top_10_init = init_sim.argsort()[-10:][::-1]
    
    found_init = False
    for i, idx in enumerate(top_10_init):
        if init_sim[idx] > 0:
            found_init = True
            # เปลี่ยน 'Recipe Title' เป็น 'Song Title' และเพิ่ม 'Artist'
            title = df.iloc[idx]['Song Title']
            artist = df.iloc[idx]['Artist']
            print(f"      [{i+1}] {title} โดย {artist} (Score: {init_sim[idx]:.4f})")
            
    if not found_init:
        print("      - ไม่พบผลลัพธ์เริ่มต้น (หาไม่เจอเลยจั้ฟ) -")

    print("-" * 67)
    # ==========================================

    # จากนั้นระบบก็จะทำการค้นหาแบบทำ Query Expansion ตามปกติ
    search_results = search_songs(clean_query, top_n=N)
    if search_results:
        print(f"\n  เจอเเล้ว!! {len(search_results)} เพลงที่ตรงกับคำค้นหา (หลังทำ Expansion) ===> '{user_query}' ")
        # ... (โค้ดแสดงผลและประเมินผลตามเดิม) ...

    search_results = search_songs(clean_query, top_n=N)
    if search_results:
        print(f"\n  เจอเเล้ว!! {len(search_results)} เพลงที่ตรงกับคำค้นหา ===> '{user_query}' ")
        for i, result in enumerate(search_results):
            print(f"[{i+1}] {result['Title']} โดย {result['Artist']}")
            print(f"     ===>  ค่าความคล้ายคลึง: {result['Score']:.4f} | หมวดหมู่ : Cluster {result['Cluster']} | URL: {result['URL']}")

        expanded_eval_query = expand_query(clean_query)
        eval_pattern = r'\b(?:' + '|'.join(expanded_eval_query.split()) + r')\b'

        relevant_docs = df[
            df['Song Title'].str.contains(eval_pattern, case=False, na=False) | 
            df['Lyrics'].str.contains(eval_pattern, case=False, na=False)
        ].index.tolist()
            
        retrieved_indices = [res['Index'] for res in search_results]
        retrieved_k = retrieved_indices[:N] 
        relevant_set = set(relevant_docs)   
        
        # True Positive (TP)
        tp_set = set(retrieved_k).intersection(relevant_set)
        TP = len(tp_set)
        
        # False Positive (FP)
        FP = len(retrieved_k) - TP
        
        # False Negative (FN)
        FN = len(relevant_set) - TP

        p_score = precision_at_k(retrieved_indices, relevant_docs, N)
        r_score = recall_at_k(retrieved_indices, relevant_docs, N)
        ap_score = average_precision(retrieved_indices, relevant_docs)

        session_history.append({
            'query': user_query,
            'p': p_score,
            'r': r_score,
            'ap': ap_score
        })

        hits = 0
        sum_precisions = 0.0
        print('\n')
        for i, doc_idx in enumerate(retrieved_indices):
            rank = i + 1
            if doc_idx in relevant_docs:
                hits += 1
                p_at_rank = hits / rank 
                sum_precisions += p_at_rank
                print('.'*67)
                print(f"   ลำดับที่ {rank}: ตรงกันเเฮะ! (Found {hits} items) -> P_{rank} = {hits}/{rank} = {p_at_rank:.2f}")
                print('.'*67)
            else:
                print(f"   ลำดับที่ {rank}: อาจจะยังน้า!")
                print('.'*67)
        
        total_relevant = len(relevant_docs)
        if total_relevant > 0:
            print(f"\n   ผลรวมของ Precisions (Relevant Documents) = {sum_precisions:.2f}")
            print(f"   หารด้วยจำนวนทั้งหมดของ Relevant Documents = {total_relevant} items")
            print(f"    AP = {sum_precisions:.2f} / {total_relevant} = {ap_score:.2f}")
        else:
            print(f"\n AP = 0.00")
        print('.'*67)
        print(f"   [+] True Positive  (TP) ได้ {TP}  ")
        print('.'*67)
        print(f"   [-] False Positive (FP) ได้ {FP}  ")
        print('.'*67)
        print(f"   [-] False Negative (FN) ได้ {FN}  ")
        print("\n" + "."*67)
        print(f"Precision_{N}: {p_score:.2f} | Recall_{N}: {r_score:.2f} | Average Precision: {ap_score:.2f}")
        print("" + "."*67)
        print("\n")
        
        ap_list = []
        print("Search Query    | Precision    | Recall     | Average Precision")
        print("." * 67)
        for history in session_history:
            print(f"{history['query']:<15} | {history['p']:<12.2f} | {history['r']:<10.2f} | {history['ap']:.2f}")
            ap_list.append(history['ap'])
        print("." * 67)
        
        map_score = np.mean(ap_list)
        print(f"ค่า MAP (Mean Average Precision) ได้ {map_score:.2f}")
    else:
        print(f"\n  หาผลลัพธ์ของ '{user_query}' ไม่เจออะคับ 😵‍💫")
        
    print("." * 67)
    print("." * 67)