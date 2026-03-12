import requests
import csv
from bs4 import BeautifulSoup

chart_api_url = "https://genius.com/api/songs/chart"
params = {
    'time_period': 'all_time', 
    'chart_genre': 'all',      
    'per_page': 50,            
    'page': 1                  
}

song_data = [["Song Title", "Artist", "Song URL", "Lyrics"]]
amount = 100  
count = 0

print("กำลังดึงข้อมูลเพลงฮิตตลอดกาลจากเว็บ Genius อยู่คับ 🤖")
print("."*67)
print("รอเเป็ปนะคับ! รอไม่นานคับ 😆")

while count < amount:
    response = requests.get(chart_api_url, params=params)
    
    if response.status_code != 200:
        print(f"ดึงข้อมูลไม่สำเร็จคับ เกิดจาก ===> {response.status_code}")
        break

    data = response.json()
    chart_items = data.get('response', {}).get('chart_items', [])
    
    if not chart_items:
        print("หาเพลงใน charts ไม่เจออีกเเล้วคับ!")
        break
        
    for item in chart_items:
        if count >= amount:
            break
            
        song_info = item.get('item', {})
        
        title = song_info.get('title', 'Unknown Title')
        artist = song_info.get('primary_artist', {}).get('name', 'Unknown Artist')
        song_url = song_info.get('url', 'No URL')
        
        lyrics = "หาเนื้อเพลงไม่เจอคับ!"
        if song_url != 'ไม่มี URL คับ!':
            try:          
                lyric_res = requests.get(song_url)
                if lyric_res.status_code == 200:
                    soup = BeautifulSoup(lyric_res.text, 'html.parser')
                    lyrics_divs = soup.find_all('div', attrs={'data-lyrics-container': 'true'})
                    if lyrics_divs:
                        lyrics = "\n".join([div.get_text(separator="\n", strip=True) for div in lyrics_divs])
            except Exception as e:
                print(f"ดึงเนื้อเพลงจากเพลง {title}: {e} ไม่สำเร็จคับ")

        song_data.append([title, artist, song_url, lyrics])
        count += 1
        print(f"[{count}/{amount}] ดึงข้อมูลเพลงเรียบร้อย! ===> {title} โดย {artist}")

    params['page'] += 1


filename = 'Genius_Top_Chart_Songs.csv'
with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerows(song_data)
    
print(f"ดึงข้อมูลเสร็จเรียบร้อยคับ 👌 มีข้อมูลเพลงทั้งหมด {count} เพลงที่เป็นเพลงฮิตตลอดกาลคับ!")
print("."*67)
print(f"ข้อมูลเพลงจะถูกจัดเก็บไว้ในไฟล์ชื่อ ===> '{filename}'นะคับ! 👍")
print("."*67)

