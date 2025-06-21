import requests
from bs4 import BeautifulSoup
import pandas as pd
from collections import defaultdict
import re
import os

# Create output directory
os.makedirs("data", exist_ok=True)

# Kanda configuration
kandas = [
    {"name": "Bala Kanda", "filename": "valmiki_ramayan_bala_kanda_book1.txt", "base_url": "https://valmikiramayan.net/utf8/baala/sarga{s}/balasans{s}.htm", "total_chapters": 77},
    {"name": "Ayodhya Kanda", "filename": "valmiki_ramayan_ayodhya_kanda_book2.txt", "base_url": "https://valmikiramayan.net/utf8/ayodhya/sarga{s}/ayodhyasans{s}.htm", "total_chapters": 119},
    {"name": "Aranya Kanda", "filename": "valmiki_ramayan_aranya_kanda_book3.txt", "base_url": "https://valmikiramayan.net/utf8/aranya/sarga{s}/aranyasans{s}.htm", "total_chapters": 75},
    {"name": "Kishkindha Kanda", "filename": "valmiki_ramayan_kishkindha_kanda_book4.txt", "base_url": "https://valmikiramayan.net/utf8/kish/sarga{s}/kishkindhasans{s}.htm", "total_chapters": 67},
    {"name": "Sundara Kanda", "filename": "valmiki_ramayan_sundara_kanda_book5.txt", "base_url": "https://valmikiramayan.net/utf8/sundara/sarga{s}/sundarasans{s}.htm", "total_chapters": 68},
    {"name": "Yuddha Kanda", "filename": "valmiki_ramayan_yuddha_kanda_book6.txt", "base_url": "https://valmikiramayan.net/utf8/yuddha/sarga{s}/yuddhasans{s}.htm", "total_chapters": 128}
]

# Scrape one kanda
def scrape_kanda(book_name, filename, base_url, total_chapters):
    verses_data = []
    content_dict = defaultdict(list)

    for chapter_num in range(1, total_chapters + 1):
        url = base_url.format(s=str(chapter_num))
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            tat_paragraphs = soup.find_all("p", class_="tat")

            for verse_idx, p in enumerate(tat_paragraphs, 1):
                verse_text = p.get_text(strip=True)
                verse_number = None

                prev_sibling = p.find_previous_sibling()
                if prev_sibling:
                    prev_text = prev_sibling.get_text(strip=True)
                    match = re.search(r'(\d+)', prev_text)
                    if match:
                        verse_number = match.group(1)

                if not verse_number:
                    match = re.match(r'^(\d+)[\.\s]', verse_text)
                    if match:
                        verse_number = match.group(1)
                        verse_text = re.sub(r'^\d+[\.\s]*', '', verse_text)

                if not verse_number:
                    verse_number = str(verse_idx)

                content_key = verse_text.strip()
                content_dict[content_key].append((chapter_num, verse_number))

                verses_data.append({
                    'Book': book_name,
                    'Chapter': chapter_num,
                    'Verse_Number': verse_number,
                    'Content': verse_text,
                    'Content_Key': content_key
                })

        except:
            continue

    # Remove duplicates
    final_data = []
    seen = set()

    for verse in verses_data:
        key = verse['Content_Key']
        if key in seen:
            continue
        seen.add(key)
        occurrences = content_dict[key]

        if len(occurrences) > 1:
            chapters = [str(c[0]) for c in occurrences]
            verses = [str(c[1]) for c in occurrences]
            final_data.append({
                'Book': book_name,
                'Chapter': ', '.join(chapters),
                'Verse_Number': ', '.join(verses),
                'Content': verse['Content']
            })
        else:
            final_data.append({
                'Book': book_name,
                'Chapter': verse['Chapter'],
                'Verse_Number': verse['Verse_Number'],
                'Content': verse['Content']
            })

    # Save TXT file
    txt_path = f"data/{filename}"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{book_name}\n{'='*50}\n\n")
        for row in final_data:
            f.write(f"Chapter: {row['Chapter']}\n")
            f.write(f"Verse: {row['Verse_Number']}\n")
            f.write(f"Content: {row['Content']}\n")
            f.write("-" * 40 + "\n\n")

    return final_data

# Main script
all_data = []

for kanda in kandas:
    kanda_data = scrape_kanda(kanda["name"], kanda["filename"], kanda["base_url"], kanda["total_chapters"])
    all_data.extend(kanda_data)

# Save master CSV
df = pd.DataFrame(all_data)
df.to_csv("data/Valmiki_Ramayana_Master.csv", index=False, encoding='utf-8')

# Download supplementary file
supplementary_url = "https://raw.githubusercontent.com/shashwath1278/IYD_hack/main/data/valmiki_ramayan_supplementary_knowledge.txt"
supplementary_path = "data/valmiki_ramayan_supplementary_knowledge.txt"

try:
    supp_res = requests.get(supplementary_url)
    supp_res.raise_for_status()
    with open(supplementary_path, "w", encoding="utf-8") as f:
        f.write(supp_res.text)
except Exception as e:
    print("Failed to download supplementary file:", e)
