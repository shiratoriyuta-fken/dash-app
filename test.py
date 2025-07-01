from icrawler.builtin import GoogleImageCrawler
import os

def download_images_with_crawler(keyword, max_num, save_dir):
    """
    icrawlerを使って指定されたキーワードの画像をダウンロードする
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    print(f"Downloading {max_num} images for '{keyword}'...")
    crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
    crawler.crawl(keyword=keyword, max_num=max_num)
    print("Download complete.")

# --- 設定 ---
# 犬の画像を10枚ダウンロード
dog_keyword = '犬'
dog_save_dir = './images/dog'
num_images_to_download = 10

# 猫の画像を10枚ダウンロード
cat_keyword = '猫'
cat_save_dir = './images/cat'

# --- 実行 ---
download_images_with_crawler(dog_keyword, num_images_to_download, dog_save_dir)
download_images_with_crawler(cat_keyword, num_images_to_download, cat_save_dir)

print("\nAll tasks finished.")
print(f"Dog images are saved in: {dog_save_dir}")
print(f"Cat images are saved in: {cat_save_dir}")
