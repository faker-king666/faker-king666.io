import urllib.request
import json

def get_random_joke():
    url = "https://official-joke-api.appspot.com/random_joke"
    print("正在从 API 获取随机笑话...\n" + "-"*30)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            setup = data.get('setup', '')
            punchline = data.get('punchline', '')
            print(f"🤖 铺垫: {setup}")
            print(f"😂 包袱: {punchline}")
            print("-" * 30)
    except Exception as e:
        print(f"获取笑话失败，原因: {e}")

if __name__ == "__main__":
    get_random_joke()
