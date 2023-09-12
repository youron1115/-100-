from bs4 import BeautifulSoup
html='<h2 class="block-title">今日最新</h2>'
soup = BeautifulSoup(html, 'lxml')
soup.get_text() # 得到結果為 '今日最新'