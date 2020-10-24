import requests
from bs4 import BeautifulSoup

# # get the data
# url = 'https://it.iiita.ac.in/?pg=faculty'
# data = requests.get(url)
# html = data.text

# soup = BeautifulSoup(html, 'html.parser')
# div = soup.find('div', { 'id': 'tooplate_content' })

# table = div.find('table')

# for tr in table.find_all('tr'):
# 	td_list = tr.find_all('td')
# 	if len(td_list) > 1:
# 		for td in td_list:
# 			txt = td.text.rstrip()
# 			txt.replace('\n', '')
# 			if(txt):
# 				values = [t.strip() for t in txt.split('\n')]
# 				print(values[0])
# 				if(values[1]):
# 					print(values[1])
# 				print(values[2])
# 				print(values[3])
# 				print()

# target URL to scrap
url = "https://www.goibibo.com/hotels/hotels-in-shimla-ct/"
response = requests.get(url)
data = BeautifulSoup(response.text, 'html.parser')
cards_price_data = data.find_all('div', attrs={'class', 'HotelCardstyles__CurrentPriceTextWrapper-sc-1s80tyk-25 lnAxKT'})
cards_url_data = data.find_all('div', attrs={'class', 'HotelCardstyles__HotelNameWrapperDiv-sc-1s80tyk-11 jkwhbV'})
hotel_price = {}
hotel_url = {}
for i in range(0, len(cards_price_data)):
	hotel_price[cards_url_data[i].text.rstrip] = cards_price_data[i].text.rstrip()
	hotel_url[cards_url_data[i].text.rstrip] = 'https://www.goibibo.com' + cards_url_data[i].find('a', href = True)['href']
	print('https://www.goibibo.com' + cards_url_data[i].find('a', href = True)['href'])