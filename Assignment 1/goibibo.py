import requests
import pandas as pd  
from bs4 import BeautifulSoup

#getting country url's
hotelname_list = []
city_list = []
countries_list = []
rating_list = []
prince_list = []
Amenities_list = []
HotelDescription_list = []
Review1_list = []
Review2_list = []

hotel_name = ""
city_name = ""
country_name = ""
ratingl = ""
pricel = ""
amenities = ""
descriptionl = ""
review1l = ""
review2l = ""

url = 'https://www.goibibo.com/destinations/intl/all-countries/'
data = requests.get(url)
html = data.text
soup = BeautifulSoup(html, 'html.parser')
cards = soup.find_all('div', {'class' : 'col-md-4 col-sm-4 col-xs-12 filtr-item posrel'})
# print('Found' , len(cards) , 'countries')
country_urls = []
country_names = []
for card in cards :
	for a in card.find_all('a', href=True):
		if a['href']:
			country_urls.append(a['href'])
			country_names.append(a.text.rstrip())

# getting all cities in a country.
length = len(country_urls)
for i in range(length):
	url = country_urls[i]
	country_name = country_names[i]
	data = requests.get(url)
	html = data.text
	soup = BeautifulSoup(html, 'html.parser')
	places_to_visit = soup.find('div', {'class' : 'place-to-visit-container'})
	if(places_to_visit):
		card = places_to_visit.find('div', {'class' : 'col-md-12'})
		city_urls = {}
		for a in card.find_all('a', href=True):
			if a['href']:
				list = a['href'].split('/')
				city_urls[list[3]] = 'https://www.goibibo.com/hotels/intl-hotels-in-' + list[3] + '-ct/'

		for city in city_urls:
			print(city)
			city_name = city
			#getting all hotels in a city along with their price
			url = city_urls[city]
			response = requests.get(url)
			data = BeautifulSoup(response.text, 'html.parser')
			cards_price_data = data.find_all('div', attrs={'class', 'HotelCardstyles__CurrentPriceTextWrapper-sc-1s80tyk-25 lnAxKT'})
			cards_url_data = data.find_all('div', attrs={'class', 'HotelCardstyles__HotelNameWrapperDiv-sc-1s80tyk-11 jkwhbV'})
			hotel_price = {}
			hotel_url = {}
			for i in range(0, len(cards_price_data)):
				hotel_price[cards_url_data[i].text.rstrip()] = cards_price_data[i].text.rstrip()
				hotel_url[cards_url_data[i].text.rstrip()] = 'https://www.goibibo.com' + cards_url_data[i].find('a', href = True)['href']

			#getting details of a hotel
			for i in range(0, len(cards_price_data)):
				url = hotel_url[cards_url_data[i].text.rstrip()]
				data = requests.get(url)
				html = data.text
				hotel_name = cards_url_data[i].text.rstrip()
				pricel = hotel_price[cards_url_data[i].text.rstrip()]
				# print('Name : ' + cards_url_data[i].text.rstrip())
				# print('Cost : ' + hotel_price[cards_url_data[i].text.rstrip()])

				soup = BeautifulSoup(html, 'html.parser')
				div = soup.find('div', { 'id': 'root' })
				description = div.find('section', {'class' : 'HotelDetailsMain__HotelDetailsContainer-sc-2p7gdu-0 kuBApH'})
				address = description.find('span', {'itemprop' : 'streetAddress'}).text.rstrip().replace(' View on Map', '')
				# print('Address : ' + address) #contains address
				descriptionl = address

				rating = 'Rating not found'
				ratingdata = description.find('span', {'itemprop' : 'ratingValue'}) #contains rating
				if ratingdata:
					rating = ratingdata.text.rstrip()
				# print('Rating : ' + rating)
				ratingl = rating

				review1 = 'Review not found'
				review2 = 'Review not found'
				reviews = div.find_all('span', {'class' : 'UserReviewstyles__UserReviewTextStyle-sc-1y05l7z-4 UNLBr'})
				# print(len(reviews))
				if(len(reviews) > 1):
					review1 = reviews[0].text.rstrip()
				if(len(reviews) > 3):
					review2 = reviews[3].text.rstrip()
				review1l = review1
				review2l = review2
				# print("Review 1 : " + review1)
				# print("Review 2 : " + review2)

				amenities_list = []  #contains all the amenities.
				amenitiesdiv = div.find('div', {'class' : 'Amenitiesstyles__AmenitiesListBlock-sc-10opy4a-4 gTiUqx'})
				for amenitiy in amenitiesdiv.find_all('span', {'class':'Amenitiesstyles__AmenityItemText-sc-10opy4a-8 izeHgc'}) :
					amenities_list.append(amenitiy.text.rstrip())
				# print('Amenities : ', end = '')
				amenities = amenities_list
				# print(amenities_list)
				hotelname_list.append(hotel_name)
				city_list.append(city_name)
				countries_list.append(country_name)
				rating_list.append(ratingl)
				prince_list.append(pricel)
				Amenities_list.append(amenities)
				HotelDescription_list.append(descriptionl)
				Review1_list.append(review1l)
				Review2_list.append(review2l)
			# 	print()
			# print()

url = 'https://www.goibibo.com/destinations/all-states-in-india/'
data = requests.get(url)
html = data.text
soup = BeautifulSoup(html, 'html.parser')
cards = soup.find_all('div', {'class' : 'col-md-4 col-sm-4 col-xs-12 filtr-item posrel'})
# print('Found' , len(cards) , 'countries')
country_urls = []
country_names = []
for card in cards :
	for a in card.find_all('a', href=True):
		if a.text.rstrip():
			country_urls.append(a['href'])
			country_names.append(a.text.rstrip())
			# print(a['href'])
			# print()
# getting all cities in a country.
length = len(country_urls)
for i in range(length):
	url = country_urls[i]
	country_name = 'India'
	print(country_name)
	data = requests.get(url)
	html = data.text
	soup = BeautifulSoup(html, 'html.parser')
	places_to_visit = soup.find('div', {'class' : 'place-to-visit-container'})
	if(places_to_visit):
		card = places_to_visit.find('div', {'class' : 'col-md-12'})
		city_urls = {}
		for a in card.find_all('a', href=True):
			if a['href']:
				list = a['href'].split('/')
				city_urls[list[4]] = 'https://www.goibibo.com/hotels/hotels-in-' + list[4] + '-ct/'

		for city in city_urls:
			print(city)
			city_name = city
			#getting all hotels in a city along with their price
			url = city_urls[city]
			response = requests.get(url)
			data = BeautifulSoup(response.text, 'html.parser')
			cards_price_data = data.find_all('div', attrs={'class', 'HotelCardstyles__CurrentPriceTextWrapper-sc-1s80tyk-25 lnAxKT'})
			cards_url_data = data.find_all('div', attrs={'class', 'HotelCardstyles__HotelNameWrapperDiv-sc-1s80tyk-11 jkwhbV'})
			hotel_price = {}
			hotel_url = {}
			for i in range(0, len(cards_price_data)):
				hotel_price[cards_url_data[i].text.rstrip()] = cards_price_data[i].text.rstrip()
				hotel_url[cards_url_data[i].text.rstrip()] = 'https://www.goibibo.com' + cards_url_data[i].find('a', href = True)['href']

			#getting details of a hotel
			for i in range(0, len(cards_price_data)):
				url = hotel_url[cards_url_data[i].text.rstrip()]
				data = requests.get(url)
				html = data.text
				hotel_name = cards_url_data[i].text.rstrip()
				pricel = hotel_price[cards_url_data[i].text.rstrip()]
				print('Name : ' + cards_url_data[i].text.rstrip())
				print('Cost : ' + hotel_price[cards_url_data[i].text.rstrip()])

				soup = BeautifulSoup(html, 'html.parser')
				div = soup.find('div', { 'id': 'root' })
				description = div.find('section', {'class' : 'HotelDetailsMain__HotelDetailsContainer-sc-2p7gdu-0 kuBApH'})
				descriptiont = description.find('span', {'itemprop' : 'streetAddress'})
				if descriptiont:
					address = descriptiont.text.rstrip().replace(' View on Map', '')
				print('Address : ' + address) #contains address
				descriptionl = address

				rating = 'Rating not found'
				ratingdata = description.find('span', {'itemprop' : 'ratingValue'}) #contains rating
				if ratingdata:
					rating = ratingdata.text.rstrip()
				print('Rating : ' + rating)
				ratingl = rating

				review1 = 'Review not found'
				review2 = 'Review not found'
				reviews = div.find_all('span', {'class' : 'UserReviewstyles__UserReviewTextStyle-sc-1y05l7z-4 UNLBr'})
				# print(len(reviews))
				if(len(reviews) > 1):
					review1 = reviews[0].text.rstrip()
				if(len(reviews) > 3):
					review2 = reviews[3].text.rstrip()
				review1l = review1
				review2l = review2
				print("Review 1 : " + review1)
				print("Review 2 : " + review2)

				amenities_list = []  #contains all the amenities.
				amenitiesdiv = div.find('div', {'class' : 'Amenitiesstyles__AmenitiesListBlock-sc-10opy4a-4 gTiUqx'})
				if amenitiesdiv:
					for amenitiy in amenitiesdiv.find_all('span', {'class':'Amenitiesstyles__AmenityItemText-sc-10opy4a-8 izeHgc'}) :
						if amenitiy:
							amenities_list.append(amenitiy.text.rstrip())
						else:
							amenities_list.append('Amenity Not Found')
				print('Amenities : ', end = '')
				amenities = amenities_list
				print(amenities_list)
				hotelname_list.append(hotel_name)
				city_list.append(city_name)
				countries_list.append(country_name)
				rating_list.append(ratingl)
				prince_list.append(pricel)
				Amenities_list.append(amenities)
				HotelDescription_list.append(descriptionl)
				Review1_list.append(review1l)
				Review2_list.append(review2l)
				print()
			print()
dict = {'Hotel_Name': hotelname_list, 'City_Name': city_list, 'country_name': countries_list,
				'Rating' : rating_list, 'Price/Night' : prince_list, 'Amenities' : Amenities_list,
				'Description' : HotelDescription_list, 'Review1' : Review1_list, 'Review2' : Review2_list}
df = pd.DataFrame(dict)
df.to_csv('hotels.csv')