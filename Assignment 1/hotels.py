# import requests
# from bs4 import BeautifulSoup
# import pandas as pd  
# # # get the data
# # url = 'https://www.goibibo.com/hotels/ao-amsterdam-zuidoost-hotel-in-amsterdam-321791056710794532/?hquery={%22ci%22:%2220200813%22,%22co%22:%2220200814%22,%22r%22:%221-2-0%22,%22ibp%22:%22v15%22}&hmd=b5377794530a455345945a593cbaa4e996e1a7d3d70f8fa5114c2f34d636d7bd69200feeea3aaf6456c5ed0621788b1c3367b69b056fbe80828a0c6e10f09c2e17f2acc4b637f6796395bee8b1c822cb530af91b7d337e9682007aacb12be705816167408b7d9593b8b1f274d4ba02a510c01b87e051757dfdc270c86aa40d6fc83a1d9dca7a5d8a2870ffe79e0da68e5955d0b5143155d2fa16911d3da0121e8665a31fe37ab28dabf5667eea8a47bc18312382201f3398b803e6a7387007089aac8ad176f440b7bcd89d929f0fb054884202abce476071bcd1ac744a4a1360dda41c3af01936cf178aac7c785dbcff799816e50cc0b19afb78b28b7b0feba3348c2899e4a4f63a7b741805f341033d5fb31bfb0a54e57b52ae4518be5845fd8e96515a7031f30e2f4bd7499946a2bcaaf56a36d65e43ca594a84e9bf542861&cc=NL'
# # # url = 'https://www.goibibo.com/hotels/-3740924996345048826/?hquery={%22ci%22:%2220200813%22,%22co%22:%2220200814%22,%22r%22:%221-2-0%22,%22ibp%22:%22v15%22}&hmd=b5377794530a455345945a593cbaa4e996e1a7d3d70f8fa5114c2f34d636d7bd69200feeea3aaf6456c5ed0621788b1c3367b69b056fbe80828a0c6e10f09c2e17f2acc4b637f6796395bee8b1c822cb530af91b7d337e9682007aacb12be705816167408b7d9593b8b1f274d4ba02a510c01b87e051757dfdc270c86aa40d6fc83a1d9dca7a5d8a2870ffe79e0da68e5955d0b5143155d2fa16911d3da0121e8665a31fe37ab28dabf5667eea8a47bc18312382201f3398b803e6a7387007089aac8ad176f440b7bcd89d929f0fb054884202abce476071bcd1ac744a4a1360dda41c3af01936cf178aac7c785dbcff799816e50cc0b19afb78b28b7b0feba3348c2899e4a4f63a7b741805f341033d5fb31bfb0a54e57b52ae4518be5845fd8e96515a7031f30e2f4bd7499946a2bcaaf56a36d65e43ca594a84e9bf542861&cc=NL'
# # # url = 'https://www.goibibo.com/hotels/parth-inn-hotel-in-jagdalpur-3287238417944512070/?hquery={%22ci%22:%2220200813%22,%22co%22:%2220200814%22,%22r%22:%221-2-0%22,%22ibp%22:%22v3%22}&hmd=dd472c27cfd67e76db54c18fdfb9599a716f72f54a30d60b1b4585dfd9e306fb068981239315a1dc7a825676a004e2def6eab84d9e1535c756467e939cd673232086f62eb1ea3b387bec12e0d0131434c61c6cbdf20e853abfe494b31fd40b5e05cb3a03b6bef9c8f6db49936885d2e681e6ead3ee8d0ccf7fc687f6d19e9490402eb8157f482ad5844651b009917f7eda57a3fcfba6046224cec82dced7fd78b3045096b92e2098333370de7ff17dbf66c3a2d00ecad5f1173e7fb74a1250841c8dd352f9df44d569b9ac3d3a328a99036b92a18dcdd2b40f4ea9f539f29f085616d4b018f0a3177628f83157bb2e51c9e86cc3138d944f863921472a57f9c7541e015237bb3069a554b2f3b8e436bcb4eae1fa75a178175592e2d728ce0b17146f48bcc8583372539aa1c4bd524b109a33d6a54a53f41ff39c&cc=IN'
# # data = requests.get(url)
# # html = data.text

# # soup = BeautifulSoup(html, 'html.parser')
# # div = soup.find('div', { 'id': 'root' })
# # description = div.find('section', {'class' : 'HotelDetailsMain__HotelDetailsContainer-sc-2p7gdu-0 kuBApH'})
# # address = description.find('span', {'itemprop' : 'streetAddress'}).text.rstrip().replace(' View on Map', '')
# # print(address) #contains address
# # rating = description.find('span', {'itemprop' : 'ratingValue'}).text.rstrip() #contains rating
# # print(rating)

# # amenities_list = []
# # amenitiesdiv = div.find('div', {'class' : 'Amenitiesstyles__AmenitiesListBlock-sc-10opy4a-4 gTiUqx'})
# # for amenitiy in amenitiesdiv.find_all('span', {'class':'Amenitiesstyles__AmenityItemText-sc-10opy4a-8 izeHgc'}) :
# # 	amenities_list.append(amenitiy.text.rstrip())

# # print(amenities_list) #contains all the amenities.

url = "https://www.goibibo.com/hotels/intl-hotels-in-amsterdam-ct/"
response = requests.get(url)
data = BeautifulSoup(response.text, 'html.parser')
cards_data = data.find_all('div', attrs={'class', 'HotelCardstyles__HotelNameWrapperDiv-sc-1s80tyk-11 jkwhbV'})
print('Total Number of Cards Found : ', len(cards_data))
print(cards_data[0].text.rstrip())

for a in cards_data[0].find_all('a', href=True):
    print("Found the URL:", a['href'])

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# import gc

# URL = "https://www.makemytrip.com/hotels/sitemap-hotels-hotel_detail_template-15-new.xml"

# r = requests.get(URL)
# soup = BeautifulSoup(r.content,"lxml")

# HotelName=[]

# for link in soup.findAll("loc"):
#     if len(HotelName)>=3000:
#       break
#     else:
#       HotelName.append(link.contents[0])

# print(len(HotelName))