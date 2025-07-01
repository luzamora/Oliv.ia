attributes_mapping = {
    "AcceptsInsurance" : lambda v: (
        "Accepts insurance payments" if v 
        else "Does not accept insurance payments"
    ),
    "AgesAllowed" : lambda v: (
        "Accepts minimum age of 18" if str(v) == "18plus" 
        else "Accepts minimum age of 21" if str(v) == "21plus" 
        else "Accepts all ages"
    ),
    "Ambience.casual" : lambda v: (
        "Casual atmosphere" if str(v) == "True"
        else None
    ),
    "Ambience.classy" : lambda v: (
        "Classy atmosphere" if str(v) == "True"
        else None
    ),
    "Ambience.divey" : lambda v: (
        "Divey atmosphere" if str(v) == "True"
        else None
    ),
    "Ambience.hipster" : lambda v: (
        "Hipster atmosphere" if str(v) == "True"
        else None
    ),
    "Ambience.intimate" : lambda v: (
        "Intimate atmosphere" if str(v) == "True"
        else None
    ),
    "Ambience.romantic" : lambda v: (
        "Romantic atmosphere" if str(v) == "True"
        else None
    ),
    "Ambience.touristy" : lambda v: (
        "Touristy atmosphere" if str(v) == "True"
        else None
    ),
    "Ambience.trendy" : lambda v: (
        "Trendy atmosphere" if str(v) == "True"
        else None
    ),
    "Ambience.upscale" : lambda v: (
        "Upscale atmosphere" if str(v) == "True"
        else None
    ),
    "BestNights.monday" : lambda v: (
        "Best on Monday" if str(v) == "True"
        else None
    ),
    "BestNights.tuesday" : lambda v: (
        "Best on Tuesday" if str(v) == "True"
        else None
    ),
    "BestNights.wednesday" : lambda v: (
        "Best on Wednesday" if str(v) == "True"
        else None
    ),
    "BestNights.thursday" : lambda v: (
        "Best on Thursday" if str(v) == "True"
        else None
    ),
    "BestNights.friday" : lambda v: (
        "Best on Friday" if str(v) == "True"
        else None
    ),
    "BestNights.saturday" : lambda v: (
        "Best on Saturday" if str(v) == "True"
        else None
    ),
    "BestNights.sunday" : lambda v: (
        "Best on Sunday" if str(v) == "True"
        else None
    ),
    "BikeParking" : lambda v: (
        "Bike parking available" if str(v) == "True"
        else "No bike parking available" if str(v) == "False"
        else None
    ),
    "BusinessAcceptsBitcoin" : lambda v: (
        "Accepts Bitcoin" if str(v) == "True"
        else "Does not accept Bitcoin" if str(v) == "False"
        else None
    ),
    "BusinessAcceptsCreditCards" : lambda v: (
        "Accepts credit cards" if str(v) == "True"
        else "Does not accept credit cards" if str(v) == "False"
        else None
    ),
    "BusinessParking.garage" : lambda v: (
        "Garage parking available" if str(v) == "True"
        else None
    ),
    "BusinessParking.street" : lambda v: (
        "Street parking available" if str(v) == "True"
        else None
    ),
    "BusinessParking.validated" : lambda v: (
        "Validated parking available" if str(v) == "True"
        else None
    ),
    "BusinessParking.lot" : lambda v: (
        "Lot parking available" if str(v) == "True"
        else None
    ),
    "BusinessParking.valet" : lambda v: (
        "Valet parking available" if str(v) == "True"
        else None
    ),
    "ByAppointmentOnly" : lambda v: (
        "By appointment only" if str(v) == "True"
        else None
    ),
    "BYOB" : lambda v: (
        "Bring your own bottle available" if str(v) == "True"
        else None
    ),"BYOBCorkage" : lambda v: (
        "Bring your own bottle free" if str(v) == "Yes_free"
        else "Bring your own bottle with corkage fee" if str(v) == "Yes_corkage"
        else "Bring your own bottle unavailable" if str(v) == "No"
        else None
    ),
    "Corkage" : lambda v: (
        "Corkage fee available" if str(v) == "True"
        else "No corkage fee available" if str(v) == "False"
        else None
    ),
    "Caters" : lambda v: (
        "Caters available" if str(v) == "True"
        else "Does not cater" if str(v) == "False"
        else None
    ),
    "CoatCheck" : lambda v: (
        "Coat check available" if str(v) == "True"
        else "No coat check available" if str(v) == "False"
        else None
    ),
    "DietaryRestrictions.gluten_free" : lambda v: (
        "Gluten-free options available" if v
        else "No gluten-free options available"
    ),
    "DietaryRestrictions.kosher" : lambda v: (
        "Kosher options available" if v
        else "No kosher options available"
    ),
    "DietaryRestrictions.vegan" : lambda v: (
        "Vegan options available" if v
        else "No vegan options available"
    ),
    "DietaryRestrictions.vegetarian" : lambda v: (
        "Vegetarian options available" if v
        else "No vegetarian options available"
    ),
    "DietaryRestrictions.dairy-free" : lambda v: (
        "Dairy-free options available" if v
        else "No dairy-free options available"
    ),
    "DietaryRestrictions.halal" : lambda v: (
        "Halal options available" if v
        else "No halal options available"
    ),
    "DietaryRestrictions.soy-free" : lambda v: (
        "Soy-free options available" if v
        else "No soy-free options available"
    ),
    "DogsAllowed" : lambda v: (
        "Dogs allowed" if str(v) == "True"
        else "Dogs not allowed" if str(v) == "False"
        else None
    ),
    "DriveThru" : lambda v: (
        "Drive-thru available" if str(v) == "True"
        else "No drive-thru available" if str(v) == "False"
        else None
    ),
    "GoodForDancing" : lambda v: (
        "Good for dancing" if v
        else "Not good for dancing"
    ),
    "GoodForKids" : lambda v: (
        "Good for kids" if str(v) == "True"
        else "Not good for kids" if str(v) == "False"
        else None
    ),
    "GoodForMeal.breakfast" : lambda v: (
        "Good for breakfast" if str(v) == "True"
        else None
    ),
    "GoodForMeal.brunch" : lambda v: (
        "Good for brunch" if str(v) == "True"
        else None
    ),
    "GoodForMeal.lunch" : lambda v: (
        "Good for lunch" if str(v) == "True"
        else None
    ),
    "GoodForMeal.dinner" : lambda v: (
        "Good for dinner" if str(v) == "True"
        else None
    ),
    "GoodForMeal.latenight" : lambda v: (
        "Good for late night" if str(v) == "True"
        else None
    ),
    "GoodForMeal.dessert" : lambda v: (
        "Good for dessert" if str(v) == "True"
        else None
    ),
    "HappyHour" : lambda v: (
        "Happy hour available" if str(v) == "True"
        else "No happy hour available" if str(v) == "False"
        else None
    ),
    "HasTV" : lambda v: (
        "Has TV" if str(v) == "True"
        else "No TV" if str(v) == "False"
        else None
    ),
    "Music.background_music" : lambda v: (
        "Background music available" if v
        else None
    ),
    "Music.dj" : lambda v: (
        "DJ available" if str(v) == "True"
        else None
    ),
    "Music.jukebox" : lambda v: (
        "Jukebox available" if str(v) == "True"
        else None
    ),
    "Music.live" : lambda v: (
        "Live music available" if str(v) == "True"
        else None
    ),
    "Music.karaoke" : lambda v: (
        "Karaoke available" if str(v) == "True"
        else None
    ),
    "Music.video" : lambda v: (
        "Video available" if v
        else None
    ),
    "NoiseLevel" : lambda v: (
        f"Noise level is {v}" if v
        else None
    ),
    "Open24Hours" : lambda v: (
        "Open 24 hours" if v
        else "Not open 24 hours"
    ),
    "OutdoorSeating" : lambda v: (
        "Outdoor seating available" if str(v) == "True"
        else "No outdoor seating available" if str(v) == "False"
        else None
    ),
    "RestaurantsAttire" : lambda v: (
        "Casual attire" if str(v) == "Casual"
        else "Formal attire" if str(v) == "Formal"
        else "Dressy attire" if str(v) == "Dressy"
        else None
    ),
    "RestaurantsCounterService" : lambda v: (
        "Counter service available" if v
        else "No counter service available"
    ),
    "RestaurantsDelivery" : lambda v: (
        "Delivery avaiable" if str(v) == "True"
        else "Delivery unavaible" if str(v) == "False"
        else None
    ),
    "RestaurantsGoodForGroups" : lambda v: (
        "Good for groups" if str(v) == "True"
        else "Not good for groups" if str(v) == "False"
        else None
    ),
    "RestaurantsPriceRange2" : lambda v: (
        "Cheap price" if str(v) == "1"
        else "Medium price" if str(v) == "2"
        else "Expensive price" if str(v) == "3"
        else "Luxury price" if str(v) == "4"
        else None
    ),
    "RestaurantsReservations" : lambda v: (
        "Reservations avaiable" if str(v) == "True"
        else "Reservations not avaiable" if str(v) == "False"
        else None
    ),
    "RestaurantsTableService" : lambda v: (
        "Table service avaiable" if str(v) == "True"
        else "Table service not avaiable" if str(v) == "False"
        else None
    ),
    "RestaurantsTakeOut" : lambda v: (
        "Take out avaiable" if str(v) == "True"
        else "Take out not avaiable" if str(v) == "False"
        else None
    ),
    "Smoking" : lambda v: (
        "Smoking avaiable indoors" if str(v) == "Yes"
        else "Smoking only avaiable outdoors" if str(v) == "Outdoor"
        else "Smoking prohibited" if str(v) == "No"
        else None
    ),
    "WheelchairAccessible" : lambda v: (
        "Wheelchair accessible" if str(v) == "True"
        else "Not accesible for wheelchairs" if str(v) == "False"
        else None
    ),
    "WiFi" : lambda v: (
        "Free WiFi" if str(v) == "Free"
        else "Paid WiFi" if str(v) == "Paid"
        else "No WiFi" if str(v) == "No"
        else None
    )
}

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import torch
import hnswlib

## data_merge
def data_merge(business_data, review_data, merge_id = "business_id", text_column = "text"):
    reviews_df_gouped = review_data.groupby(merge_id)[text_column].apply(list).reset_index()
    return pd.merge(business_data, reviews_df_gouped, on=merge_id, how="left")

## Funcion
def inferred_function(hypothesis_values = dict, df_embeddings = pd.DataFrame):
    # Cargar modelo
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Detectar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Codificar hipótesis
    hypothesis_texts = list(hypothesis_values.values())
    hypothesis_embeddings = model.encode(hypothesis_texts, convert_to_tensor=True).to(device)

    # Convertir embeddings a tensor
    restaurant_embeddings = torch.tensor(df_embeddings.values, dtype=torch.float).to(device)

    # Calcular similitud
    cosine_scores = util.cos_sim(restaurant_embeddings, hypothesis_embeddings)

    # Umbral para decidir si el atributo aplica
    threshold = 0.4 # 40%
    predictions = (cosine_scores >= threshold).int()

    # Crear DataFrame
    attribute_names = list(hypothesis_values.keys())
    df_predictions = pd.DataFrame(predictions.cpu().numpy(), columns=attribute_names)

    return df_predictions

def final_condition(data1 = pd.DataFrame, data2 = pd.DataFrame):
    condicion_1_0_or_1_1 = (data1 == 1)
    condicion_0_1 = (data1 == 0) & (data2 == 1)

    return (condicion_1_0_or_1_1 | condicion_0_1).astype(int)

hypothesis_attributes = {
    'RestaurantsDelivery': "This restaurant offers delivery.",
    'OutdoorSeating': "This restaurant has outdoor seating.",
    'BusinessAcceptsCreditCards': "This business accepts credit cards.",
    'BusinessParking.garage': "This business has a parking garage.",
    'BusinessParking.street': "This business has street parking.",
    'BusinessParking.validated': "This business offers validated parking.",
    'BusinessParking.lot': "This business has a parking lot.",
    'BusinessParking.valet': "This business offers valet parking.",
    'BikeParking': "There is parking space for bikes.",
    'RestaurantsPriceRange2_1': "This restaurant has a price range of 1 (e.g., $).",
    'RestaurantsPriceRange2_2': "This restaurant has a price range of 2 (e.g., $$).",
    'RestaurantsPriceRange2_3': "This restaurant has a price range of 3 (e.g., $$$).",
    'RestaurantsPriceRange2_4': "This restaurant has a price range of 4 (e.g., $$$$).",
    'RestaurantsTakeOut': "This restaurant offers takeout.",
    'ByAppointmentOnly': "You can only visit this business by appointment.",
    'WiFi_free': "This place offers free Wi-Fi.",
    'WiFi_no': "This place does not offer Wi-Fi.",
    'WiFi_paid': "This place offers paid Wi-Fi.",
    'Alcohol_beer_and_wine': "This place serves beer and wine only.",
    'Alcohol_full_bar': "This place has a full bar.",
    'Caters': "This business offers catering services.",
    'RestaurantsAttire_casual': "This restaurant has a casual dress code.",
    'RestaurantsAttire_dressy': "This restaurant has a dressy dress code.",
    'RestaurantsAttire_formal': "This restaurant has a formal dress code.",
    'RestaurantsReservations': "Reservations are accepted at this restaurant.",
    'GoodForKids': "This place is good for kids.",
    'CoatCheck': "This place offers coat check.",
    'DogsAllowed': "Dogs are allowed at this place.",
    'RestaurantsTableService': "This restaurant offers table service.",
    'RestaurantsGoodForGroups': "This restaurant is good for groups.",
    'WheelchairAccessible': "This business is wheelchair accessible.",
    'HasTV': "This place has a TV.",
    'HappyHour': "This place has a happy hour.",
    'DriveThru': "This business has a drive-thru.",
    'NoiseLevel_average': "The noise level at this place is average.",
    'NoiseLevel_loud': "The noise level at this place is loud.",
    'NoiseLevel_quiet': "The noise level at this place is quiet.",
    'NoiseLevel_very_loud': "The noise level at this place is very loud.",
    'Ambience.romantic': "This place has a romantic ambience.",
    'Ambience.intimate': "This place has an intimate ambience.",
    'Ambience.touristy': "This place has a touristy ambience.",
    'Ambience.hipster': "This place has a hipster ambience.",
    'Ambience.divey': "This place has a divey ambience.",
    'Ambience.classy': "This place has a classy ambience.",
    'Ambience.trendy': "This place has a trendy ambience.",
    'Ambience.upscale': "This place has an upscale ambience.",
    'Ambience.casual': "This place has a casual ambience.",
    'GoodForMeal.dessert': "This place is good for dessert.",
    'GoodForMeal.latenight': "This place is good for late-night meals.",
    'GoodForMeal.lunch': "This place is good for lunch.",
    'GoodForMeal.dinner': "This place is good for dinner.",
    'GoodForMeal.brunch': "This place is good for brunch.",
    'GoodForMeal.breakfast': "This place is good for breakfast.",
    'BusinessAcceptsBitcoin': "This business accepts Bitcoin.",
    'Smoking_no': "Smoking is not allowed at this place.",
    'Smoking_outdoor': "Smoking is only allowed outdoors at this place.",
    'Smoking_yes': "Smoking is allowed at this place.",
    'Music.dj': "This place has a DJ.",
    'Music.background_music': "This place plays background music.",
    'Music.jukebox': "This place has a jukebox.",
    'Music.live': "This place has live music.",
    'Music.video': "This place has music videos.",
    'Music.karaoke': "This place has karaoke.",
    'GoodForDancing': "This place is good for dancing.",
    'BestNights.monday': "This place is popular on Mondays.",
    'BestNights.tuesday': "This place is popular on Tuesdays.",
    'BestNights.friday': "This place is popular on Fridays.",
    'BestNights.wednesday': "This place is popular on Wednesdays.",
    'BestNights.thursday': "This place is popular on Thursdays.",
    'BestNights.sunday': "This place is popular on Sundays.",
    'BestNights.saturday': "This place is popular on Saturdays.",
    'BYOB': "This place allows BYOB (Bring Your Own Bottle).",
    'Corkage': "This place charges a corkage fee.",
    'BYOBCorkage_no': "This place does not allow BYOB or charge corkage.",
    'BYOBCorkage_yes_corkage': "This place allows BYOB with a corkage fee.",
    'BYOBCorkage_yes_free': "This place allows BYOB for free.",
    'RestaurantsCounterService': "This restaurant offers counter service.",
    'Open24Hours': "This business is open 24 hours.",
    'AgesAllowed_18plus': "Only ages 18 and older are allowed at this place.",
    'AgesAllowed_21plus': "Only ages 21 and older are allowed at this place.",
    'AgesAllowed_allages': "All ages are allowed at this place.",
    'DietaryRestrictions.dairy-free': "This place offers dairy-free options.",
    'DietaryRestrictions.gluten-free': "This place offers gluten-free options.",
    'DietaryRestrictions.vegan': "This place offers vegan options.",
    'DietaryRestrictions.halal': "This place offers halal options.",
    'DietaryRestrictions.soy-free': "This place offers soy-free options.",
    'DietaryRestrictions.vegetarian': "This place offers vegetarian options.",
    'AcceptsInsurance': "This business accepts insurance."
}

hypothesis_categories = {
    'Acai_Bowls': "This business is an Açaí Bowls establishment.",
    'Active_Life': "This business belongs to the Active Life category.",
    'Adult_Entertainment': "This business is an Adult Entertainment venue.",
    'Afghan': "This business serves Afghan cuisine.",
    'African': "This business serves African cuisine.",
    'Airport_Lounges': "This business is an Airport Lounge.",
    'American_(New)': "This business serves New American cuisine.",
    'American_(Traditional)': "This business serves Traditional American cuisine.",
    'Antiques': "This business sells Antiques.",
    'Appliances': "This business sells Appliances.",
    'Arabic': "This business serves Arabic cuisine.",
    'Arcades': "This business is an Arcade.",
    'Argentine': "This business serves Argentine cuisine.",
    'Armenian': "This business serves Armenian cuisine.",
    'Art_Galleries': "This business is an Art Gallery.",
    'Arts_&_Crafts': "This business sells Arts & Crafts.",
    'Arts_&_Entertainment': "This business belongs to the Arts & Entertainment category.",
    'Asian_Fusion': "This business serves Asian Fusion cuisine.",
    'Australian': "This business serves Australian cuisine.",
    'Austrian': "This business serves Austrian cuisine.",
    'Auto_Repair': "This business provides Auto Repair services.",
    'Automotive': "This business belongs to the Automotive category.",
    'Bagels': "This business sells Bagels.",
    'Bakeries': "This business is a Bakery.",
    'Bangladeshi': "This business serves Bangladeshi cuisine.",
    'Bar_Crawl': "This business organizes or is part of a Bar Crawl.",
    'Barbeque': "This business serves Barbeque.",
    'Bars': "This business is a Bar.",
    'Bartenders': "This business provides Bartender services.",
    'Basque': "This business serves Basque cuisine.",
    'Beaches': "This business is located at or related to Beaches.",
    'Bed_&_Breakfast': "This business is a Bed & Breakfast.",
    'Beer': "This business sells Beer.",
    'Beer_Bar': "This business is a Beer Bar.",
    'Beer_Gardens': "This business is a Beer Garden.",
    'Belgian': "This business serves Belgian cuisine.",
    'Beverage_Store': "This business is a Beverage Store.",
    'Bike_Rentals': "This business offers Bike Rentals.",
    'Bistros': "This business is a Bistro.",
    'Boating': "This business offers Boating activities.",
    'Books': "This business sells Books.",
    'Bookstores': "This business is a Bookstore.",
    'Bowling': "This business is a Bowling alley.",
    'Brasseries': "This business is a Brasserie.",
    'Brazilian': "This business serves Brazilian cuisine.",
    'Breakfast_&_Brunch': "This business serves Breakfast & Brunch.",
    'Breweries': "This business is a Brewery.",
    'Brewpubs': "This business is a Brewpub.",
    'British': "This business serves British cuisine.",
    'Bubble_Tea': "This business sells Bubble Tea.",
    'Buffets': "This business is a Buffet.",
    'Burgers': "This business serves Burgers.",
    'Burmese': "This business serves Burmese cuisine.",
    'Butcher': "This business is a Butcher shop.",
    'Cabaret': "This business is a Cabaret.",
    'Cafes': "This business is a Cafe.",
    'Cafeteria': "This business is a Cafeteria.",
    'Cajun/Creole': "This business serves Cajun/Creole cuisine.",
    'Cambodian': "This business serves Cambodian cuisine.",
    'Canadian_(New)': "This business serves New Canadian cuisine.",
    'Candy_Stores': "This business is a Candy Store.",
    'Cantonese': "This business serves Cantonese cuisine.",
    'Car_Dealers': "This business is a Car Dealer.",
    'Caribbean': "This business serves Caribbean cuisine.",
    'Casinos': "This business is a Casino.",
    'Champagne_Bars': "This business is a Champagne Bar.",
    'Cheese_Shops': "This business is a Cheese Shop.",
    'Cheesesteaks': "This business serves Cheesesteaks.",
    'Chicken_Shop': "This business is a Chicken Shop.",
    'Chicken_Wings': "This business serves Chicken Wings.",
    'Chinese': "This business serves Chinese cuisine.",
    'Chocolatiers_&_Shops': "This business is a Chocolatier & Shop.",
    'Churches': "This business is a Church.",
    'Cideries': "This business is a Cidery.",
    'Cigar_Bars': "This business is a Cigar Bar.",
    'Cinema': "This business is a Cinema.",
    'Cocktail_Bars': "This business is a Cocktail Bar.",
    'Coffee_&_Tea': "This business serves Coffee & Tea.",
    'Coffee_Roasteries': "This business is a Coffee Roastery.",
    'Colombian': "This business serves Colombian cuisine.",
    'Comedy_Clubs': "This business is a Comedy Club.",
    'Comfort_Food': "This business serves Comfort Food.",
    'Community_Service/Non-Profit': "This business is a Community Service/Non-Profit organization.",
    'Convenience_Stores': "This business is a Convenience Store.",
    'Cooking_Classes': "This business offers Cooking Classes.",
    'Country_Clubs': "This business is a Country Club.",
    'Creperies': "This business serves Crepes.",
    'Cuban': "This business serves Cuban cuisine.",
    'Cupcakes': "This business sells Cupcakes.",
    'Custom_Cakes': "This business creates Custom Cakes.",
    'Czech': "This business serves Czech cuisine.",
    'Dance_Clubs': "This business is a Dance Club.",
    'Delicatessen': "This business is a Delicatessen.",
    'Delis': "This business is a Deli.",
    'Desserts': "This business serves Desserts.",
    'Dim_Sum': "This business serves Dim Sum.",
    'Diners': "This business is a Diner.",
    'Dinner_Theater': "This business offers Dinner Theater.",
    'Distilleries': "This business is a Distillery.",
    'Dive_Bars': "This business is a Dive Bar.",
    'Do-It-Yourself_Food': "This business offers Do-It-Yourself Food experiences.",
    'Dominican': "This business serves Dominican cuisine.",
    'Donairs': "This business serves Donairs.",
    'Donuts': "This business sells Donuts.",
    'Eatertainment': "This business belongs to the Eatertainment category.",
    'Egyptian': "This business serves Egyptian cuisine.",
    'Empanadas': "This business serves Empanadas.",
    'Ethiopian': "This business serves Ethiopian cuisine.",
    'Ethnic_Food': "This business serves Ethnic Food.",
    'Ethnic_Grocery': "This business is an Ethnic Grocery store.",
    'Falafel': "This business serves Falafel.",
    'Farmers_Market': "This business is a Farmers Market.",
    'Farms': "This business is a Farm.",
    'Fashion': "This business belongs to the Fashion category.",
    'Fast_Food': "This business serves Fast Food.",
    'Filipino': "This business serves Filipino cuisine.",
    'Fish_&_Chips': "This business serves Fish & Chips.",
    'Fishing': "This business offers Fishing activities.",
    'Fitness_&_Instruction': "This business provides Fitness & Instruction.",
    'Fondue': "This business serves Fondue.",
    'Food_Court': "This business is a Food Court.",
    'Food_Delivery_Services': "This business provides Food Delivery Services.",
    'Food_Stands': "This business is a Food Stand.",
    'Food_Trucks': "This business is a Food Truck.",
    'French': "This business serves French cuisine.",
    'Fruits_&_Veggies': "This business sells Fruits & Veggies.",
    'Gas_Stations': "This business is a Gas Station.",
    'Gastropubs': "This business is a Gastropub.",
    'Gay_Bars': "This business is a Gay Bar.",
    'Gelato': "This business sells Gelato.",
    'German': "This business serves German cuisine.",
    'Gluten-Free': "This business offers Gluten-Free options.",
    'Golf': "This business offers Golf.",
    'Greek': "This business serves Greek cuisine.",
    'Grocery': "This business is a Grocery store.",
    'Gyms': "This business is a Gym.",
    'Haitian': "This business serves Haitian cuisine.",
    'Halal': "This business serves Halal food.",
    'Hawaiian': "This business serves Hawaiian cuisine.",
    'Health_Markets': "This business is a Health Market.",
    'Herbs_&_Spices': "This business sells Herbs & Spices.",
    'Himalayan/Nepalese': "This business serves Himalayan/Nepalese cuisine.",
    'Home_&_Garden': "This business belongs to the Home & Garden category.",
    'Honduran': "This business serves Honduran cuisine.",
    'Hong_Kong_Style_Cafe': "This business is a Hong Kong Style Cafe.",
    'Hookah_Bars': "This business is a Hookah Bar.",
    'Hot_Dogs': "This business serves Hot Dogs.",
    'Hot_Pot': "This business serves Hot Pot.",
    'Hotels': "This business is a Hotel.",
    'Hotels_&_Travel': "This business belongs to the Hotels & Travel category.",
    'Hungarian': "This business serves Hungarian cuisine.",
    'Iberian': "This business serves Iberian cuisine.",
    'Ice_Cream_&_Frozen_Yogurt': "This business sells Ice Cream & Frozen Yogurt.",
    'Imported_Food': "This business sells Imported Food.",
    'Indian': "This business serves Indian cuisine.",
    'Indonesian': "This business serves Indonesian cuisine.",
    'Indoor_Playcentre': "This business is an Indoor Playcentre.",
    'International': "This business serves International cuisine.",
    'International_Grocery': "This business is an International Grocery store.",
    'Internet_Cafes': "This business is an Internet Cafe.",
    'Irish': "This business serves Irish cuisine.",
    'Irish_Pub': "This business is an Irish Pub.",
    'Italian': "This business serves Italian cuisine.",
    'Izakaya': "This business is an Izakaya.",
    'Japanese': "This business serves Japanese cuisine.",
    'Japanese_Curry': "This business serves Japanese Curry.",
    'Jazz_&_Blues': "This business offers Jazz & Blues music.",
    'Juice_Bars_&_Smoothies': "This business is a Juice Bar & Smoothies shop.",
    'Karaoke': "This business offers Karaoke.",
    'Kebab': "This business serves Kebab.",
    'Kids_Activities': "This business offers Kids Activities.",
    'Kitchen_&_Bath': "This business specializes in Kitchen & Bath.",
    'Kombucha': "This business sells Kombucha.",
    'Korean': "This business serves Korean cuisine.",
    'Kosher': "This business serves Kosher food.",
    'Laotian': "This business serves Laotian cuisine.",
    'Latin_American': "This business serves Latin American cuisine.",
    'Lebanese': "This business serves Lebanese cuisine.",
    'Leisure_Centers': "This business is a Leisure Center.",
    'Live/Raw_Food': "This business serves Live/Raw Food.",
    'Local_Flavor': "This business offers Local Flavor.",
    'Local_Services': "This business provides Local Services.",
    'Lounges': "This business is a Lounge.",
    'Macarons': "This business sells Macarons.",
    'Mags': "This business sells Magazines.",
    'Malaysian': "This business serves Malaysian cuisine.",
    'Meat_Shops': "This business is a Meat Shop.",
    'Mediterranean': "This business serves Mediterranean cuisine.",
    'Mexican': "This business serves Mexican cuisine.",
    'Middle_Eastern': "This business serves Middle Eastern cuisine.",
    'Mini_Golf': "This business offers Mini Golf.",
    'Modern_European': "This business serves Modern European cuisine.",
    'Mongolian': "This business serves Mongolian cuisine.",
    'Moroccan': "This business serves Moroccan cuisine.",
    'Museums': "This business is a Museum.",
    'Music_&_Video': "This business sells Music & Video.",
    'Music_Venues': "This business is a Music Venue.",
    'Musicians': "This business provides Musician services.",
    'New_Mexican_Cuisine': "This business serves New Mexican Cuisine.",
    'Nicaraguan': "This business serves Nicaraguan cuisine.",
    'Nightlife': "This business belongs to the Nightlife category.",
    'Noodles': "This business serves Noodles.",
    'Organic_Stores': "This business is an Organic Store.",
    'Pakistani': "This business serves Pakistani cuisine.",
    'Pan_Asian': "This business serves Pan Asian cuisine.",
    'Pancakes': "This business serves Pancakes.",
    'Parks': "This business is a Park.",
    'Pasta_Shops': "This business is a Pasta Shop.",
    'Patisserie/Cake_Shop': "This business is a Patisserie/Cake Shop.",
    'Performing_Arts': "This business offers Performing Arts.",
    'Persian/Iranian': "This business serves Persian/Iranian cuisine.",
    'Peruvian': "This business serves Peruvian cuisine.",
    'Pet_Adoption': "This business facilitates Pet Adoption.",
    'Pets': "This business sells Pets or pet supplies.",
    'Piano_Bars': "This business is a Piano Bar.",
    'Pizza': "This business serves Pizza.",
    'Poke': "This business serves Poke.",
    'Polish': "This business serves Polish cuisine.",
    'Pool_&_Billiards': "This business offers Pool & Billiards.",
    'Pool_Halls': "This business is a Pool Hall.",
    'Pop-Up_Restaurants': "This business is a Pop-Up Restaurant.",
    'Pop-up_Shops': "This business is a Pop-up Shop.",
    'Popcorn_Shops': "This business is a Popcorn Shop.",
    'Portuguese': "This business serves Portuguese cuisine.",
    'Poutineries': "This business serves Poutineries.",
    'Pretzels': "This business sells Pretzels.",
    'Public_Markets': "This business is a Public Market.",
    'Pubs': "This business is a Pub.",
    'Puerto_Rican': "This business serves Puerto Rican cuisine.",
    'Ramen': "This business serves Ramen.",
    'Recreation_Centers': "This business is a Recreation Center.",
    'Religious_Organizations': "This business is a Religious Organization.",
    'Resorts': "This business is a Resort.",
    'Russian': "This business serves Russian cuisine.",
    'Salad': "This business serves Salad.",
    'Salvadoran': "This business serves Salvadoran cuisine.",
    'Sandwiches': "This business serves Sandwiches.",
    'Scandinavian': "This business serves Scandinavian cuisine.",
    'Scottish': "This business serves Scottish cuisine.",
    'Seafood': "This business serves Seafood.",
    'Seafood_Markets': "This business is a Seafood Market.",
    'Senegalese': "This business serves Senegalese cuisine.",
    'Shanghainese': "This business serves Shanghainese cuisine.",
    'Shaved_Ice': "This business sells Shaved Ice.",
    'Shaved_Snow': "This business sells Shaved Snow.",
    'Sicilian': "This business serves Sicilian cuisine.",
    'Singaporean': "This business serves Singaporean cuisine.",
    'Skating_Rinks': "This business is a Skating Rink.",
    'Smokehouse': "This business is a Smokehouse.",
    'Social_Clubs': "This business is a Social Club.",
    'Soul_Food': "This business serves Soul Food.",
    'Soup': "This business serves Soup.",
    'South_African': "This business serves South African cuisine.",
    'Southern': "This business serves Southern cuisine.",
    'Spanish': "This business serves Spanish cuisine.",
    'Speakeasies': "This business is a Speakeasy.",
    'Sporting_Goods': "This business sells Sporting Goods.",
    'Sports_Bars': "This business is a Sports Bar.",
    'Sports_Clubs': "This business is a Sports Club.",
    'Steakhouses': "This business is a Steakhouse.",
    'Street_Vendors': "This business is a Street Vendor.",
    'Sushi_Bars': "This business is a Sushi Bar.",
    'Syrian': "This business serves Syrian cuisine.",
    'Szechuan': "This business serves Szechuan cuisine.",
    'Tabletop_Games': "This business offers Tabletop Games.",
    'Tacos': "This business serves Tacos.",
    'Taiwanese': "This business serves Taiwanese cuisine.",
    'Tapas_Bars': "This business is a Tapas Bar.",
    'Tapas/Small_Plates': "This business serves Tapas/Small Plates.",
    'Tea_Rooms': "This business is a Tea Room.",
    'Tennis': "This business offers Tennis facilities.",
    'Teppanyaki': "This business serves Teppanyaki.",
    'Tex-Mex': "This business serves Tex-Mex cuisine.",
    'Thai': "This business serves Thai cuisine.",
    'Themed_Cafes': "This business is a Themed Cafe.",
    'Tiki_Bars': "This business is a Tiki Bar.",
    'Tobacco_Shops': "This business is a Tobacco Shop.",
    'Travel_Services': "This business provides Travel Services.",
    'Trinidadian': "This business serves Trinidadian cuisine.",
    'Turkish': "This business serves Turkish cuisine.",
    'Tuscan': "This business serves Tuscan cuisine.",
    'Ukrainian': "This business serves Ukrainian cuisine.",
    'Uzbek': "This business serves Uzbek cuisine.",
    'Vegan': "This business serves Vegan food.",
    'Vegetarian': "This business serves Vegetarian food.",
    'Venezuelan': "This business serves Venezuelan cuisine.",
    'Vietnamese': "This business serves Vietnamese cuisine.",
    'Waffles': "This business serves Waffles.",
    'Wedding_Planning': "This business offers Wedding Planning services.",
    'Whiskey_Bars': "This business is a Whiskey Bar.",
    'Wholesalers': "This business is a Wholesaler.",
    'Wine_&_Spirits': "This business sells Wine & Spirits.",
    'Wine_Bars': "This business is a Wine Bar.",
    'Wine_Tasting_Room': "This business is a Wine Tasting Room.",
    'Wineries': "This business is a Winery.",
    'Wraps': "This business serves Wraps."
}

def top_similarities(query_embedding, comparison_embeddings, top_n):
    cosine_similarities = cosine_similarity(query_embedding, comparison_embeddings)[0]
    return cosine_similarities.argsort()[-top_n:][::-1]

def top_similarities_hnsw(query_embedding, comparison_embeddings, top_n, space='cosine', ef=50, ef_construction=200, M=16):
    """
    Parámetros:
    - query_embedding: array shape (1, dim)
    - comparison_embeddings: array shape (N, dim)
    - top_n: número de vecinos a devolver
    - space: 'cosine', 'l2', o 'ip' (dot product)
    - ef: número de candidatos evaluados en la búsqueda (calidad vs velocidad)
    - ef_construction: calidad de construcción del índice
    - M: número de conexiones por nodo (memoria vs calidad)
    """
    comparison_embeddings = np.array(comparison_embeddings)
    dims = comparison_embeddings.shape[1]

    index = hnswlib.Index(space=space, dim=dims)
    index.init_index(max_elements=len(comparison_embeddings), ef_construction=ef_construction, M=M)
    index.add_items(comparison_embeddings)
    index.set_ef(ef)

    query_embedding = np.array(query_embedding)
    labels, _ = index.knn_query(query_embedding, k=top_n)
    return labels[0]
