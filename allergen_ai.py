#### Allergen.ai application code ####
#### Summer '23 MIDS 210 Capstone ####

##############################################################################

#### Import Libraries ####

import streamlit as st
import requests
import pandas as pd
import s3fs
import csv
import numpy as np
import re

from textblob import Word
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import pickle

##############################################################################

##### Page Set Up #####

# set up page name
st.set_page_config(page_title = 'Allergen.ai', page_icon='🍤')
# set up title on the web application
st.title('Alergen.ai')
st.header('Allergen Identifier for Food Dishes')

##############################################################################

#### Allergen Reference List ####

# get allergens list
allergen14 = ['celery','crustaceans','egg','fish','gluten','lupin','milk','molluscs','mustard','none','peanuts','sesame','soy','sulphur dioxide and sulphites','tree nuts']

##############################################################################

#### Node 1 ####

# pull in sample node 1 output for testing purposes
# will be replaced with code to run model on user-entered dish name
node1_sample_url = 'https://raw.githubusercontent.com/tordavis/Allergen.ai/main/davinci_112_0_1_1.csv'
node_1_output_df = pd.read_csv(node1_sample_url, usecols=['title','model_output'])
# make sure all ingredients are lowercase
node_1_output_df['model_output'] = node_1_output_df['model_output'].str.lower()
# remove new line separator
node_1_output_df['model_output'] = node_1_output_df['model_output'].str.replace('\n', '')
node_1_output_df['model_output'] = node_1_output_df['model_output'].str.replace('.', '')
# split string of ingredients
lst_col = 'model_output'
node_1_output_df = node_1_output_df.assign(**{lst_col:node_1_output_df[lst_col].str.split(',')})
# explode ingredients into their own rows
node_1_output_exp_df = node_1_output_df.explode('model_output')

##############################################################################

#### Node 2 ####

# set up file system interface
fs = s3fs.S3FileSystem(anon=False)
# create definition for reading file from s3
def read_file(filename):
    with fs.open(filename) as f:
        # only pull in certain columns
        return pd.read_csv(f,usecols=['brands_spaces','product_name','allergens_from_dict'])
# read the ingredient to allergen OpenFoodFacts file
off_df = read_file("toridavis-test-8675309/open_food_facts_products_example_df.csv")
# rename the columns to be more user friendly
off_df = off_df.rename(columns={'brands_spaces':'brand','product_name':'product','allergens_from_dict':'allergen'})

##############################################################################

#### Main function ####

def main():

    #### Dish Selection ####

    # get user input for dish name
    # add rules for what can be entered
    # dish = st.text_input('Please enter a dish name', 'Beef Stroganoff')
    dish = st.selectbox('select one of the predetermined dishes',['cabbage soup','baked french toast'])

    # will be replaced
    # just keep node 1 results that match dish selected
    dish_selected = node_1_output_exp_df[node_1_output_exp_df.title == dish]

    # will be replaced
    # convert dish ingredients to a list
    dish_ingredients = dish_selected.model_output.tolist()
    # present user with the predicted ingredients
    st.write('### Here are the ingredients we have identified in your dish')
    # format list into a nice bulleted markdown
    dish_ingredients = [i.lstrip() for i in dish_ingredients]
    for i in dish_ingredients:
        st.markdown("- " + i)
    
    ##############################################################################

    #### OpenFoodFacts Reference ####

    # only keep products that match recipe ingredients
    # will be replaced with fuzzy matching code
    off_df_curated = off_df[off_df['product'].isin(dish_ingredients)]

    # have the user choose an allergen
    user_allergen = st.selectbox('Please select an allergen to show the products containing it:',allergen14)

    ##############################################################################

    #### Allergen Selection ####

    # if button is pressed to show products with allergens
    if st.button('Show products with this allergen that could be used in this dish'):
        # if a dish is entered
        if dish:
            # reduce OpenFoodFacts dataframe to just rows with allergen selected
            final_df = off_df_curated.loc[off_df_curated.allergen == user_allergen]
            # if there are no products, tell the user
            if final_df.empty:
                st.write('Based on the products available in our dataset, we did not find any potential ingredients in this dish with', user_allergen)
            # if there are products
            else:
                # may not need this anymore?
                final_df = final_df.drop_duplicates()
                # present a dataframe of brand, product, ingredient, and allergen
                st.write('### Ingredients with Allergen Present', final_df.sort_index())
        # no dish was entered so prompt the user to enter in a dish name
        else:
            st.write('Please enter a dish')

if __name__ == '__main__':
    main()