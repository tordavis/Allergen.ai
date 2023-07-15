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
nltk.download("wordnet")
from nltk.corpus import wordnet as wn
import pickle
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

##############################################################################

##### Page Set Up #####

# set up page name
st.set_page_config(page_title="Allergen.ai", page_icon="🍤")
# set up title on the web application
st.title("Alergen.ai")
st.header("Allergen Identifier for Food Dishes")

##############################################################################

#### Allergen Reference List ####

# get allergens list
allergen14 = [
    "celery",
    "crustaceans",
    "egg",
    "fish",
    "gluten",
    "lupin",
    "milk",
    "molluscs",
    "mustard",
    "none",
    "peanuts",
    "sesame",
    "soy",
    "sulphur dioxide and sulphites",
    "tree nuts",
]

##############################################################################

#### Node 1 - Recipe Generator ####

# pull in sample node 1 output for testing purposes
# will be replaced with code to run model on user-entered dish name
node1_sample_url = (
    "https://raw.githubusercontent.com/tordavis/Allergen.ai/main/davinci_112_0_1_1.csv"
)
node_1_output_df = pd.read_csv(node1_sample_url, usecols=["title", "model_output"])
# make sure all ingredients are lowercase
node_1_output_df["model_output"] = node_1_output_df["model_output"].str.lower()
# remove new line separator
node_1_output_df["model_output"] = node_1_output_df["model_output"].str.replace(
    "\n", ""
)
# split string of ingredients
lst_col = "model_output"
node_1_output_df = node_1_output_df.assign(
    **{lst_col: node_1_output_df[lst_col].str.split(",")}
)
# explode ingredients into their own rows
node_1_output_exp_df = node_1_output_df.explode("model_output")

##############################################################################

#### OpenFoodFacts Reference File ####

# set up file system interface
fs = s3fs.S3FileSystem(anon=False)
# create definition for reading file from s3
def read_file(filename):
    with fs.open(filename) as f:
        # only pull in certain columns
        return pd.read_csv(
            f, usecols=["product_name", "brands_tags", "allergens_from_dict"]
        )
# read the ingredient to allergen OpenFoodFacts file
off_df = read_file("toridavis-test-8675309/off_products_allergens_only_final_df.csv")
# rename the columns to be more user friendly
off_df = off_df.rename(
    columns={
        "product_name": "product",
        "brands_tags": "brand",
        "allergens_from_dict": "allergen",
    }
)

##############################################################################

#### Fuzzy Matching From Dish Ingredient to Product ####

def fuzzy_string_match(ingredient, score, products):
  """
  This function gets the fuzzy string match for a given ingredient that is greater than the score provided, or none
  Inputs:
  ingredient: string containing ingredients, with words separated by spaces
  score: Int containing minimum fuzzy match score to return a result
  products: An iterable (list, series) of unique products in OpenFoodFacts

  Returns: Best matching product (or none)
  """
  fmatch, fscore, ind = process.extractOne(
      ingredient, products, scorer=fuzz.token_sort_ratio
  )
  if int(fscore) >= score:
      return fmatch
  else:
      return None

#### Synonym Matching From Dish Ingredient to Product ####

def check_synonyms(ingredient, products):
  """
  Inputs:
  ingredient: ingredients, separated by spaces
  products: An iterable (list, series) of unique products in OpenFoodFacts

  Returns: Best matching product (or none)
  """
  # replace spaces with underscores
  ingredient_ = re.sub(" ", "_", ingredient)
  # Get synonym for ingredient_ (whole ingredient)
  syn = wn.synsets(ingredient_, pos=wn.NOUN)

  if len(syn) == 0:
      return None
  for s in syn:
      # Check if synonyms are in allergen list
      if s.lemmas()[0].name() in products:
          return s.lemmas()[0].name()

  # Check if hypernyms are in allergen list
  if len(s.hypernyms()) >= 1:
      if s.hypernyms()[0].lemmas()[0].name() in products:
          return s.lemmas()[0].name()
  return None

#### Check All Matching Methods for Dish Ingredients to Products ####

def check_products_pipeline(ingredient, products_spaces, products_underscores):
  """
  Inputs:
    ingredient: ingredients, separated by spaces

  Returns: string containing allergen class or None
  """

  # Initialize variables
  products = {
      "direct": None,
      "fuzz": None,
      "synonyms": None,
      "spellcheck": None,
      "spellcheck_fuzz": None,
      "spellcheck_synonyms": None,
  }
  # Score for fuzzy matching
  s = 75
  # Split ingredient words to list
  ingredient_split = ingredient.split()
  # Check if ingredient (or a single word in the ingredient) is in allergen list
  if ingredient in products_spaces:
    products["direct"] = ingredient
  # Check if ingredient in products list using fuzzy match
  products["fuzz"] = fuzzy_string_match(ingredient, s, products_spaces)
  # Get synonyms
  products["synonyms"] = check_synonyms(ingredient, products_underscores)

  # Get ingredient spelling suggestions
  # for one word ingredients
  if len(ingredient_split) == 1:
      spellcheck_results = Word(ingredient).spellcheck()
      for res in spellcheck_results:
          # Check spellcheck results against products
          if res[0] in products_spaces:
              products["spellcheck"] = res[0]
              break
          # Check spellcheck results against product list using fuzzy match
          fuzz_match = fuzzy_string_match(res[0], s, products_spaces)
          if fuzz_match in products_spaces:
              products["spellcheck_fuzz"] = fuzz_match
              break
  else:
      # for multi-word ingredients
      for i in range(len(ingredient_split)):
          spellcheck_results = Word(ingredient_split[i]).spellcheck()
          for res in spellcheck_results:
              # Check spellcheck results against products
              ingredient_part1 = ingredient_split[:i]
              ingredient_part3 = ingredient_split[i + 1 :]
              ingredient_part1.append(res[0])
              ingredient_part1.extend(ingredient_part3)
              spellcheck_ingredient = " ".join(ingredient_part1)
              if spellcheck_ingredient in products_spaces:
                  products["spellcheck"] = spellcheck_ingredient
                  break
              # Check spellcheck results against product list using fuzzy match
              spellcheck_fuzz_ingredient = fuzzy_string_match(
                  spellcheck_ingredient, s, products_spaces
              )
              # if there are spellchecked ingredients that match, store them
              if spellcheck_fuzz_ingredient is not None:
                  products["spellcheck_fuzz"] = spellcheck_fuzz_ingredient
                  break
              # Check spellcheck results synonymns against allergen list
              spellcheck_synonyms_ingredient = check_synonyms(
                  spellcheck_ingredient, products_underscores
              )
              # if there are synonyms for spellchecked ingredients, store them
              if spellcheck_synonyms_ingredient is not None:
                  products["spellcheck_synonyms"] = spellcheck_synonyms_ingredient
                  break
  # go through the priority list to find the matches
  final_product = ""
  method = ""
  if products["direct"] is not None:
      final_product = products["direct"]
      method = "direct"
  else:
      if products["fuzz"] is not None:
          final_product = products["fuzz"]
          method = "fuzz"
      else:
          if products["synonyms"] is not None:
              final_product = products["synonyms"]
              method = "synonyms"
          else:
              if products["spellcheck"] is not None:
                  final_product = products["spellcheck"]
                  method = "spellcheck"
              else:
                  if products["spellcheck_fuzz"] is not None:
                      final_product = products["spellcheck_fuzz"]
                      method = "spellcheck_fuzz"
                  else:
                      if products["spellcheck_synonyms"] is not None:
                          final_product = products["spellcheck_synonyms"]
                          method = "spellcheck_synonyms"
  return final_product, method

##############################################################################

#### Main function ####

def main():
  #### Dish Selection ####

  # get user input for dish name
  # add rules for what can be entered
  # dish = st.text_input('Please enter a dish name', 'Beef Stroganoff')
  dish = st.selectbox(
      "select one of the predetermined dishes", ["cabbage soup", "baked french toast"]
  )

  # will be replaced
  # just keep node 1 results that match dish selected
  dish_selected = node_1_output_exp_df[node_1_output_exp_df.title == dish]

  # will be replaced
  # convert dish ingredients to a list
  dish_ingredients = dish_selected.model_output.tolist()

  ##############################################################################

  #### Node 2 - Recipe Ingredient Matching ####

  # Get unique products from OpenFoodFacts dataframe
  unique_products = off_df['product'].unique()
  unique_products_series = pd.Series(unique_products)
  unique_products_underscore_series = unique_products_series.apply(lambda x: re.sub(' ', '_', x))
  # create a list of the matching products
  products = []
  for i in dish_ingredients:
      print(i)
      products.append(check_products_pipeline(i, unique_products_series, unique_products_underscore_series))
  # only keep the relevant products from the OpenFoodFacts dataframe
  off_df_curated = pd.DataFrame(columns = off_df.columns)
  for p in products:
      print(p[0])
      off_df_curated = pd.concat([off_df_curated, off_df[off_df['product'] == p[0]]])

  ##############################################################################

  #### Present Dish Ingredients to Users ####

  # present user with the predicted ingredients
  st.write("### Here are the ingredients we have identified in your dish")
  # format list into a nice bulleted markdown
  dish_ingredients = [i.lstrip() for i in dish_ingredients]
  for i in dish_ingredients:
      st.markdown("- " + i)

  ##############################################################################

  #### Allergen Selection ####

  # have the user choose an allergen
  user_allergen = st.selectbox("Please select an allergen to show the products containing it:", allergen14)

  # if button is pressed to show products with allergens
  if st.button("Show products with this allergen that could be used in this dish"):
      # if a dish is entered
      if dish:
          # reduce OpenFoodFacts dataframe to just rows with allergen selected
          final_df = off_df_curated.loc[off_df_curated.allergen == user_allergen]
          # if there are no products, tell the user
          if final_df.empty:
              st.write(
                  "Based on the products available in our dataset, we did not find any potential ingredients in this dish with",
                  user_allergen,
              )
          # if there are products
          else:
              # may not need this anymore?
              final_df = final_df.drop_duplicates()
              # present a dataframe of brand, product, ingredient, and allergen
              st.write("### Ingredients with Allergen Present", final_df.sort_index())
      # no dish was entered so prompt the user to enter in a dish name
      else:
          st.write("Please enter a dish")

if __name__ == "__main__":
    main()
