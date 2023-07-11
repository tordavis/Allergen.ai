import streamlit as st
import requests
import pandas as pd

# set up page name
st.set_page_config(page_title = 'Allergen.ai', page_icon='🍤')

# set up title on the web application
st.title('Alergen.ai')
st.header('Allergen Identifier for Food Dishes')

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

# get allergens list
allergen14 = ['celery','crustaceans','egg','fish','gluten','lupin','milk','molluscs','mustard','none','peanuts','sesame','soy','sulphur dioxide and sulphites','tree nuts']

# pull in ingredients to allergens dataframe
# node_2_url = 'https://drive.google.com/file/d/16-faxB25Cjb8dXmMqg1RGHgbhnUIZOF4/view?usp=sharing'
# path = 'https://drive.google.com/uc?export=download&id='+node_2_url.split('/')[-2]
# node_2_df = pd.read_csv(path)

data = [['milk','butter','brand 1','product 1'],
        ['milk','butter','brand 1','product 2'],
        ['milk','butter','brand 2','product 1'],
        ['milk','butter','brand 3','product 1'],
        ['milk','milk','brand 1','product 1'],
        ['milk','milk','brand 1','product 2'],
        ['milk','milk','brand 1','product 3'],
        ['milk','milk','brand 2','product 1'],
        ['peanuts','peanut butter','brand 1','product 1'],
        ['peanuts','peanut butter','brand 1','product 2'],
        ['peanuts','peanut butter','brand 2','product 1'],
        ['tree nuts','chopped nuts','brand 1','product 1'],
        ['tree nuts','chopped nuts','brand 2','product 1'],
        ['tree nuts','chopped nuts','brand 2','product 2'],
        ['tree nuts','chopped nuts','brand 3','product 1'],
        ]
node_2_df = pd.DataFrame(data, columns=['Allergen','Ingredient','Brand','Product'])

def main():
    # dish = st.text_input('Please enter a dish name', 'Beef Stroganoff')
    dish = st.selectbox('select one of the predetermined dishes',node_1_output_exp_df.title.unique())

    # just keep node 1 results that match dish selected
    # dish_selected = node_1_output_exp_df[node_1_output_exp_df.title == dish]
    dish_ingredients = set(node_1_output_exp_df.model_output) & set(node_2_df.Ingredient)

    # find the allergens that are present
    dish_allergens = set(node_2_df.Allergen)
    allergen_select = set(allergen14) & set(dish_allergens)

    if len(allergen_select) == 0:
        st.write('No allergens found.')
    else:
        user_allergen = st.selectbox('Please select an allergen',allergen_select)

    if st.button('Predict the allergens'):
        if dish:
            if len(node_1_output_exp_df) == 0:
                st.write('No dishes found.')
            else:
                final_df = node_2_df.loc[node_2_df.Allergen == user_allergen]
                st.write('### Ingredients with Allergen Present', final_df.sort_index())
        else:
            st.write('Please select a dish')

if __name__ == '__main__':
    main()