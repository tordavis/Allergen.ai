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
node2_sample_url = 'https://raw.githubusercontent.com/tordavis/Allergen.ai/main/open_food_facts_products_example_df.csv'
node_2_output_df = pd.read_csv(node2_sample_url, usecols=['brands_spaces','product_name','allergens_from_dict'])
node_2_output_df = node_2_output_df.rename(columns={'brands_spaces':'brand','product_name':'product','allergens_from_dict':'allergen'})

def main():
    # dish = st.text_input('Please enter a dish name', 'Beef Stroganoff')
    # dish = st.selectbox('select one of the predetermined dishes',node_1_output_exp_df.title.unique())
    dish = st.selectbox('select one of the predetermined dishes',['cabbage soup','baked french toast'])

    # just keep node 1 results that match dish selected
    dish_selected = node_1_output_exp_df[node_1_output_exp_df.title == dish]
    dish_ingredients = dish_selected.model_output.tolist()
    dish_ingredients = [i.lstrip() for i in dish_ingredients]

    node_2_curated = node_2_output_df[node_2_output_df['product'].isin(dish_ingredients)]

    user_allergen = st.selectbox('Please select an allergen',allergen14)

    if st.button('Show ingredients with this allergen'):
        if dish:
            if len(node_1_output_exp_df) == 0:
                st.write('No dishes found.')
            else:
                final_df = node_2_curated.loc[node_2_curated.allergen == user_allergen]
                if final_df.empty:
                    st.write('Based on the products available in our dataset, we did not find any potential ingredients in this dish with', user_allergen)
                else:
                    final_df = final_df.drop_duplicates()
                    st.write('### Ingredients with Allergen Present', final_df.sort_index())
                    # user_ingredient = st.selectbox('Please select an ingredient',allergen_df.ingredient.unique())
                    # ingredient_df = allergen_df.loc[allergen_df.ingredient == user_ingredient]
                    # if st.button('Show ingredients with this allergen'):
                    #     st.write('### Ingredients with Allergen Present', ingredient_df.sort_index())
                    # else:
                    #     st.write('Please select an ingredient')
        else:
            st.write('Please select a dish and allergen')

if __name__ == '__main__':
    main()