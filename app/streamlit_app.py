

import streamlit as st
import pandas as pd
import numpy as np
import time

# df = pd.DataFrame({
#     "first column": [1, 2, 3, 4],
#     "second column": [10, 20, 30, 40]
# })

# df

# magic means that df is passed to st.write()

# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns = ('col %d' % i for i in range(20))
# )
# st.table(dataframe)
# st.dataframe(dataframe.style.highlight_max(axis=0))  


chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)

st.line_chart(chart_data)


map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns = ['lat', 'lon'])

st.map(map_data)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)
# idea is each time you interact with the widge the script reruns from top to bottom

st.text_input("Your name", key = "name")

st.session_state.name

if st.checkbox("Show dataframe"):
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c']
    )
    chart_data

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

option = st.selectbox(
    "Which number do you like best?",
    df["first column"])

'You selected: ', option

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("Email", "Home phone", "Mobile phone")
)

# Add a slider to the sidebar:
add_sidebar = st.sidebar.slider(
    "Select a range of values",
    0, 100, (25, 75)
)

left_column, right_column = st.columns(2)
left_column.button('Press me!')

with right_column:
    chosen = st.radio(
        "Sorting hat",
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin")
    )
    st.write(f"You are in {chosen} house!")

# "Starting a long computation..."

# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#     # Update the progress bar with each iteration
#     latest_iteration.text(f"Iteration {i+1}")
#     bar.progress(i + 1)
#     time.sleep(0.1)

# "...and now we're done!"

# st.cache_data is how you store computations to avoid recomputing

# st.cache_rescource is the recommended way to chace gloabal resources like ML models or database connections

# So the idea is when streamlit reaches this function it checks if there is anything in cache, if there is it usese that otherwise runs function and sotres output in cache
@st.cache_data
def long_running_function(a,b):
    return "Done!"

# Session state is basically a dict unique to your session
if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times!")
st.button("Run it again")

# The key use of session state is when you want different behaviour in different sessions, so like different random data
# cached data seeems to be global across all sessions

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(np.random.randn(20,2), columns = ["x", "y"])

st.header("Choose a datapoint colour")
colour = st.color_picker("Colour", "#FF0000")
st.divider()
st.scatter_chart(st.session_state.df, x = "x", y = "y", color = colour)

# You can also cache connections
conn = st.connection("my_database")
df = conn.query("SELECT * FROM table")
st.data_editor(df)

# # To do pages as follows
# # Define the pages
# main_page = st.Page("main_page.py", title="Main Page", icon="🎈")
# page_2 = st.Page("page_2.py", title="Page 2", icon="❄️")
# page_3 = st.Page("page_3.py", title="Page 3", icon="🎉")

# # Set up navigation
# pg = st.navigation([main_page, page_2, page_3])

# # Run the selected page
# pg.run()