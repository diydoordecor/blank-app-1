import streamlit as st

st.title("🎈 My new Streamlit app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df
# Draw a title and some text to the app:
'''
# This is the document title

This is some _markdown_.
'''

import pandas as pd
df = pd.DataFrame({'col1': [1,2,3]})
df  # 👈 Draw the dataframe

x = 10
'x', x  # 👈 Draw the string 'x' and then the value of x

import streamlit as st
import numpy as np

dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(dataframe.style.highlight_max(axis=0))

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
st.table(dataframe)

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    chart_data

df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

option = st.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected: ', option

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

    
left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")    

if "counter" not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1

st.header(f"This page has run {st.session_state.counter} times.")
st.button("Run it again")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Generate random numbers and create a pie chart
data = np.random.randint(1, 11, size=5)
labels = [f"Category {i+1}" for i in range(len(data))]

fig, ax = plt.subplots()
ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90)
ax.axis('equal')

st.pyplot(fig)

from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

st.title("3D Surface Plot Example")
st.write("This is an awesome visualization using NumPy and Matplotlib!")

# Create a slider to adjust the grid size
grid_size = st.slider("Select grid size (resolution)", min_value=10, max_value=100, value=50)

# Generate data for the surface plot
x = np.linspace(-5, 5, grid_size)
y = np.linspace(-5, 5, grid_size)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create the 3D surface plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# Add a color bar and labels
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
ax.set_title("3D Surface Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

# Display the plot in Streamlit
st.pyplot(fig)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Real Estate Investment Analysis Calculator")
st.write("Use this tool to analyze a real estate investment purchase decision and view its performance over time.")

# Inputs
st.header("Property Details")
purchase_price = st.number_input("Purchase Price ($)", min_value=0, value=250000)
down_payment = st.number_input("Down Payment (%)", min_value=0.0, max_value=100.0, value=20.0)
interest_rate = st.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
loan_term = st.number_input("Loan Term (years)", min_value=1, value=30)

st.header("Income & Expenses")
monthly_rent = st.number_input("Monthly Rent ($)", min_value=0, value=2000)
vacancy_rate = st.number_input("Vacancy Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
operating_expenses = st.number_input("Monthly Operating Expenses ($)", min_value=0, value=500)

# Calculations
st.header("Results")

# Loan details
loan_amount = purchase_price * (1 - down_payment / 100)
monthly_interest_rate = (interest_rate / 100) / 12
num_payments = loan_term * 12
monthly_mortgage_payment = (
    loan_amount
    * monthly_interest_rate
    * (1 + monthly_interest_rate) ** num_payments
) / ((1 + monthly_interest_rate) ** num_payments - 1)

# Net operating income (NOI)
vacancy_loss = (vacancy_rate / 100) * monthly_rent
effective_rent = monthly_rent - vacancy_loss
noi = effective_rent - operating_expenses

# Cash flow
monthly_cash_flow = noi - monthly_mortgage_payment
annual_cash_flow = monthly_cash_flow * 12

# Cap rate
cap_rate = (noi * 12 / purchase_price) * 100

# ROI (Cash on Cash Return)
total_cash_investment = (down_payment / 100) * purchase_price
roi = (annual_cash_flow / total_cash_investment) * 100

# Display results
st.subheader("Investment Summary")
st.write(f"**Loan Amount:** ${loan_amount:,.2f}")
st.write(f"**Monthly Mortgage Payment:** ${monthly_mortgage_payment:,.2f}")
st.write(f"**Net Operating Income (NOI):** ${noi:,.2f}/month")
st.write(f"**Monthly Cash Flow:** ${monthly_cash_flow:,.2f}")
st.write(f"**Annual Cash Flow:** ${annual_cash_flow:,.2f}")
st.write(f"**Cap Rate:** {cap_rate:.2f}%")
st.write(f"**ROI (Cash on Cash Return):** {roi:.2f}%")

# Time-based analysis
st.header("Time-Based Analysis")
years = np.arange(1, loan_term + 1)
loan_balance = loan_amount * (1 + monthly_interest_rate) ** (years * 12) - (
    monthly_mortgage_payment
    * ((1 + monthly_interest_rate) ** (years * 12) - 1)
) / monthly_interest_rate
cumulative_cash_flow = annual_cash_flow * years

# Plot loan balance and cumulative cash flow
fig, ax1 = plt.subplots(figsize=(10, 6))

# Loan balance
ax1.plot(years, loan_balance, label="Loan Balance", color="red", linewidth=2)
ax1.set_ylabel("Loan Balance ($)", color="red")
ax1.tick_params(axis="y", labelcolor="red")

# Cumulative cash flow
ax2 = ax1.twinx()
ax2.plot(years, cumulative_cash_flow, label="Cumulative Cash Flow", color="green", linewidth=2)
ax2.set_ylabel("Cumulative Cash Flow ($)", color="green")
ax2.tick_params(axis="y", labelcolor="green")

# Titles and legends
ax1.set_title("Loan Balance and Cumulative Cash Flow Over Time")
ax1.set_xlabel("Years")
fig.tight_layout()

# Display the chart
st.pyplot(fig)

# Conclusion
if monthly_cash_flow > 0:
    st.success("This investment has a positive cash flow!")
else:
    st.error("This investment has a negative cash flow. Consider adjusting your inputs.")
