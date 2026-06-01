# # Data Handling Mini Project using NumPy and Pandas
# 
# ## Project Overview
# 
# I am excited to work on this mini project as an opportunity to apply the concepts I have learned in NumPy and Pandas to a real-world data handling task.
# 
# Throughout this project, I will explore how data can be loaded, cleaned, transformed, analyzed, and visualized using Python's powerful data analysis libraries. This project will help strengthen my understanding of data structures, numerical computation, and data manipulation techniques that are essential in data science and artificial intelligence.
# 
# ## Objectives
# 
# - Practice working with datasets using Pandas.
# - Perform data cleaning and preprocessing.
# - Use NumPy for efficient numerical operations.
# - Generate meaningful insights from data.
# - Improve problem-solving and analytical skills through hands-on implementation.
# 
# ## Learning Goals
# 
# By completing this project, I aim to:
# 
# - Gain confidence in handling real datasets.
# - Understand how NumPy and Pandas work together.
# - Learn best practices for organizing data analysis workflows.
# - Build a strong foundation for future machine learning and AI projects.
# 
# Let's begin the analysis.

# # Global Countries Analysis
# 
# An exploratory data analysis of countries across continents, regions, governments, demographics, and political structures using Python and Pandas.

import numpy as np 
import pandas as pd 

#Now lets read the file so we can extract its content
df = pd.read_csv(r"Countries.csv")

# ## Dataset Overview
# 
# Before conducting analysis, the dataset was inspected to understand its structure, data types, and overall quality.
# 
# The inspection included:
# 
# - Dataset dimensions
# - Column information and data types
# - Summary statistics
# - Missing value identification
# - Duplicate record detection
# 
# These checks help ensure that subsequent analyses are based on reliable and well-understood data.

#View dimensions of matrix
df.shape

df.columns

df.nunique()

df.info()

df.describe()

df.head()

df.tail()

df.isna().sum().sort_values(ascending=False)

df.duplicated().sum()

# # Population and Demographics
# 
# Population size, age structure, and fertility patterns provide important insights into demographic trends and future population growth.
# 
# This section explores population distribution, median age, and fertility patterns across countries and continents.

df.sort_values("population",ascending=False)[["country","population"]].head(10)

df.sort_values("population")[["country","population"]].head(10)

df.groupby("continent")["population"].sum().sort_values(ascending=False)

df.groupby("continent")["median_age"].mean().sort_values(ascending=False)

df.sort_values("fertility_rate")[["country","fertility_rate"]].head(10)

df.sort_values("fertility_rate")[["country","fertility_rate"]].tail(10)

# ### Fertility Patterns
# 
# Fertility rates vary significantly across countries and regions.
# 
# Countries with high fertility rates typically experience faster population growth and younger age structures, while countries with low fertility rates often face aging populations and slower demographic expansion.
# 
# These differences can have major implications for workforce size, economic development, healthcare systems, and long-term population sustainability.

# # Democracy and Governance
# 
# Political institutions play a major role in shaping national development.
# 
# This section explores how democratic systems vary across countries and continents, examining democracy scores, governance structures, and the distribution of political systems around the world.

df.sort_values("democracy_score")[["country","democracy_score"]].head(10)

df.sort_values("democracy_score")[["country","democracy_score"]].tail(10)

# ### Global Democracy Rankings
# 
# Democracy scores provide a standardized measure of political freedoms, electoral processes, government functioning, and civil liberties.
# 
# Comparing the highest- and lowest-scoring countries helps identify major differences in governance quality across the world.

df.groupby("continent")[["democracy_score"]].mean()

# ### Continental Comparison
# 
# Democracy levels are not distributed evenly across continents.
# 
# By examining average democracy scores at the continental level, we can identify broader regional trends in governance and political development.

df.groupby("democracy_type")[["democracy_score"]].mean()

df["democracy_type"].value_counts()

# ### Distribution of Political Systems
# 
# Countries are classified into different democracy categories based on their political institutions and democratic performance.
# 
# Examining the frequency of each category provides insight into the global balance between stronger and weaker democratic systems.

# ## Democracy and Development
# 
# Democracy is often associated with improvements in education, healthcare, economic performance, and access to information.
# 
# To explore these relationships, average life expectancy, internet access, and GDP are compared across different democracy categories.

df.groupby("democracy_type")[["life_expectancy"]].mean().sort_values("life_expectancy", ascending=False)

df.groupby("democracy_type")[["internet_pct"]].mean().sort_values("internet_pct", ascending=False)

df.groupby("democracy_type")[["gdp"]].mean().sort_values("gdp", ascending=False)

# ### Observations
# 
# Comparing development indicators across democracy categories helps identify broader governance patterns.
# 
# While these results do not establish causation, they provide insight into whether more democratic systems tend to be associated with higher living standards, greater connectivity, and stronger economic performance.
# 
# Further analysis would be required to determine the strength and significance of these relationships.

# # Health and Quality of Life
# 
# Health indicators provide insight into the well-being of populations and the effectiveness of national healthcare systems.
# 
# This section explores life expectancy, healthcare spending, and medical infrastructure across countries and continents.

df.sort_values("life_expectancy", ascending=False)[["country","life_expectancy"]].head(10)

df.sort_values("life_expectancy")[["country","life_expectancy"]].head(10)

df.groupby("continent")[["life_expectancy"]].mean().sort_values("life_expectancy", ascending=False)

# ### Life Expectancy Analysis
# 
# Life expectancy is one of the most widely used measures of overall development and public health.
# 
# Countries with high life expectancy often benefit from stronger healthcare systems, better living conditions, and improved access to education and nutrition.

df.groupby("continent")[["hospital_beds"]].mean().sort_values("hospital_beds", ascending=False)

df.groupby("continent")[["health_expenditure_pct_gdp"]].mean().sort_values("health_expenditure_pct_gdp", ascending=False)

# ### Continental Health Comparison
# 
# Comparing life expectancy across continents highlights regional differences in health outcomes and development levels.
# 
# These differences may be influenced by economic conditions, healthcare accessibility, education levels, and political stability.

# # Economic Analysis
# 
# Economic indicators provide valuable insight into national prosperity, labor market conditions, and fiscal capacity.
# 
# This section examines GDP, inflation, unemployment, and tax revenue patterns across countries and continents.
# 
# ### Key Areas Explored
# 
# - Countries with the highest and lowest GDP
# - Inflation extremes across the world
# - Continental differences in unemployment rates
# - Tax revenue patterns by continent
# 
# Together, these indicators provide a broader picture of global economic development and stability.

# Top 10 countries by GDP
df.sort_values("gdp", ascending=False)[["country","gdp"]].head(10)

# Bottom 10 countries by GDP
df.sort_values("gdp")[["country","gdp"]].head(10)

# Average GDP by continent
df.groupby("continent")[["gdp"]].mean().sort_values("gdp", ascending=False)

# Top 10 highest inflation countries
df.sort_values("inflation", ascending=False)[["country","inflation"]].head(10)

# Top 10 lowest inflation countries
df.sort_values("inflation")[["country","inflation"]].head(10)

# Average unemployment by continent
df.groupby("continent")[["unemployment_pct"]].mean().sort_values("unemployment_pct", ascending=False)

# Average tax revenue percentage by continent
df.groupby("continent")[["tax_revenue_pct_gdp"]].mean().sort_values("tax_revenue_pct_gdp", ascending=False)

# # Energy and Environment
# 
# Energy production and environmental indicators reveal how countries balance economic growth, energy demand, and environmental sustainability.
# 
# This section explores renewable energy usage, fossil fuel dependence, and greenhouse gas emissions across countries and continents.

# Top 10 renewable energy consumers
df.sort_values("renewable_energy_consumption_pct", ascending=False)[["country","renewable_energy_consumption_pct"]].head(10)

# Top 10 fossil fuel consumers
df.sort_values("fossil_energy_consumption_pct", ascending=False)[["country","fossil_energy_consumption_pct"]].head(10)

# Average renewable energy consumption by continent
df.groupby("continent")[["renewable_energy_consumption_pct"]].mean().sort_values("renewable_energy_consumption_pct", ascending=False)

# Average fossil fuel consumption by continent
df.groupby("continent")[["fossil_energy_consumption_pct"]].mean().sort_values("fossil_energy_consumption_pct", ascending=False)

# Top 10 CO2 emitters
df.sort_values("co2_emissions", ascending=False)[["country","co2_emissions"]].head(10)

# Average CO2 emissions by continent
df.groupby("continent")[["co2_emissions"]].mean().sort_values("co2_emissions", ascending=False)

# # Key Findings
# 
# ## Demographics
# - Population is highly concentrated among a small number of countries.
# - Fertility rates vary significantly across regions.
# 
# ## Governance
# - Democracy scores differ considerably across continents.
# - Certain democracy types are substantially more common than others.
# 
# ## Health
# - Life expectancy shows clear regional variation.
# - Healthcare expenditure differs significantly across continents.
# 
# ## Economy
# - GDP is heavily concentrated among a small number of economies.
# - Inflation and unemployment vary widely across countries.
# 
# ## Energy & Environment
# - Renewable energy adoption differs substantially across regions.
# - Major CO₂ emissions are concentrated among a limited number of countries.

