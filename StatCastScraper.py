##SCRAPER TO GET DATA FROM STATCAST: Fill start date and end date

from pybaseball import statcast, cache
import pandas as pd
cache.enable()
# Define the date range
start_date = "2024-03-28" #mlb 2024 regular season start date
end_date = "2024-09-30"  #mlb 2024 regular season end date

try:
    # Fetch the Statcast data
    data = statcast(start_dt=start_date, end_dt=end_date)

    if not data.empty:
        # Print columns to ensure data was fetched
        print(data.columns)

        # Save the data to a CSV file
        data.to_csv('2024regSeason.csv', index=False)
        print("Data saved to 2024regSeason.csv.")
    else:
        print("No data fetched for this date range.")
except Exception as e:
    print(f"An error occurred: {e}")