##SCRAPER TO GET DATA FROM STATCAST: Fill start date and end date

from pybaseball import statcast, cache

cache.enable()
# Define the date range
start_date = "2025-03-20" #mlb 2024 regular season start date
end_date = "2025-04-20"  #mlb 2024 regular season end date

try:
    # Fetch the Statcast data
    data = statcast(start_dt=start_date, end_dt=end_date)

    if not data.empty:
        # Print columns to ensure data was fetched
        print(data.columns)

        # Save the data to a CSV file

        data.to_csv('2025regSeason.csv', index=False)
        print("Data saved to 2025regSeason.csv.")
    else:
        print("No data fetched for this date range.")
except Exception as e:
    print(f"An error occurred: {e}")