import pandas as pd
INLAND_CITIES = ['AliceSprings', 'Moree', 'Woomera', 'Uluru']

with open("data/weatherAUS.csv") as w:
    data = pd.DataFrame(pd.read_csv(w))


inland = data[data["Location"].isin(INLAND_CITIES)]
print(len(inland[inland["RainTomorrow"] == "No"]) / len(inland))
