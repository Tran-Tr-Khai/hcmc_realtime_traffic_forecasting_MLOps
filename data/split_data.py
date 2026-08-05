import json
from pathlib import Path


DATA_PATH = Path(__file__).with_name("hcmc-traffic-data.json")
HISTORY_PATH = Path(__file__).with_name("hcmc-traffic-history.json")
REALTIME_PATH = Path(__file__).with_name("hcmc-traffic-realtime.json")

def read_json(path): 
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

def get_latest_date_key(data):
    data_keys = list(data.keys())
    return data_keys[-1]

def split_latest_day(data):
    latest_date_key = get_latest_date_key(data)
    latest_day_data = {latest_date_key: data[latest_date_key]}

    with REALTIME_PATH.open("w", encoding="utf-8") as file:
        json.dump(latest_day_data, file, ensure_ascii=False, indent=2)

    data.pop(latest_date_key)

    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    print(f"\nLatest date key: {latest_date_key}")
    print(f"Saved file: {REALTIME_PATH}")
   
def main():
    data = read_json(DATA_PATH)
    print(data.keys())
    split_latest_day(data)


if __name__ == "__main__":
    main()