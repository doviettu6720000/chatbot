import json

# info = open('data/final_data.json',)

# res = json.load(info)
# print(res)

with open('data/final_data.json', 'r', encoding="utf8") as openfile:
  
    # Reading from json file
    json_object = json.load(openfile)
  
print(json_object)