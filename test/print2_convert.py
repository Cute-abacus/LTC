import json

params =  {
    "h": 0.3,
    "q": 30,
    "alpha": 0.3,
    "sample_rate": 0.4,
    "rank": 2,
    "times": 1,
    "ap_cap": 10,
    "sub_rank": 1
}

path = "../data/"

with open(path+"print4") as infp:
    content = []
    for line in infp:
        if line[0] == 'C' or line[0] == 'L' :
            content.append(line.split())

file_dict_list =[]
data = []
i = 0
while i < len(content):
    data = []
    for j in range(i, i+10, 2):
        try:
            run = {}
            run['CP'] = {
                "MeanTrErr": float(content[j][10].strip(',')), # Train
                "MeanPrErr": float(content[j][15].strip(','))  # Predict
            }
            run['LTC'] = {
                "MeanTrErr": float(content[j+1][10].strip(',')), # Train
                "MeanPrErr": float(content[j+1][15].strip(','))  # Predict
            }
            data.append(run)
        except IndexError:
            print("[ERR] Missing a test")




    cur_params = params.copy()
    cur_params['h'] += int(i/10)*.05
    file_dict_list.append({
        'params': cur_params,
        'data': data
    })

    i += 10


# serializing json
with open(path + "print4.json", "w") as outfile:
    json.dump(file_dict_list, outfile)


