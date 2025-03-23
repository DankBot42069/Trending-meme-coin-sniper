from gmgn import gmgn

gmgn = gmgn()

getNewPairs = gmgn.getNewPairs(limit=15)
data = gmgn.getNewPairs(limit=15)

print(getNewPairs)
for pair in data['pairs']:
    print(f"Pair ID: {pair['id']} - Base Token: {pair['base_token_info']['name']} ({pair['base_token_info']['symbol']})")
