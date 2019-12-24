import json

with open('copydays_hash_dict_Lou.json', 'r') as f:
	content = json.loads(f.read())

with open('copydays_hash_dict.json', 'w') as f:
	content = json.dumps(content, indent=2, sort_keys=True).replace("'", '"').replace('\n    ', ' ')
	f.write(content)