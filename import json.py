import json

with open(r"C:\Users\Wyfz0\Downloads\CWTS_Admission_Bot\google_creds.json", "r") as f:
    creds = json.load(f)

creds["private_key"] = creds["private_key"].replace("\n", "\\n")

print('GOOGLE_SHEET_CREDS = """')
print(json.dumps(creds, indent=2))
print('"""')