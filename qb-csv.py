import csv
import json

if __name__ == '__main__':
	with open('Workbook2.csv', 'rU') as infile, open('receiptIDs.json', 'w') as outfile:
		ids = []
		reader = csv.reader(infile)
		for row in reader:
			ids.append(int(row[0]))
		json.dump(ids,outfile)