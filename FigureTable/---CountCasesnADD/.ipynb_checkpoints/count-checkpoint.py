import csv

PD,FTD,VD,LBD, DLB, PDD,other = 0, 0, 0, 0, 0, 0, 0
with open('../../lookupcsv/dataset_table/NACC_ALL/NACC.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        if row['AD'] == '0' and row['DLB'] == '1':
            DLB += 1
        if row['AD'] == '0' and row['LBD'] == '1':
            LBD += 1
        if row['AD'] == '0' and row['PDD'] == '1':
            PDD += 1
        if row['AD'] == '0' and row['PD'] == '1':
            PD+=1
        if row['AD'] == '0' and row['FTD'] == '1':
            FTD+=1
        if row['AD'] == '0' and row['OTHER'] == '1':
            other+=1
        if row['AD'] == '0' and row['VD'] == '1':
            VD+=1
print("LBD", LBD)
print("DLB", DLB)
print("PDD", PDD)
print("PD", PDD)
print("VD", PDD)
print("Other", PDD)
print("FTD", PDD)

