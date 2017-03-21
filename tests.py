#!/usr/bin/python3
from sortekat import SorteKat
from time import time

clf = SorteKat()
print("dette tar nesten 20 sekunder")


t1 = time()
texts = ['Hardangervidda er midelertidig stengt mens brøytamanskap måker veien etter kraftig snøvær. Nedbør og sol. Solfaktor. Snø. Nedbør. Vind.',
     "Statsminister Stoltenberg besøkte bedrifter på vestlandet.",
    "Brann vant kveldens kamp not Drammen.",
    "Arbeidsledighetstallene går ned viser nye tall fra SSB.",
    "Ny forskning viser at profesorer forsker mer.",
    "Den nye storfilmen fra ",
    "Italia er rangert som verdens sunneste land i Bloombergs «Global Health Index», som omfatter 163 land. En nyfødt i Italia kan se frem til å leve i godt over 80 år. Det er omtrent 30 år lenger enn barn født i Sierra Leone som i gjennomsnitt vil dø når de er 52 år.",
    "Bak de store glassrutene med solen rett på ble det til slutt så varmt at duoen bestemte seg for å ta lunsjen der ute. Med pledd til disposisjon gikk det helt fint, selv midt i mars. Maten kom overraskende fort. Pepper hadde gått for salaten «Caesar-ish», Salt for en burger med blåmuggost og bacon.",
    "– Eg kastar opp viss eg må gå på do der Over heile landet fortvilar bussjåførar over skitne toalett, manglande vatn og såpe eller ingen sanitære forhold i det heile."]

print("Men klassifisering er raskt")
pred = clf.predict_one(texts[0])
print("..."+texts[0]+"...")
print("er", pred[0], "og det tok bare", time()-t1, "sekunder")
t1 = time()
print()
preds = clf.predict_many(texts)
for p, t in zip(preds, texts):
    print(p,"==>", t)
print()
print("og", len(texts), "tok bare", time()-t1, "sekunder")
