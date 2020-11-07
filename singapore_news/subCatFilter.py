import re
import configV2
sNum2cat = configV2.sNum2cat
needAdditionalFilter = configV2.needAdditionalFilter
#needAdditionalFilter = []

###filterPest###
mos=re.compile(r"\bmosquito\b|\bmosquitos\b|\bmosquitoes\b",re.I)
rat=re.compile(r"\brat\b|\brodent\b|\brats\b|\brodents\b",re.I)
cock=re.compile(r"\bcockroach\b|\bcockroaches\b",re.I)

###filterNoise###
reno=re.compile(r"\brenovation\b|\bdrilling\b|\brenovating\b",re.I)
neib=re.compile(r"\bneighbour\b|\bneighbours\b",re.I)
cons=re.compile(r"\bconstruction\b",re.I)

###filterAnimal###
cat=re.compile(r"\bcat\b|\bcats\b",re.I)
dog=re.compile(r"\bdog\b|\bdogs\b",re.I)
bird=re.compile(r"\bbird\b|\bcrow\b|\bbirds\b|\bcrows\b|\bpigeon\b|\bpigeons\b",re.I)

###filterPublicTransport###
bus=re.compile(r"\bbus\b|\bbuses\b",re.I)
mrt=re.compile(r"\bmrt\b|\bsmrt\b|\bSMRT_Singapore\b|\btrain\b|\btrains\b",re.I)
taxi=re.compile(r"\bcab\b|\bcabs\b|\btaxi\b|\bgrab\b|\buber\b",re.I)

###filterHDB###
carpark=re.compile(r"\bcarpark\b|\bcarparks\b|\bparking\b|\bpark\b",re.I)
light=re.compile(r"\blight\b|\blighting\b|\bdark\b",re.I)
infra=re.compile(r"\bplayground\b|\blift\b|\bstaircase\b|\bfitness\b|\bwater\b|\bleakage\b|\bdrain\b|\bpower\b",re.I)
clean=re.compile(r"\brubbish\b|\bdirty\b|\bcleaning\b",re.I)
noise=re.compile(r"\bloud\b|\bnoise\b|\bnoisy\b|\brenovation\b",re.I)
animal=re.compile(r"\bcat\b|\bcats\b|\bdogs\b|\bdog\b",re.I)
smoke=re.compile(r"\bsmoking\b|\bsmoke\b",re.I)
est=re.compile(r"\blink\b|\bwebsite\b|\bhuperlink\b",re.I)
neighbour=re.compile(r"\bneighbour\b|\bneighbours\b",re.I)


def findSubCategory(sNum,text,category,agency):
	if sNum not in needAdditionalFilter:
		subCategory = category
	elif category == 'HDB':
		if carpark.search(text) != None:
			subCategory = 'HDB_Carpark'
		elif light.search(text) != None:
			subCategory = 'HDB_Lighting'
		elif infra.search(text) != None:
			subCategory = 'HDB_Infrastructure'
		elif animal.search(text) != None:
	 		category = 'Animal_Issues'
		elif noise.search(text) != None:
	 		category = 'Noise'
		elif clean.search(text) != None:
	 		subCategory = 'HDB_Cleanliness'
		elif smoke.search(text) != None:
	 		subCategory = 'HDB_Smoking'
		elif est.search(text) != None:
	 		subCategory = 'eServices'
		elif neighbour.search(text) != None:
	 		subCategory = 'HDB_Neighbour'
		else:
			subCategory = 'Others'
	else:
		subCategory = category
	
	if category == 'Pest_Control':
		if mos.search(text) != None:
			subCategory = 'PestControl_Mosquito'
		elif rat.search(text) != None:
			subCategory = 'PestControl_Rat'
		elif cock.search(text) != None:
			subCategory = 'PestControl_Cockroach'
		else:
			subCategory = 'PestControl_Others'
	elif category == 'Noise':
		if reno.search(text) != None:
			subCategory = 'Noise_Renovation'
			agency = 'HDB'
		elif neib.search(text) != None:
			subCategory = 'Noise_Neighbour'
			agency = 'HDB'
		elif cons.search(text) != None:
			subCategory = 'Noise_Construction'
			agency = 'LTA'
		else:
			subCategory = 'Noise_Others'
			agency = 'na'
	elif category == 'Animal_Issues':
		if cat.search(text) != None:
			subCategory = 'Animals_Cat'
		elif dog.search(text) != None:
			subCategory = 'Animals_Dog'
		elif bird.search(text) != None:
			subCategory = 'Animals_Bird'
		else:
			subCategory = 'Animals_Others'
	elif category == 'Public_Transport':
		if bus.search(text) != None:
			subCategory = 'PublicTransport_Bus'
		elif mrt.search(text) != None:
			subCategory = 'PublicTransport_MRT'
		elif taxi.search(text) != None:
			subCategory = 'PublicTransport_Taxi'
		else:
			subCategory = 'PublicTransport_Others'

	return (subCategory,agency) 
	
