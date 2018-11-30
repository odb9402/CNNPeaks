
#regular bed
def bed_file_load(input_bed, chrom = None):
	"""loading bed files and translate to Python Object"""
	bed_file = open(input_bed,'r')
	peak_data = bed_file.readlines()

	peak_table = ['chr','region_s','region_e','peak_name','score']
	peak_labels = []

	if len(peak_data) is 0:
		return []

	while True:
		if len(peak_data) > 0 and peak_data[0][0] == '#':
			del peak_data[0]
		else:
			break

	for peak in peak_data:
		peak_labels.append(dict(zip(peak_table,peak.split())))

	return peak_labels


#ENCODE narrowPeak
def narrow_peak_file_load(input_Npeak, chrom = None):
	""""""
	Npeak_file = open(input_Npeak, 'r')
	peak_data = Npeak_file.readlines()

	peak_table = ['chr','region_s','region_e','name','score','strand','signalValue','pValue','qValue','peak']
	peak_labels = []

	for peak in peak_data:
		peak_labels.append(dict(zip(peak_table,peak.split())))

	return peak_labels


#ENCODE broadPeak
def broad_peak_file_load(input_Bpeak, chrom = None):
	""""""
	Bpeak_file = open(input_Bpeak, 'r')
	peak_data = Bpeak_file.readlines()

	peak_table = ['chr','region_s','region_e','name','score','strand','signalValue','pValue','qValue']
	peak_labels = []

	for peak in peak_data:
		peak_labels.append(dict(zip(peak_table,peak.split())))

	return peak_labels


#ENCODE gappedPeak
def gapped_peak_file_load(input_Gpeak, chrom = None):
	pass


def run(input_file_name):
	format = input_file_name.rsplit('.', 1)[1]

	if format == "bed":
		return bed_file_load(input_file_name)
	elif format == "narrowPeak":
		return narrow_peak_file_load(input_file_name)
	elif format == "broadPeak":
		return broad_peak_file_load(input_file_name)
	elif format == "gappedPeak":
		return gapped_peak_file_load(input_file_name)
