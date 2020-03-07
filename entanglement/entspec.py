import re  # # for passing an argument and list of variables ## regexes, math functions, random numbers
import sys
import ast
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import subprocess
pi = 3.14159265



##### ----------------------
def main(total, cmdargs):
	if total != 1:
			print (" ".join(str(x) for x in cmdargs))
			raise ValueError('I did not ask for arguments')
	#Kz = str(cmdargs[1])


	### -------- onsite observables -------------------
	ES_all = []
	EE_all = []

	# KzRange = np.round(np.arange(1.0, 5.1, 0.1),2)
	HRange = grepfilenum("H", "../runForInput")
	for H in HRange:
		filename="../runForInput/H_"+str('%.2f' % H)+"/runForInput.cout"
		string1 = "Entanglement Spectrum: "

		file = open(filename, 'r')
		lines = file.readlines()
		file.close()

		for i in range(0, len(lines)):
			line = lines[i]

			if re.search(string1, line, re.I):
				matchindex = i+1

		try:
			matchindex
		except NameError:
			print ("***match string", string1)
			print ('***MYERROR:matchindex in read_density function is not defined - no match found')

		line = lines[matchindex].strip(' \n]').split(" ")
		ES = np.zeros(len(line))

		EE = float(lines[matchindex+1].split(" = ")[-1])
		for i in range(len(line)):
			ES[i] = float(line[i])

		ES_all.append(ES)
		EE_all.append(EE)
		#print(ES)
		#print(countES)
		print(str(H)+" Done")
	#print(ES_conter[1])
	#print(len(KzRange))

	#print(EE_all)

	#===========================

	fig = plt.figure(figsize=(8, 8))
	gs = gridspec.GridSpec(2, 1)
	gs.update(left=0.12, right=0.75, top=0.97, bottom=0.08, wspace=0.0, hspace=0.35)
	ax = plt.subplot(111)
	ax.plot(HRange, EE_all, marker='^', color='red', label="E.E.")
	for i in range(len(HRange)): #len(KzRange)
		#ax.scatter(HRange[i], EE_all[i], marker='^', color='red')
		for j in range(len(ES_all[i])):  #len(ES_all[i])
			ax.scatter(HRange[i], np.abs(ES_all[i][j]), marker='_', color='blue')
			#print(ES_conter[i][j,0])
	#ax.set_ylim(ymin=0,ymax=0.1)
	plt.axvline(x=1, color='black', linestyle='--')
	plt.grid(which='major', linestyle='--',alpha=0.35)
	#plt.xscale('log')
	#plt.yscale('log')
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)
	plt.xlabel(r"$H/J$", fontsize=18)
	plt.ylabel(r"$E.S.$", fontsize=18)

	plt.legend(loc='best',ncol=1, fontsize=20, frameon=False,columnspacing=1,handlelength=1.0,handletextpad=0.4)

	plt.savefig('EE.pdf', bbox_inches='tight', dpi=300)

	#for i, txt in enumerate(n):
	    #ax.annotate(txt, (countES[i], y[i]))
#













#### ========================================================================================
#### ========================= Functions ====================================================
#### ========================================================================================
#### ========================================================================================
#### ========================= Functions ====================================================
#### ========================================================================================
def read_density(NumberOfSites,str,matchstring):
	file = open(str,'r')
	lines = file.readlines()
	file.close()
	for i in range(0,len(lines)):
			line = lines[i]
			if re.search(matchstring, line, re.I):
				matchindex=i+1
				break
	try:
		matchindex
	except NameError:
		print ("***match string", matchstring)
		print ('***MYERROR:matchindex in read_density function is not defined - no match found')
	rows = NumberOfSites
	cols = 2
	m = [[0.0 for x in range(cols)] for y in range(rows)] ## allocate m list
	for i in range(0,rows):
		line = lines[matchindex+1+i];
		temp = line.split(" ")
		val = ConvertImag(temp[1])
		m[i][0] = val.real;
		m[i][1] = val.imag;  	### defined 2D Matrix
	return np.asarray(m);

##### ----------------------
def split(mat):
	orb=2
	nrows = mat.shape[0];
	ncols = mat.shape[1]; ##if mat else 0
	mat = np.asarray(mat)
	assert(nrows==ncols)
	fatrows=int(nrows/orb);
	fatcols=int(ncols/orb);

	AA = np.zeros((fatrows,fatcols));
	BB = np.zeros((fatrows,fatcols));
	AB = np.zeros((fatrows,fatcols));

	for i in range(0,fatrows):
		ia = i*orb+0
		ib = i*orb+1
		for j in range(0,fatcols):
			ja = j*orb+0
			jb = j*orb+1

			AA[i,j] = mat[ia,ja]
			BB[i,j] = mat[ib,jb]
			AB[i,j] = mat[ia,jb]

	return AA, BB, AB

##### ----------------------
def ConvertImag(s):
	repart = float(s.split(",")[0].split("(")[1])
	impart = float(s.split(",")[1].split(")")[0])
	return complex(repart,impart)

##### ----------------------
def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

##### ----------------------
def readMatrix(str):
	file = open(str,'r')
	lines = file.readlines()
	file.close()

	Type = lines[0].split(" ")[2]
	Type = ConvertImag(Type);
	rows = int(Type.real)
	cols = int(Type.imag)

	counter = 0
	m = np.zeros((rows,cols)) ### [[0.0 for x in range(rows)] for y in range (cols)]
	for i in range(1,rows+1):
		line = lines[i];
		temp = line.split(" ")
		for j in range(0,cols):
			#m[j+i*cols] = float(temp[j])  ### defined 1D aray
			val = ConvertImag(temp[j]);
			m[i-1,j] = val.real	   ### defined 2D Matrix
		counter = counter + 1
	file.close()

	for i in range(0,rows):
		for j in range(i,cols):
			m[j,i] = m[i,j]	   ### defined 2D Matrix
			if(i==j):
				m[i,j] = 2.0;
	return m;
##### ----------------------


def PrintMatrix(m):
	string="""
DegreesOfFreedom=1
GeometryKind=LongRange
GeometryOptions=none
Connectors"""

	num_rows = len(m);
	num_cols = m.shape[1] #len(m[0]) if m else 0
	string += " "+str(num_rows)+" "+str(num_cols)
	print (string )
	for i in range(0,num_rows):
		print (" ".join(str(x) for x in m[i]))


def grepfilenum(keyword, filedir):
		List = []
		ls = subprocess.Popen(["ls", filedir], stdout=subprocess.PIPE)
		grep = subprocess.Popen(["grep", keyword],stdin=ls.stdout, stdout=subprocess.PIPE)
		endOfPipe = grep.stdout
		for line in endOfPipe:
	    		List.append(str(line).strip("b'n\$"))
		List.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
		res = [float(sub.split('_')[1]) for sub in List]
		return res
##### ----------------------


if __name__ == '__main__':
	sys.argv ## get the input argument
	total = len(sys.argv)
	cmdargs = sys.argv
	main(total, cmdargs)
































