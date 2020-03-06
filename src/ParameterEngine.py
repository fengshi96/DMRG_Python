import ast
import re


def readinput(filedir):
    #filedir = "../input.inp"
    file = open(filedir, 'r')
    lines = file.readlines()
    file.close()

    for i in range(0, len(lines)):
        if '#' in lines[i]:
            pass
        else:
            line = lines[i].strip('\n')
            if re.search("num_sites", line, re.I):
                num_sites = [int(s) for s in line.split("=") if s.isdigit()][0]

            elif re.search("dmax", line, re.I):
                dmax = [int(s) for s in line.split('=') if s.isdigit()][0]

            elif re.search("interaction", line, re.I):
                interaction = ast.literal_eval(line.split('=')[-1])

            elif re.search("field", line, re.I):
                field = ast.literal_eval(line.split('=')[-1])

    return num_sites, dmax, interaction, field

#print(readinput("../input.inp"))
