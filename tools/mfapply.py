#! /usr/bin/env python

import math
import sys

#############################################
# Define the following parameter
#############################################

separator = ' '

##############################################

usage = """Applies a function to each value of an .mmc file. The function
body is specified as an argument to this program (variable
name: x).

Usage  : ./mfapply.py <input_file> <output_file> <function_body>
Example: ./mfapply.py input.mmc output.mmc \"math.log(x, 10)+1\""""

if len(sys.argv) != 4:
    print usage
    sys.exit()
    
input_file_name = sys.argv[1]
output_file_name = sys.argv[2]
function_body = sys.argv[3]    
exec("f = lambda x : " + function_body)

def convert(s):
    entries = s.split()
    return entries[0] + separator + entries[1] + separator + str( f(float(entries[2])) )

in_file = open(input_file_name, 'r')
out_file = open(output_file_name, 'w')

line = in_file.readline()

# Skip comments
while line[0] == '%':
    out_file.write(line)
    line = in_file.readline()
    
# Write dimensions unmodified
out_file.write(line)

line = in_file.readline().strip()
while line != '':
    out_file.write(convert(line)+'\n')
    line = in_file.readline().strip()

in_file.close()
out_file.close()
