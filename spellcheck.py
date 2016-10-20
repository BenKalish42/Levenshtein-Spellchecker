# __author__ = Benjamin Kalish

import sys
import re
import os
import time
import string
import math
from random import randint
import csv #this would be for reading our text files
import numpy as np #this would be for computing levenshtein_distance (multi-dimensional arrays are easier with these)
import matplotlib.pyplot as plt #this would be for making many of the plots required in this assignment

# Global Constants

DELETION = 1
INSERTION = 1
SUBSTITUTION = 1

def LevenshteinDistance(s, t, deletionCost=DELETION, insertionCost=INSERTION, substitutionCost=SUBSTITUTION):
	"""Calculates the Levenshtein Distance between two strings"""
	# for all i and j, d[i,j] will hold the Levenshtein distance between
	#   the first i characters of s and the first j characters of t
	# NOTE: the standard approach is to set
	# Default: deletionCost = insertionCost = substitutionCost = 1
	m = len(s)
	n = len(t)

	d = np.zeros([(m+1), (n+1)], np.int32)
	for i in range(0, m+1):
		d[i, 0]=i*deletionCost
	for j in range(0, n+1):
		d[0, j]=j*insertionCost
	# print d
	for j in range(0, n):
		for i in range(0, m):
			if s[i] == t[j]:
				d[i+1, j+1] = d[i, j]
			else:
				d[i+1, j+1] = min((d[i, j+1]+deletionCost), (d[i+1, j]+insertionCost), (d[i, j]+substitutionCost))
	# print d
	return d[m, n]

def qwerty_levenshtein_distance(s, t, deletionCost=1, insertionCost=1):
	KEYBOARD = np.array([['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
				['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ' '], 
				['z', 'x', 'c', 'v', 'b', 'n', 'm', ' ', ' ', ' ']])

	m = len(s)
	n = len(t)

	d = np.zeros([(m+1), (n+1)], np.int32)
	for i in range(0, m+1):
		d[i, 0]=i*deletionCost
	for j in range(0, n+1):
		d[0, j]=j*insertionCost
	# print d
	for j in range(0, n):
		for i in range(0, m):
			if s[i] == t[j]:
				d[i+1, j+1] = d[i, j]
			else:
				#calculate QWERTY substitution cost
				s_tuple = np.where(KEYBOARD == s[i])
				t_tuple = np.where(KEYBOARD == t[j])
				substitutionCost = int(math.fabs(s_tuple[0][0] - t_tuple[0][0]) + math.fabs(s_tuple[1][0] - t_tuple[1][0]))
				d[i+1, j+1] = min((d[i, j+1]+deletionCost), (d[i+1, j]+insertionCost), (d[i, j]+substitutionCost))
	# print d
	return d[m, n]

def spellcheckText():
	"""Spellchecker built for problem 2B"""
	textfile, rawfile = parseText(1, True)
	dictionary = parseDictionary(2)
	print textfile
	newWordDict = {}
	for i in range(0, len(textfile)):
		new_word = find_closest_word(textfile[i], dictionary)
		if new_word != textfile[i]:
			newWordDict[textfile[i]] = new_word
		# rawfile = rawfile.replace(textfile[i], new_word)
	rawfile = rawfileReplace(rawfile, newWordDict)
	print rawfile
	newfile = open("corrected.txt", 'w')
	newfile.write(rawfile)

def rawfileReplace(text, dictionary):
    """ helper for spellcheckText, replaces words in formatted text with correct spelling while preserving punctuation"""
    replace = re.compile('|'.join(map(re.escape, dictionary)))
    def tl(match):
        return dictionary[match.group(0)]
    return replace.sub(tl, text)



def find_closest_word(string1, dictionary):
	""" Helper for spellcheckText, finds word closest to a string in a dictionary"""
	closest_word = None
	closest_distance = 100

	# ***experimental***
	# firstLetterDict = filter(lambda s: s[0] == string1[0], dictionary)
	# for word in firstLetterDict:
	# 	if word == string1:
	# 		closest_word = word
	# 		closest_distance = 0
	# 		break
	# 	current_distance = LevenshteinDistance(string1, word)
	# 	if  current_distance < closest_distance:
	# 		closest_word = word
	# 		closest_distance = current_distance
	# if closest_distance == 1:
	# 	return closest_word

	for word in dictionary:
		if word == string1:
			closest_word = word
			closest_distance = 0
			break
		current_distance = LevenshteinDistance(string1, word)
		if  current_distance < closest_distance:
			closest_word = word
			closest_distance = current_distance
	return closest_word


def measure_error(typos, truewords, dictionarywords, numtrials):
	"""find whethers the corrected typos using the dictionary matches the true word and reports the error rate"""
	errors = 0
	trials = numtrials		# will normally be len(typos)
	# start_point = randint(0, len(typos) - trials)
	for i in range(0, trials): # start_point, start_point + trials
		word_num = randint(0, len(typos))
		guess_word = find_closest_word(typos[word_num], dictionarywords)
		
		possibletruewords = truewords[word_num].split(', ')
		match = False
		for j in range(0, len(possibletruewords)):
			if guess_word == possibletruewords[j]:
				match = True
				break
		if match == False:
			errors += 1
		print "\nGuess Word: " + guess_word + "\nTrue Word: " + truewords[word_num] + "\nCorrect: " + str(match)
	error_rate = float(errors) / float(trials)
	return error_rate


def test_measure_error_time(typos, truewords, dictionarywords):
	x = []
	y = []
	error_rate_sum = 0
	for i in range(10, 26, 5):
		for j in range (0, 3):
			x.append(i/5)
			start = time.time()
			error_rate_sum += measure_error(typos, truewords, dictionarywords, i)
			end = time.time() - start
			y.append(end)
	print x
	print y
	fitline = np.polyfit(x, y, 1)
	print fitline
	error_rate = float(error_rate_sum)/float(12)
	print error_rate
	plt.plot(x, y)
	return fitline, error_rate


def reduced_permutations(valueSet):
	"""Eliminates redundant permutations from sets of costs"""
	testSet = []
	permutations = []
	for i in range(0, 3):
		testSet.append(valueSet)
	for x in testSet[0]:
		for y in testSet[1]:
			for z in testSet[2]:
				if x%2!=0 or y%2!=0 or z%2!=0:
					permutations.append([x, y, z])
	print len(permutations)
	return permutations


def part_3C_experiment():

	typoset = parseDictionary(1)
	truewords = parseDictionary(1, 2)
	dictionary = parseDictionary(2)
	permutations = reduced_permutations([0, 1, 2, 4])
	permutations.pop(0)
	results = []
	for i in range(0, len(permutations)):
		DELETION = permutations[i][0]
		INSERTION = permutations[i][1]
		SUBSTITUTION = permutations[i][2]
		print "\n\nTrial : #" + str(i+1) + "\nPermutation: "
		print "Deletion Cost: " + str(DELETION) + "\nInsertion Cost: " + str(INSERTION) + "\nSubstitution Cost: " + str(SUBSTITUTION)
		results.append(measure_error(typoset, truewords, dictionary, 15))
		print "Error Rate: " + str(results[0] * 100) + "%"
	best_cost_set_index = results.index(max(results))
	print "Cost Values with least error: " + str(permutations[best_cost_set_index])
	print "Error Rate: " + str(min(results)*100) + "%"
	return permutations, results

# File Parsers


def parseDictionary(arg_num=1, column=1):
	inputFile = sys.argv[arg_num]
	dictList = []
	with open(inputFile) as csvfile:
		fileread = csv.reader(csvfile, delimiter='	')
		for row in fileread:
			dictList.append(row[column-1])
	return dictList

def parseText(arg_num=1, ret_file=False):
	inputFile = sys.argv[arg_num]
	dictList = []
	regexp=re.compile(r'[A-Za-z\']+(?:\`[A-Za-z]+)?')
	with open(inputFile) as openfile:
	 	fileread = openfile.read()
		parsedText = regexp.findall(fileread)
	if(ret_file):
		return parsedText, fileread
	else:
		return parsedText

# MAIN EXECUTION


if __name__ == '__main__':

	spellcheckText()

	# 3C Test
	# start = time.time()
	# permutations1, results1 = part_3C_experiment()
	# for p in range(0, len(permutations1)):
	# 	permutations1[p] = str(permutations1[p])
	# print "Time: " + str(time.time() - start)
	# with open("3cResults.txt", 'w') as newfile:
	# 	newfile.write("Error Rate: " + str(min(results1)*100) + "\n{0}".format(dict(zip(permutations1, results1))))
	

