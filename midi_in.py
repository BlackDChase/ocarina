import glob
from music21 import converter, instrument, note, chord
import torch.nn as nn
import torch as pt
import numpy as np
from random import randint
from copy import deepcopy

def make_dictonary(keys,dic):
    for i in range(len(keys)):
        temp = []
        for z in range(len(keys)):
            if(z==i):
                temp.append(1)
            else:
                temp.append(0)
        dic[str(keys[i])] = (temp,keys[i])
    return dic

def parse_to_notes(iteration):
    NoteList = []
    notes = []
    #this is for an output which is seprated by diffrent music files because every file is different
    for files in glob.glob("ocarina_midi_input_data/*.mid"):
        iteration -= 1
        musicnote = []
        try:
            midi = converter.parse(files)
        except:
            print("Bad file : ",str(files))
            continue
        print("Parsing : ",str(files))

        for element in midi.recurse():
            note = deepcopy(element)
            musicnote.append(note)
            if note not in NoteList:
                NoteList.append(note)
                
        if len(musicnote)>0:
            notes.append(musicnote)
        if iteration == 0:
            break
    return notes,NoteList

def transform(n_files_to_load,dic):
    keys = []
    unorganised_data, keys = parse_to_notes(n_files_to_load)
    print("Files loaded, making referance dictonary")
    dic = make_dictonary(keys,dic)
    data = []
    it = -1
    for music_file in unorganised_data:
        it +=1
        if len(music_file)<99:
            #to be able to have atleast 100 sets notes
            print("\tFile too small : ",it)
            continue
        print("Tokenising file no : ",it)
        tokenized_notes = []
        for notes in music_file:
            tokenized_notes.append(dic[str(keys[keys.index(notes)])][0])
        for i in range(len(music_file)-99):
            #for making sets of 100 notes
            noteSet = []
            for j in range(i,i+100):
                noteSet.append(tokenized_notes[j])
            data.append(noteSet)
    return data,dic

def transform_test(dic):
    keys = []
    unorganised_data, keys = parse_to_notes(1)
    dic = make_dictonary(keys,dic)
    
    #for key in dic.keys():
    #    print(key,dic[key])
    
    print("Referrance dictionary made")
    tokenized_notes = []
    unorganised_data = unorganised_data[0]
    for notes in unorganised_data:
        tokenized_notes.append(dic[str(keys[keys.index(notes)])][0])
    return tokenized_notes,dic
