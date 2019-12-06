import glob
from music21 import converter, instrument, note, chord
import torch.nn as nn
import torch as pt
import numpy as np
from random import randint

def make_dictonary(keys,dic):
    for i in range(len(keys)):
        temp = []
        for z in range(len(keys)):
            if(z==i):
                temp.append(1)
            else:
                temp.append(0)
        dic[str(keys[i])] = temp
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
        notes_to_parse = None
        parts = instrument.partitionByInstrument(midi)
        if parts:  
            # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else:
            # file has notes in a flat structure
            notes_to_parse = midi.flat.notes
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                musicnote.append(str(element.pitch))
                if element.pitch not in NoteList:
                    NoteList.append(element.pitch)
            elif isinstance(element, chord.Chord):
                for cord in element.normalOrder:
                    musicnote.append(str(cord))
                    if cord not in NoteList:
                        NoteList.append(cord)
        if len(musicnote)>0:
            notes.append(musicnote)
        if iteration == 0:
            break
    return notes, NoteList;

def transform(n_files_to_load,dic):
    keys = []
    unorganised_data, keys = parse_to_notes(n_files_to_load)
    dic = make_dictonary(keys,dic)
    data = []
    it = -1
    for music_file in unorganised_data:
        diced_data = []
        randomized_data = []
        check_list = []
        it +=1
        if len(music_file)<199:
            #to be able to have atleast 100 sets of 100 notes
            print("File too small : ",it)
            continue
        print("Tokenising file no : ",it)
        tokenized_notes = []
        for notes in music_file:
            tokenized_notes.append(dic[str(notes)])
        for i in range(len(music_file)-99):
            #for making sets of 100 notes
            noteSet = []
            for j in range(i,i+100):
                noteSet.append(tokenized_notes[j])
            diced_data.append(noteSet)
        for i in range(100):
            random_number = randint(0,len(diced_data)-1)
            while(random_number in check_list):
                random_number = randint(0,len(diced_data)-1)                
            check_list.append(random_number)
            randomized_data.append(diced_data[random_number])
        data.append(randomized_data)
    return dic, data
