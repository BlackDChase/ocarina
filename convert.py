import glob
from music21 import converter, instrument, note, chord

NoteList = []

def parse_to_notes(iteration):
    notes = []
    #this is for an output which is seprated by diffrent music files because every file is different
    for files in glob.glob("ocarina_midi_data/*.mid"):
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
'''
def parse_to_notes():
    notes = []
    #this is for an output which is seprated by diffrent music files because every file is different
    for files in glob.glob("ocarina_midi_data/*.mid"):
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
    return notes, NoteList;
'''


