import glob
from music21 import converter, instrument, note, chord

notes = []

#this is for an output which is seprated by diffrent music files because every file is different

for files in glob.glob("ocarina_midi_data/*.mid"):
    musicnote = []
    try:
        midi = converter.parse(files)
    except:
        print(str(files), "is bad")
        continue
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
        elif isinstance(element, chord.Chord):
            for cord in element.normalOrder:
                musicnote.append(str(cord))
    
    if len(musicnote)>0:
        notes.append(musicnote)
        #print(musicnote)
    #print("To stop Give 0 : ",end=' ')
    #stopNot = input()
    #if stopNot=='0':
    #    break

with open("data/ocarina.txt", "w+") as file:
    for i in notes:
        file.write(str(i))
        file.write("\n")

def dataLoader():
    notes = []
    #this is for an output which is seprated by diffrent music files because every file is different
    for files in glob.glob("ocarina_midi_data/*.mid"):
        musicnote = []
        try:
            midi = converter.parse(files)
        except:
            continue
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
            elif isinstance(element, chord.Chord):
                for cord in element.normalOrder:
                    musicnote.append(str(cord))
        
        if len(musicnote)>0:
            notes.append(musicnote)
    return notes
