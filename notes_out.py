from music21 import converter, instrument, note, chord

def note_finder(data,dic):
    for key in dic.keys():
        if dic[key][0]==data:
            return dic[key][1]
    return None

def make_notes(data,dic):
    notes = []
    for tokened_note in data:
        notes.append(note_finder(tokened_note,dic))
    return notes

def prep_files(data,dic):
    notes = make_notes(data,dic)
    #print(dic)
    return notes
