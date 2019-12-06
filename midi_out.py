from music21 import converter, instrument, note, chord

def note_finder(data,dic):
    for key in dic.keys():
        if dic[key]==data:
            return key
    return None

def refurbished(data,dic):
    refurbished_data = []
    for tokened_note in data:
        refurbished_data.append(note_finder(tokened_note,dic))
    return refurbished_data

def make_music(data):
    if(type(data[0])!=list):
        #code
        return
    for lists in data:
        make_music(lists)

def prep_files(data,dic):
    refurbished_data = refurbished(data,dic)
    #print(dic)
    #print(len(data),len(data[0]),data[0])
    print("Making Music")
    print(refurbished_data)
    music = make_music(refurbished_data)
    return music
