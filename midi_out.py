from music21 import converter, instrument, note, chord

def note_finder(data,dic):
    for key in dic.keys():
        if dic[key]==data:
            return key

def refurbished(data,dic):
    if data in dic.values():
        return note_finder(data,dic)
    temp = []
    for lists in data:
        temp.append(refurbished(lists,dic))
    return temp

def make_music(data):
    if(type(data[0])!=list):
        #code
        return
    for lists in data:
        make_music(lists)

def prep_files(data,dic):
    refurbished_data = refurbished(data,dic)
    print("Making Music")
    make_music(refurbished_data)
