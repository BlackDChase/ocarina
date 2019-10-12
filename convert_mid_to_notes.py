import glob
from music21 import converter, instrument, note, chord

notes = []
for file in glob.glob("ocarina_midi_data/*.mid"):
    try:
        midi = converter.parse(file)
    except:
        print(str(file), "is bad")
        continue
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts:  # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))
print(notes)
with open("data/ocarina.txt", "w+") as file:
    file.write(str(notes))


'''
for file in glob.glob("ocarina_midi_data/*.mid"):
    musicnote = []
    try:
        midi = converter.parse(file)
    except:
        print(str(file), "is bad")
        continue
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts:  # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            musicnote.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            musicnote.append('.'.join(str(n) for n in element.normalOrder))
    if len(musicnote>0):
        notes.append(musicnote)
print(notes)
with open("data/ocarina.txt", "w+") as file:
    for i in notes:
        file.write(str(i))
'''

