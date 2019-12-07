from music21 import converter, instrument, note, chord, stream

def create_music_object(notes):
    song_offset = 0
    
    # create note and chord objects based on the values generated by the model
    for music_element in notes:
        if music_element==note.Rest:
            new_rest = note.Rest()
            new_rest.offset = song_offset
            music_element = new_rest
        else:
            new_noteOrCord = music_element
            new_noteOrCord.offset = song_offset
            music_element = new_noteOrCord
        #i think not needed as we used the similar kind of dictonary (NoteList/dic)
        '''
        # music element is a chord
        elif '|' in music_element:
            notes_in_chord = music_element.split('|')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = song_offset
            music_objects.append(new_chord)
        # music element is a note
        else:
            new_note = note.Note(music_element)
            new_note.offset = song_offset
            new_note.storedInstrument = instrument.Piano()
            music_objects.append(new_note)
        '''
        # increase offset each iteration so that notes do not stack
        song_offset += 0.5

    return notes


def make(notes):
    #code to convert it into music
    music = create_music_object(notes)
    
    write_to_file(music)
    return
    '''
    command_book = {'q': return,
                    'w': write_to_file(music)#calling the play function,
                    'p': #call write to file function,
    }
    while(1)
        command = input("Next command: ")
        if command in command_book.keys():
            command_book[command]
        else:
            print("Try Again")
    '''
### Write to file
def write_to_file(music_objects):
    #file_name = input("Enter a file name: ")
    file_name = "test"
    file_name+=".midi"
    midi_stream = stream.Stream(music_objects)
    midi_stream.write('midi', fp=file_name)
### Play file
