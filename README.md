# ocarina
__Music Generator using LSTM__
- The _midi_ files in `ocarina_midi_data/` are over dataset.
- We have converted these files into notes and chords and save it into `data/ocarina.txt` through a _music21_ library in `convert_mid_to_notes.py`.
	_master_ Branch
		Nearby chords are in the same string and seprated by `.`. This makes our Hash dictionary to have 286 Keys.
	_nodot_ Branch
		Nearby chords are in diffreant string and in text file start with `>`. This makes our Hash dictionary to have 106 Keys.
- This is now used in `model.py` which has CUDA support on.
- This input data is tokenized using a random function to give int values
- The tokenized data is then converted into a tensor `[a , b ,c]`
	- `c` is decided through _n_, this value indicates the number of notes that belong to same music file are going to help genrate next node.
	- `b` is _n//2_, which is the number of data under the same label.
		___more information required___
	- `a` indicates number of possible such labels
		___more information required___
- LSTM model as ofknow has some input output issues, explained by error in line _157_ and hence line _157 to 165_ is just to check for if 1 output is available or not.
- Learning Rate is chosen `0.1`
- Loss criterion is `Cross Entropy Loss`
- `SGD` is used for backporpagation
- Training part is commented out, until Model is fixed.
- `playMusic` function will later have the ability to read the output file and use `music21` to convert this back to _midi_ file, to be played back our own generated melody.

```Some_variable_need_a_thinking_over
batch
n_iters
num_epochs
inpu_dim
hidden_dim
output_dim
seq_dim
```