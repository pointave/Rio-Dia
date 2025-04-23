# Rio-Dia
This is a gui for the Dia tts model that generates multiple speakers. Rio-Dia is an easy way to type in your scripts, the limit currently is 30 seconds but its recommended to aim for a shorter script. There is no need to type in the [S1] or [S2], just add a new line and start typing. 

There is a filelist that shows recently generated audio. Feel free to type in a seed to try to preserve speakers.

Installation
1. git clone
2. python -m venv .venv
3. install pytorch
4. pip install -e .
5. Download model and its config from huggingface and add to model folder

To run the program python rio.py. 
