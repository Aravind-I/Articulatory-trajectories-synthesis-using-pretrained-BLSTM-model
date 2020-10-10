# Articulatroy-trajectories-synthesis-using-pretrained-model-using-BLSTM
wav2ema.py performs the following operations:
- input: 'test.wav' (default) audio file (will be read using librosa library, specify sample rate if needed, by default it is 22khz)
- loads a trained model and performs evaulation on the mfcc data
- Optional: uncomment line 32 for adding global stats from training set(from a single subject)
- output: 12 dim articulatory features:
Upper Lip (UL), Lower Lip (LP), Jaw, Tongue Tip (TT), Tongue Body (TB), and Tongue Dorsum (TD)
ULx, ULy, LLx, LLy, Jawx, Jawy, TTx, TTy, TBx, TBy, TDx, TDy
dumps the result as a .npy (can be loaded with np.load()).
