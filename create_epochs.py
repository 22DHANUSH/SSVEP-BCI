import numpy as np
import mne
import scipy.io as sio
import matplotlib.pyplot as plt

# Load the electrode positions from the .loc file
loc_file = 'data/64-channels.loc'
electrode_data = np.genfromtxt(loc_file, dtype=None, encoding='utf-8')

# Extract channel names and positions
indices, angles, radii, ch_names = zip(*electrode_data)

# Convert angles and radii to 3D Cartesian coordinatesś
def polar_to_cartesian(angle, radius):
    angle_rad = np.deg2rad(angle)
    radius = radius/5

    x = radius * np.sin(angle_rad)
    y = radius * np.cos(angle_rad)
    z = 0  # (2D plot) Assuming all electrodes are on the scalp surface
    return x, y, z

positions = np.array([polar_to_cartesian(angle, radius) for angle, radius in zip(angles, radii)])

# Create a dictionary of channel positions
ch_pos = {ch_names[i]: positions[i] for i in range(len(ch_names))}

# Create a montag
montage = mne.channels.make_dig_montage(ch_pos, coord_frame='head')

# Verify the montage
montage.plot(kind='topomap')  # You can use kind='topomap' for a 2D plot

# Create an info structure
sfreq = 250  # downsampled to 250 Hz
n_channels = 64
# ch_names = ['O1', 'Oz','O2']
info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types=['eeg'] * n_channels)

# info = mne.pick_channels(info['ch_names'], selected_channels)
# Apply the montage to the info
info.set_montage(montage)
info

all_epochs = []

for i in range(1,36):
    # Load subject data
    subject_index = f'S{i}'  # Example for the first subject
    data = sio.loadmat(f'data/{subject_index}.mat')['data']  # shape: (64, 1500, 40, 6)
    
    # Prepare the data for MNE: combining blocks and trials# channels = O1 at 60, O2 at 62 and Oz at 61, therefore:
    eeg_data = data
    eeg_data = eeg_data[:,:,0:5,:] # 3 channels, 1500 time points, five frequencies, 6 blocks 
    
    new_data = eeg_data.transpose(3,1,2,0) # 6 blocks, 1500 time points, five frequencies, 3 channels
    new_data = new_data.reshape(-1,5,n_channels) # 9000 time points, five frequenceis, 3 channels 
    
    new_data = new_data.transpose(1,0,2) # 5 frequencies, 9000 time points(1500*6) , 3 channels 
    new_data = new_data.reshape(-1,n_channels) # 45000 time points (1500*6*5), 3 channels 
    new_data = new_data.transpose(1,0) # now it is ready for raw 
    new_data = new_data*1e-6
    
    raw = mne.io.RawArray(new_data, info)
    if (i==1):
        #create events array
        event_ids = {f'Freq_{i+8}': i+1 for i in range(5)}
        event1  = mne.make_fixed_length_events(raw,id =1, start = 0, stop = 36, duration = 6)
        event2  = mne.make_fixed_length_events(raw,id =2, start = 36, stop = 36*2, duration = 6)
        event3  = mne.make_fixed_length_events(raw,id =3, start = 36*2, stop = 36*3, duration = 6)
        event4  = mne.make_fixed_length_events(raw,id =4, start = 36*3, stop = 36*4, duration = 6)
        event5  = mne.make_fixed_length_events(raw,id =5, start = 36*4, stop = 36*5, duration = 6)
        events = np.vstack((event1, event2, event3, event4, event5))

    epochs = mne.Epochs(raw, events, event_ids, tmin=0, tmax=5.995,baseline=(0, 0), preload=True)
    # epochs_file_path = f'S{i}-epo.fif'
    all_epochs.append(epochs)

# Concatenate all the epochs into a single Epochs object
combined_epochs = mne.concatenate_epochs(all_epochs)


# Save the combined epochs to a new file
combined_epochs_file_path = 'all_subjects-epo.fif'
combined_epochs.save(combined_epochs_file_path, overwrite=True)
