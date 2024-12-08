import argparse
import numpy as np
import os  # To check and create directories

from utils.utils import run_class_time_CV_fmri_crossval_ridge

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--nlp_feat_type", required=True)
    parser.add_argument("--nlp_feat_dir", required=True)
    parser.add_argument("--layer", type=int, required=False)
    parser.add_argument("--sequence_length", type=int, required=False)
    parser.add_argument("--output_dir", required=True)
    
    args = parser.parse_args()
    print(args)
        
    predict_feat_dict = {'nlp_feat_type': args.nlp_feat_type,
                         'nlp_feat_dir': args.nlp_feat_dir,
                         'layer': args.layer,
                         'seq_len': args.sequence_length}

    # Construct the correct fMRI file path with updated subject format
    fmri_file = f'./data/fMRI/sub-{args.subject}_ses-2015_T1w.npy'  # Adjusted file path to match subject format

    if not os.path.exists(fmri_file):
        raise FileNotFoundError(f"fMRI data file not found for subject {args.subject}: {fmri_file}")

    print(f"Loading fMRI data for subject {args.subject}...")
    data = np.load(fmri_file)
    print(f"Shape of fMRI data: {data.shape}")
    
    # Running the prediction function
    corrs_t, _, _, preds_t, test_t = run_class_time_CV_fmri_crossval_ridge(data,
                                                                predict_feat_dict)

    # Prepare the output file name
    fname = 'predict_{}_with_{}_layer_{}_len_{}'.format(args.subject, args.nlp_feat_type, args.layer, args.sequence_length)
    output_path = os.path.join(args.output_dir, fname + '.npy')  # Create the full path

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)  # Create the output directory if it doesn't exist
    
    # Print where the file is being saved
    print(f'Saving: {output_path}')

    # Save the results as a .npy file in the output directory
    np.save(output_path, {'corrs_t': corrs_t, 'preds_t': preds_t, 'test_t': test_t})
