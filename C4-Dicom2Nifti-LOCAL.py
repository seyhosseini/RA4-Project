#! pip install dicom2nifti
#! pip install glob
import dicom2nifti
import os
import glob

#Assumes dicoms are within the present working directory (Where python script is run):
#dicom_folders = os.path.dirname(os.path.realpath(__file__))

#-----

#Assumes parent directory to all dicom directories is supplied by the user:
dicom_folders = input("Path to dicom parent directory: ")

#-----

#Convert dicom to nii.gz (specify output directory)
for root, dirs, files in os.walk(dicom_folders):
    for current in dirs:

        #Stores path to directory to store converted .nii.gz file
        current_dir = os.path.join(root, current)

        #Initializes value to replace converted .nii.gz file name
        renamingFile = current + '.nii.gz'
        renamingFile2 = dicom_folders + '\\' + renamingFile

        #If the dicom dataset has already been converted & renamed, it will be skipped and look at the next directory
        if not os.path.exists(renamingFile2):
            # Convert dicoms to .nii.gz - compression=True allows for the .gz compression
            # 'reorient' will update the converted dataset to be LAS oriented if set to 'True'
            dicom2nifti.convert_directory(current_dir, dicom_folders, compression=True, reorient=False)
        else:
            continue

        #Return list of all files in parent directory
        list_of_files = glob.glob(dicom_folders + '\*')  # * means all if need specific format then *.csv

        #Return the full path of the most recently created file in parent directory to find newly converted .nii.gz file
        latest_file = str(max(list_of_files, key=os.path.getctime))

        #Changes default file name of most recently converted .nii.gz dataset based on name of the dicom's directory
        os.rename(latest_file, renamingFile2)
        print('Dicom dataset --- ' + current + ' --- converted to .nii.gz')
#-----