# NOTE: prepare dataset yourself using create_dataset.py before running this script
root_dir=$1
synthesis_dir=$2
vocoder_dir=$3
synthesiser_checkpoint_path=$4
vocoder_checkpoint_path=$5
type=test
inference_only=${INFERENCE_ONLY:-1}
detect_landmark=${DETECT_LANDMARK:-0}
align_mouth=${ALIGN_MOUTH:-0}

if [[ $inference_only -ne 1 ]]; then
    echo Generating file list...
    python create_dataset.py $root_dir create $type generate_file_list

    # extract mouth frames
    echo Extracting mouth frames...
    TYPE=$type DETECT_LANDMARK=$detect_landmark ALIGN_MOUTH=$align_mouth ./extract_mouth_frames.sh $root_dir

    # create manifests for extracting speech units
    echo Creating manifests...
    python create_dataset.py $root_dir create $type manifests

    # extract speech units
    echo Extracting speech units...
    TYPE=test ./extract_speech_units.sh $root_dir
fi

# run synthesis
echo Running synthesis...
./synthesise.sh $root_dir/label $synthesis_dir $synthesiser_checkpoint_path

# setup vocoder
echo Setting up vocoder...
python create_dataset.py $root_dir create $type vocoder $synthesis_dir

# run vocoder
echo Running vocoder...
./vocoder.sh $synthesis_dir/vocoder $vocoder_dir $vocoder_checkpoint_path
