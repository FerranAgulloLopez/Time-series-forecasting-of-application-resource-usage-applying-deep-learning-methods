#!/bin/sh

# variables initialization
input_directory_path="../input/configs"
output_directory_path="../output"
temporary_launcher_path="./temporary_launcher_file"

# loop by all config files of the input directory and run all
file_names=$(find $input_directory_path/* -iname "*.json")
for file_name in $file_names; do
    base_name=$(basename "$file_name")
    file_name=${base_name%.*}  # no extension
    echo "$file_name"

    # create output directory for the specific job
    mkdir -p ${output_directory_path}/"$file_name"

    # create temporary launcher file
    {
      echo "#!/bin/bash"
      echo "#SBATCH --job-name=${file_name}"
      echo "#SBATCH --qos=bsc_cs"
      echo "#SBATCH -D ./"
      echo "#SBATCH --ntasks=1"
      echo "#SBATCH --output=${output_directory_path}/${file_name}/log_%j.out"
      echo "#SBATCH --error=${output_directory_path}/${file_name}/log_%j.err"
      echo "#SBATCH --cpus-per-task=160"
      echo "#SBATCH --gres gpu:4"
      echo "#SBATCH --time=00:05:00"
      echo "module purge; module load cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 gcc/8.3.0 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML"
      echo "PYTHONUNBUFFERED=false PYTHONPATH=$PYTHONPATH:/.. python3 ./app/train_test_model.py --config ${input_directory_path}/${file_name}.json --output ${output_directory_path}/${file_name}"
    } > "$temporary_launcher_path"

    # run job
    sbatch "$temporary_launcher_path"

done

# remove temporary launcher file
rm ${temporary_launcher_path}

