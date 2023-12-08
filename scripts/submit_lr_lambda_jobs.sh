#!/bin/bash


# paths
padam_path='/full/path/to/PAdam'



# Array of values for max_lr and lambda_p
max_lr_values=(4e-3 2e-3 1e-3 5e-4 2e-4)  # Example values
lambda_p_values=(4e-3 2e-3 1e-3 5e-4 2e-4)  # Example values

# Loop over the parameter values
for max_lr in "${max_lr_values[@]}"
do
    for lambda_p in "${lambda_p_values[@]}"
    do
        # Format max_lr and lambda_p for directory names
        # Replacing '.' with 'p' to denote decimal point
        formatted_max_lr=$(echo "$max_lr" | sed 's/\./p/g')
        formatted_lambda_p=$(echo "$lambda_p" | sed 's/\./p/g')

        # Create a Slurm script for this combination of parameters
        cat << EOF > temp_slurm_script_${formatted_max_lr}_${formatted_lambda_p}.sh
        #!/bin/bash
        #SBATCH --job-name=python_job_${formatted_max_lr}_${formatted_lambda_p}
        #SBATCH --output=result_${formatted_max_lr}_${formatted_lambda_p}_%j.out
        #SBATCH --gres=gpu:1
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --time=02:00:00
        #SBATCH --mem=4GB

        # Run the Python script with the current combination of parameters
        python $padam_path/python/train_CIFAR.py --max_lr=$max_lr --lambda_p=$lambda_p --relative_paths=True --out_dir='../results/PAdam_lr_wd_scan/lr_${formatted_max_lr}_wd_${formatted_lambda_p}'
EOF
        # Submit the job to Slurm
        sbatch temp_slurm_script_${formatted_max_lr}_${formatted_lambda_p}.sh

        # Optionally, remove the temporary Slurm script after submission
        rm temp_slurm_script_${formatted_max_lr}_${formatted_lambda_p}.sh
    done
done
