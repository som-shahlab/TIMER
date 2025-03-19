# Activate Conda environment
CONDA_DIR="/local-scratch/nigam/users/$USER/miniconda3"
source $CONDA_DIR/etc/profile.d/conda.sh
COND_ENV="synth-instruct"
conda activate "${COND_ENV}"

# Verify Conda environment
echo "Using Conda environment: ${COND_ENV}"


EXPERIMENT="random_100_visit_evidence_pid_stratified_general"

DATE=$(date +%Y%m%d)
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
DATASET="ehr_dataset"
CONTEXT_LENGTH=16348
OUTPUT_DIR="/share/pi/nigam/users/$USER/synth-instruct/models"
EHR_DATA_PATH="/share/pi/nigam/users/$USER/synth-instruct/synth_output/same_personids/alpaca/${EXPERIMENT}/ehr_data.json"
RESULT_DIR="/share/pi/nigam/users/$USER/synth-instruct/result"
mkdir -p "${RESULT_DIR}/${DATE}"
DATA_SIZE=1000


# --from_peft_checkpoint /share/pi/nigam/users/aunell/synth-instruct/models/20241209_general_general_visit_evidence_test/best_model \
echo "Running training script..."
python /share/pi/nigam/users/aunell/timer-private/instruct_tune/tune_llama_recipes.py \
    --use_peft \
    --peft_method lora \
    --dataset "${DATASET}" \
    --quantization \
    --use_fp16 \
    --model_name "${MODEL_NAME}" \
    --context_length="${CONTEXT_LENGTH}" \
    --grad_accumulation_steps=16 \
    --lr=1e-5 \
    --wd=1e-4 \
    --batch_size_training=1 \
    --num_epochs=3 \
    --ehr_dataset.data_path "${EHR_DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}/${DATE}_${EXPERIMENT}_1000_MISTRAL_${DATA_SIZE}" \
    --dataset_size "${DATA_SIZE}" 
