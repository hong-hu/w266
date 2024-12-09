#!/bin/bash

PARAMS=("$@")

# Get the operating system name
os_name=$(uname -s)

# Check if the OS is Windows (Cygwin or WSL)
if [ "$os_name" == MSYS_NT* ]; then
  ROOT_PATH="C:/Users/O772985/OneDrive - JPMorgan Chase/MSDE/tmp/data_backup"
  PROJ_PATH="C:/Users/O772985/OneDrive - JPMorgan Chase/MEGA/MyProjects/study/python/aiml/pytorch/ner/nlp"
elif [ "$os_name" == "Linux" ]; then
  ROOT_PATH="/home/omniai-jupyter/nlp/pyanalysis/data_backup"
  PROJ_PATH="/home/omniai-jupyter/nlp"
else
  ROOT_PATH="/Users/honghu/Doc/w266/Project/data_backup"
  PROJ_PATH="/Users/honghu/MEGA/MyProjects/study/python/aiml/pytorch/ner/nlp"
fi

# If no datasets are provided, default to "small"
if [ ${#PARAMS[@]} -eq 0 ]; then
  DATASETS=("small")
else
  DATASETS=("${PARAMS[@]}")
fi

# Define the path to the JSON file
json_file="$PROJ_PATH/params.json"

# Extract the value of "is_bert" using grep and sed
embedding=$(grep -o '"embedding": *"[^"]*"' "$json_file" | sed -E 's/"embedding": *"([^"]*)"/\1/')
model=$(grep -o '"model": *"[^"]*"' "$json_file" | sed -E 's/"model": *"([^"]*)"/\1/')
pre_trained_tar=$(grep "pre_trained_tar" $json_file | sed 's/.*"pre_trained_tar": *//; s/[",]//g; s/^[ \t]*//; s/[ \t]*$//')
learning_rate=$(grep '"learning_rate":' "$json_file" | sed 's/.*: //; s/,//')
nhead=$(grep '"nhead":' "$json_file" | sed 's/.*: //; s/,//')
num_layers=$(grep '"num_layers":' "$json_file" | sed 's/.*: //; s/,//')

# Check if is_bert is equal to "true"
if [ "$embedding" == "vocab" ]; then
  EMBEDDING="vocab"
else
  EMBEDDING="$embedding"
fi

# Check if is_attention is equal to "true"
if [ "$model" == "lstm" ]; then
  MODEL="lstm"
else
  MODEL="$model"
fi

if [[ $pre_trained_tar == *.tar ]]; then
    DIR_NAME="transfer"
else
    DIR_NAME="train"
fi


if [ ${#PARAMS[@]} -eq 6 ]; then
  DATASETS=("${PARAMS[0]}")
  EMBEDDINGS=("${PARAMS[1]}")
  MODELS=("${PARAMS[2]}")
  LSTM_LRS=("${PARAMS[3]}")
  TRAN_LRS=("${PARAMS[3]}")
  BERT_LRS=("${PARAMS[3]}")
  FASTTEST_HEADS=("${PARAMS[4]}")
  OTHER_HEADS=("${PARAMS[4]}")
  LAYERS=("${PARAMS[5]}")
#  echo "${PARAMS[0]} ${PARAMS[1]} ${PARAMS[2]} ${PARAMS[3]} ${PARAMS[4]} ${PARAMS[5]}"
else
  EMBEDDINGS=("fasttext")  # ("uncased" "fasttext" "vocab")
  MODELS=("lstm")  # ("bert" "lstm" "transformer")
  LSTM_LRS=(0.01)  # (0.01 0.001)
  TRAN_LRS=(0.0001)  # (0.0001 0.00001)
  BERT_LRS=(0.00005)  # (0.00005 0.00001)
  FASTTEST_HEADS=(4)  # (4 10)
  OTHER_HEADS=(4)  # (4 8)
  LAYERS=(1)  # (1 2 4 8 16)

#  EMBEDDINGS=("$EMBEDDING")
#  MODELS=("$MODEL")
#  LSTM_LRS=("$learning_rate")
#  TRAN_LRS=("$learning_rate")
#  BERT_LRS=("$learning_rate")
#  FASTTEST_HEADS=("$nhead")
#  OTHER_HEADS=("$nhead")
#  LAYERS=("$num_layers")
fi

# Iterate through each dataset provided
for DATASET in "${DATASETS[@]}"; do
  for EMBEDDING in "${EMBEDDINGS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
      if [[ "$EMBEDDING" == *cased ]] && [[ "$MODEL" != "bert" ]]; then
#        echo "$EMBEDDING embedding is only associated to Bert model. Skipping..."
        continue  # Skip to the next iteration
      fi
      if [[ "$EMBEDDING" != *cased ]] && [[ "$MODEL" == "bert" ]]; then
#        echo "Bert model is only associated to $EMBEDDING embedding. Skipping..."
        continue  # Skip to the next iteration
      fi
      if [ "$MODEL" == "transformer" ]; then
        LRS=("${TRAN_LRS[@]}")
      elif [ "$MODEL" == "bert" ]; then
        LRS=("${BERT_LRS[@]}")
      else
        LRS=("${LSTM_LRS[@]}")
      fi
      for LR in "${LRS[@]}"; do
        if [ "$MODEL" == "transformer" ]; then
          if [ "$EMBEDDING" == "fasttext" ]; then
            HEADS=("${FASTTEST_HEADS[@]}")
          else
            HEADS=("${OTHER_HEADS[@]}")
          fi
          for HEAD in "${HEADS[@]}"; do
            for LAYER in "${LAYERS[@]}"; do
              COMBINE_DIR="${EMBEDDING}_${MODEL}_${LR}_${HEAD}x${LAYER}"
              MODEL_DIR="${ROOT_PATH}/${DIR_NAME}/${DATASET}/${COMBINE_DIR}"

              # Check if the directory already exists
              if [ -d "$MODEL_DIR" ]; then
                echo "Directory $MODEL_DIR already exists. Skipping..."
                continue  # Skip to the next iteration
              fi

              echo "------------------------------------------------------------------------------------------------------------------------"
              mkdir -p $MODEL_DIR
#              echo "cp ${PROJ_PATH}/params.json $MODEL_DIR"
              cp "${PROJ_PATH}/params.json" "$MODEL_DIR"
              echo "python train.py --data_dir ${PROJ_PATH}/data/${DATASET} --model_dir $MODEL_DIR --embedding $EMBEDDING --model $MODEL --learning_rate $LR --nhead $HEAD --num_layers $LAYER"
              python train.py --data_dir "${PROJ_PATH}/data/${DATASET}" --model_dir $MODEL_DIR --embedding $EMBEDDING --model $MODEL --learning_rate $LR --nhead $HEAD --num_layers $LAYER
              echo "python evaluate.py --data_dir data/${DATASET} --model_dir $MODEL_DIR --embedding $EMBEDDING --model $MODEL --learning_rate $LR --nhead $HEAD --num_layers $LAYER"
              python evaluate.py --data_dir "${PROJ_PATH}/data/${DATASET}" --model_dir $MODEL_DIR --embedding $EMBEDDING --model $MODEL --learning_rate $LR --nhead $HEAD --num_layers $LAYER
            done
          done
        else
          COMBINE_DIR="${EMBEDDING}_${MODEL}_${LR}"
          MODEL_DIR="${ROOT_PATH}/${DIR_NAME}/${DATASET}/${COMBINE_DIR}"

          # Check if the directory already exists
          if [ -d "$MODEL_DIR" ]; then
            echo "Directory $MODEL_DIR already exists. Skipping..."
            continue  # Skip to the next iteration
          fi

          echo "------------------------------------------------------------------------------------------------------------------------"
          mkdir -p $MODEL_DIR
#          echo "cp ${PROJ_PATH}/params.json $MODEL_DIR"
          cp "${PROJ_PATH}/params.json" "$MODEL_DIR"
          echo "python train.py --data_dir ${PROJ_PATH}/data/${DATASET} --model_dir $MODEL_DIR --embedding $EMBEDDING --model $MODEL --learning_rate $LR"
          python train.py --data_dir "${PROJ_PATH}/data/${DATASET}" --model_dir $MODEL_DIR --embedding $EMBEDDING --model $MODEL --learning_rate $LR
          echo "evaluate.py --data_dir ${PROJ_PATH}/data/${DATASET} --model_dir $MODEL_DIR --embedding $EMBEDDING --model $MODEL --learning_rate $LR"
          python evaluate.py --data_dir "${PROJ_PATH}/data/${DATASET}" --model_dir $MODEL_DIR --embedding $EMBEDDING --model $MODEL --learning_rate $LR
        fi
      done
    done
  done
done