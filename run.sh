num_runs=2
set -e
echo "Running gan.py"
python gan_diff.py

echo "Running AMP_Classification.py"
python AMP_Classification.py

echo "Running gan_diff.py"
python gan_diff.py


for ((i = 1; i <= $num_runs; i++)); do
  echo "Iteration $i of $num_runs:"
  # 生成AMP序列
  echo "Running gan_generate.py"
  python gan_generate.py

  echo "Running AMP-Classification_Prediction.py"
  python AMP_Classification_Prediction.py

  echo "Running generate_pos.py"
  python generate_pos.py

  echo "Running predict.py"
  python predict.py -test_fasta_file examples/samples.fasta -output_file_name prediction_results

  echo "Running attribute_selection.py"
  python attribute_selection.py -i prediction_results.csv -o selected_data.csv -c AMP antibacterial -v Yes No

  echo "Running gan_update.py"
  python gan_update.py

done

echo "All programs have been executed."
echo "enter to quit "
sleep 500
